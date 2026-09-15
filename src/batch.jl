#= 

BatchParam is very similar to TimeDependentParam, but instead 
of storing a function, it stores an arbitrary array of parameters.

Given a batch = [k1, k2, k3], batch parameters are seen by the particles as

Particle 1: k1
Particle 2: k2
Particle 3: k3
Particle 4: k1
Particle 5: k2
Particle 6: k3

etc.

Just like time, there are two types for batches - one type unstable 
generic wrapper for an AbstractArray/Number which is manipulated at the 
highest level, and a lowered representation where the numbers are made
numbers and arrays are kept but the length of the array is stored in 
the type. The reason for this is so that scalars are not unnecessarily
represented as large arrays, to save memory both outside and inside the 
kernel. The goal is that for large, batched simulations the type instability
of the UNPACKING step (simulation step is always type stable) is outweighted 
by the simulation step.

The lowered type does NOT have any arithmetic operations defined on it, as 
it should be untouched after lowering and in the kernel evaluated for each 
particle.

=#
struct BatchParam
  batch::Union{AbstractArray,Number}
  #=
    Currently, batches of TimeDependentParam are not supported
    It is essentially impossible on the GPU because accessing an 
    array of functions is not type stable (every particle has its 
    own function). The alternative - a time function which returns a
    big array - is also not possible for CPU-SIMD and would run into 
    memory problems on both CPU and GPU depending on how large that 
    static array is.
    
    On the CPU, the first option may be doable using FunctionWrappers,
    but that would require special handling because current Time
    uses bona-fide Julia functions for GPU compatibility.

    If one would like to do this right now, they should just start up 
    separate processes where each process has its own lattice with its 
    own TimeDependentParams.
  =#
  function BatchParam(batch::AbstractArray)
    if length(batch) == 1
      error("Cannot make BatchParam with array of length 1")
    end
    return new(batch)
  end
  BatchParam(n::Number) = new(n)
end

struct _LoweredBatchParam{N,V<:AbstractArray}
  batch::V
  _LoweredBatchParam(batch::AbstractArray) = new{length(batch),typeof(batch)}(batch)
  _LoweredBatchParam{N}(batch::AbstractArray) where {N} = new{N,typeof(batch)}(batch)
end

# Necessary for GPU compatibility if batch is a GPU array
function Adapt.adapt_structure(to, lbp::_LoweredBatchParam{N}) where {N}
    batch = Adapt.adapt_structure(to, lbp.batch)
    return _LoweredBatchParam{N}(batch)
end

# BatchParam will act like a number
# Conversion of types to BatchParam
BatchParam(a::BatchParam) = a

# Make these apply via convert
Base.convert(::Type{BatchParam}, a::Number) = BatchParam(a) # Scalar BatchParam
Base.convert(::Type{BatchParam}, a::BatchParam) = a
Base.eltype(b::BatchParam) = eltype(b.batch)

(::Type{T})(b::BatchParam) where {T<:Number} = BatchParam(T.(b.batch))

Base.zero(b::BatchParam) = BatchParam(zero(first(b.batch)))
Base.one(b::BatchParam)  = BatchParam(one(first(b.batch))) 

# Now define the math operations:
# The operations are individually-specialized for each operator, assuming that the 
# most expensive step is creating temporary arrays, not type instability. As such, 
# each are defined in a way as to minimize number of temporary arrays during unpacking,
# checking for e.g. 0's and 1's.
function _batch_addsub(batch_a, batch_b, op::T) where {T<:Union{typeof(+),typeof(-)}}
  if batch_a isa Number
    if batch_b isa Number
      return BatchParam(op(batch_a, batch_b))
    else
      if batch_a ≈ 0 # add/sub by zero gives identity
        return BatchParam(batch_b)
      else
        let a = batch_a
          return BatchParam(map((bi)->op(a, bi), batch_b))
        end
      end
    end
  elseif batch_b isa Number
    if batch_b ≈ 0 # add/sub by zero gives identity
      return BatchParam(batch_a)
    else
      let b = batch_b
        return BatchParam(map((ai)->op(ai, b), batch_a))
      end
    end
  elseif length(batch_a) == length(batch_b)
    return BatchParam(map((ai,bi)->op(ai, bi), batch_a, batch_b))
  else
    error("Cannot perform operation $(op) with two non-scalar BatchParams of differing 
            lengths (received lengths $(length(batch_a)) and $(length(batch_b))).")
  end
end

Base.:+(ba::BatchParam, n::Number)      = _batch_addsub(ba.batch, n, +)
Base.:+(n::Number, bb::BatchParam)      = _batch_addsub(n, bb.batch, +)
Base.:+(ba::BatchParam, bb::BatchParam) = _batch_addsub(ba.batch, bb.batch, +)

Base.:-(ba::BatchParam, n::Number)      = _batch_addsub(ba.batch, n, -)
Base.:-(n::Number, bb::BatchParam)      = _batch_addsub(n, bb.batch, -)
Base.:-(ba::BatchParam, bb::BatchParam) = _batch_addsub(ba.batch, bb.batch, -)

function _batch_mul(batch_a, batch_b)
  if batch_a isa Number
    if batch_b isa Number
      return BatchParam(*(batch_a, batch_b))
    else
      if batch_a ≈ 0 # mul by 0 gives 0 -> make scalar
        return BatchParam(0f0)
      elseif batch_a ≈ 1 # mul by 1 gives identity
        return BatchParam(batch_b)
      else
        let a = batch_a
          return BatchParam(map((bi)->*(a, bi), batch_b))
        end
      end
    end
  elseif batch_b isa Number
    if batch_b ≈ 0 # mul by 0 gives 0 -> make scalar
      return BatchParam(0f0)
    elseif batch_b ≈ 1 # mul by 1 gives identity
        return BatchParam(batch_a)
    else
      let b = batch_b
        return BatchParam(map((ai)->*(ai, b), batch_a))
      end
    end
  elseif length(batch_a) == length(batch_b)
    return BatchParam(map((ai,bi)->*(ai, bi), batch_a, batch_b))
  else
    error("Cannot perform operation * with two non-scalar BatchParams of differing 
            lengths (received lengths $(length(batch_a)) and $(length(batch_b))).")
  end
end

Base.:*(ba::BatchParam, n::Number)      = _batch_mul(ba.batch, n)
Base.:*(n::Number, bb::BatchParam)      = _batch_mul(n, bb.batch)
Base.:*(ba::BatchParam, bb::BatchParam) = _batch_mul(ba.batch, bb.batch)

function _batch_div(batch_a, batch_b)
  if batch_a isa Number
    if batch_b isa Number
      return BatchParam(/(batch_a, batch_b))
    else
      let a = batch_a
        return BatchParam(map((bi)->/(a, bi), batch_b))
      end
    end
  elseif batch_b isa Number
    if batch_b ≈ 0 # div by 0 gives Inf -> make scalar
      return BatchParam(Inf32)
    elseif batch_b ≈ 1 # div by 1 gives identity
        return BatchParam(batch_a)
    else
      let b = batch_b
        return BatchParam(map((ai)->/(ai, b), batch_a))
      end
    end
  elseif length(batch_a) == length(batch_b)
    return BatchParam(map((ai,bi)->/(ai, bi), batch_a, batch_b))
  else
    error("Cannot perform operation / with two non-scalar BatchParams of differing 
            lengths (received lengths $(length(batch_a)) and $(length(batch_b))).")
  end
end

Base.:/(ba::BatchParam, n::Number)      = _batch_div(ba.batch, n)
Base.:/(n::Number, bb::BatchParam)      = _batch_div(n, bb.batch)
Base.:/(ba::BatchParam, bb::BatchParam) = _batch_div(ba.batch, bb.batch)

# for now no special things for pow, unsure if called anywhere.
function _batch_pow(batch_a, batch_b)
  if batch_a isa Number
    if batch_b isa Number
      return BatchParam(^(batch_a, batch_b))
    else
      let a = batch_a
        return BatchParam(map((bi)->^(a, bi), batch_b))
      end
    end
  elseif batch_b isa Number
    let b = batch_b
      return BatchParam(map((ai)->^(ai, b), batch_a))
    end
  elseif length(batch_a) == length(batch_b)
    return BatchParam(map((ai,bi)->^(ai, bi), batch_a, batch_b))
  else
    error("Cannot perform operation ^ with two non-scalar BatchParams of differing 
            lengths (received lengths $(length(batch_a)) and $(length(batch_b))).")
  end
end

Base.:^(ba::BatchParam, n::Number)      = _batch_pow(ba.batch, n)
Base.:^(n::Number, bb::BatchParam)      = _batch_pow(n, bb.batch)
Base.:^(ba::BatchParam, bb::BatchParam) = _batch_pow(ba.batch, bb.batch)

function Base.literal_pow(::typeof(^), ba::BatchParam, ::Val{N}) where {N}
  return BatchParam(map(x->Base.literal_pow(^, x, Val{N}()), ba.batch))
end

atan2(bpa::BatchParam, bpb::BatchParam) = _batch_atan2(bpa.batch, bpb.batch)

function _batch_atan2(batch_a, batch_b)
  if batch_a isa Number
    if batch_b isa Number
      return BatchParam(atan2(batch_a, batch_b))
    else
      let a = batch_a
        return BatchParam(map((bi)->atan2(a, bi), batch_b))
      end
    end
  elseif batch_b isa Number
    let b = batch_b
      return BatchParam(map((ai)->atan2(ai, b), batch_a))
    end
  elseif length(batch_a) == length(batch_b)
    return BatchParam(map((ai,bi)->atan2(ai, bi), batch_a, batch_b))
  else
    error("Cannot perform operation ^ with two non-scalar BatchParams of differing 
            lengths (received lengths $(length(batch_a)) and $(length(batch_b))).")
  end
end

Base.:+(b::BatchParam) = b # identity

for t = (:-, :sqrt, :exp, :log, :sin, :cos, :tan, :cot, :sinh, :cosh, :tanh, :inv,
  :coth, :asin, :acos, :atan, :acot, :asinh, :acosh, :atanh, :acoth, :sinc, :csc, :float,
  :csch, :acsc, :acsch, :sec, :sech, :asec, :asech, :conj, :log10, :sign, :abs)
  @eval begin
    Base.$t(b::BatchParam) = BatchParam(map(x->($t)(x), b.batch))
  end
end

Base.isinf(b::BatchParam) = all(x->isinf(x), b.batch)
Base.isnan(b::BatchParam) = all(x->isnan(x), b.batch)


for t = (:unit, :sincu, :sinhc, :sinhcu, :asinc, :asincu, :asinhc, :asinhcu, :erf, 
         :erfc, :erfcx, :erfi, :wf, :rect)
  @eval begin
    GTPSA.$t(b::BatchParam) = BatchParam(map(x->($t)(x), b.batch))
  end
end


Base.promote_rule(::Type{BatchParam}, ::Type{U}) where {U<:Number} = BatchParam
Base.promote_rule(::Type{BatchParam}, ::Type{TimeDependentParam}) = error("Unable to combine BatchParams with TimeDependentParams")
Base.promote_rule(::Type{TimeDependentParam}, ::Type{BatchParam}) = error("Unable to combine BatchParams with TimeDependentParams")
Base.broadcastable(o::BatchParam) = Ref(o)

Base.isapprox(b1::BatchParam, b2::BatchParam; kwargs...) = all(isapprox.(b1.batch, b2.batch, kwargs...))
Base.isapprox(b::BatchParam, n::Number; kwargs...) = all(x->isapprox(x, n, kwargs...), b.batch)
Base.isapprox(n::Number, b::BatchParam; kwargs...) = all(x->isapprox(n, x, kwargs...), b.batch)
for t = (:(<), :(<=), :isless, :(==))
@eval begin

function Base.$t(b1::BatchParam, b2::BatchParam)
  return all(($t).(b1.batch, b2.batch))
end

function Base.$t(b1::BatchParam, a::Number)
  return all(x->($t)(x, a), b1.batch)
end

function Base.$t(a::Number, b1::BatchParam)
  return all(x->($t)(a, x), b1.batch)
end

end
end

batch_lower(bp) = bp

# Batch lowering should convert types to _LoweredBatchParam
function batch_lower(b::BatchParam)
  if b.batch isa AbstractArray
    return _LoweredBatchParam(b.batch) # Only arrays are lowered to batchparams
  else
    return b.batch
  end
end

# We can use map on the CPU, but not the GPU. This step of batch_lower-ing is on 
# the CPU and we are already type unstable here anyways, so we should do this.
batch_lower(bp::T) where {T<:Tuple} = map(bi->batch_lower(bi), bp)
@inline batch_lower(nt::NamedTuple{names}) where {names} =
  NamedTuple{names}(batch_lower(Tuple(nt)))

# Arrays MUST be converted into tuples, for SIMD
batch_lower(bp::SArray{N,BatchParam}) where {N} = batch_lower(Tuple(bp))
static_batchcheck(bp) = false
static_batchcheck(::_LoweredBatchParam) = true
@inline static_batchcheck(nt::NamedTuple) = static_batchcheck(Tuple(nt))
@unroll function static_batchcheck(t::Tuple)
  @unroll for ti in t
    if static_batchcheck(ti)
      return true
    end
  end
  return false
end

@inline beval(nt::NamedTuple{names}, i) where {names} =
  NamedTuple{names}(beval(Tuple(nt), i))

@inline beval(b::_LoweredBatchParam{B}, i) where {B} = b.batch[mod1(i, B)]

@inline function beval(b::_LoweredBatchParam{B}, lane::SIMD.VecRange{N}) where {B,N}
  @static if (VERSION < v"1.11" && Sys.ARCH == :x86_64)
    error("Julia's explicit SIMD.jl has a compiler bug that appears with batch 
           parameters on versions < 1.11 AND an x86_64 bit architecture, which we 
           detected that you have. To get around this, specify the `track!` 
           keyword argument `use_explicit_SIMD=false`")
  end
  m = rem(lane2vec(lane), B)
  i = vifelse(m == 0, B, m)
  return b.batch[i]
end

"""
    lane2vec(lane::SIMD.VecRange{N}) 
    
Given a SIMD.VecRange, will return an equivalent SIMD.Vec that
can be used in arithmetic operations for mapping integer indices
of particles to a given element in a batch.
"""
function lane2vec(lane::SIMD.VecRange{N}) where {N}
  # Try to match with vector register size, but 
  # only up to UInt32 -> ~4.3 billion particles, 
  # probably max on CPU...
  if Int(pick_vector_width(UInt32)) == N
    return SIMD.Vec{N,UInt32}(ntuple(i->lane.i+i-1, Val{N}()))
  else
    return SIMD.Vec{N,UInt64}(ntuple(i->lane.i+i-1, Val{N}()))
  end
end

# === THIS BLOCK WAS PARTIALLY WRITTEN BY CLAUDE ===
# Generated function for arbitrary-length tuples
@generated function beval(f::T, t) where {T<:Tuple}
  N = length(T.parameters)
  # Use getfield with literal integer arguments
  exprs = [:(beval(Base.getfield(f, $i), t)) for i in 1:N]
  return :(tuple($(exprs...)))
end
# === END CLAUDE ===

@inline beval(b, i) = b
