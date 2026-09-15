
const REGISTER_SIZE = register_size()

# This is here in case kernel chain needs to be run 
# but is not fully filled. It does nothing
blank_kernel!(args...) = nothing

@kwdef struct KernelCall{K,A}
  kernel::K = blank_kernel!
  args::A   = ()
end

function make_kernel_call(kernel=blank_kernel!, args=())
  _args = map(t->time_lower(batch_lower(t)), args)
  return KernelCall(kernel, _args)
end

num_lower(::Type, t) = t
num_lower(::Type{T}, t::Float64) where {T<:Union{Float32,Float16}} = T(t)
num_lower(::Type{T}, t::SArray{S,Float64}) where {T<:Union{Float32,Float16},S} = T.(t)
# Index each tuple member directly so nested field parameters remain inferable
# on Julia 1.10, where recursive map calls can widen the conversion type.
@generated function num_lower(::Type{T}, t::S) where {T<:Union{Float32,Float16},S<:Tuple}
  return Expr(:tuple, [:(num_lower(T, getfield(t, $i))) for i in 1:fieldcount(S)]...)
end
@inline num_lower(::Type{T}, nt::NamedTuple{names}) where {T<:Union{Float32,Float16},names} =
  NamedTuple{names}(num_lower(T, Tuple(nt)))
function num_lower(::Type{T}, tf::TimeFunction) where {T<:Union{Float32,Float16}}
  S = typeof(tf(0))
  if S != T
    error("
      Failure to lower TimeFunction with output type $S to $T: if you are ramping
      the reference energy, for $T support you will need to specify a ramping 
      TimeFunction of `p_over_q_ref` that outputs $T.
    ")
  end
  return tf
end

function num_lower(::Type{T}, b::BatchParam) where {T<:Union{Float32,Float16}}
  S = eltype(b)
  if S != T
    return BatchParam(T.(b.batch))
  else
    return b
  end
end

# Kernel calls lower batch wrappers before numeric conversion in _push.
@inline function num_lower(::Type{T}, b::_LoweredBatchParam{N}) where {T<:Union{Float32,Float16},N}
  return eltype(b.batch) == T ? b : _LoweredBatchParam{N}(T.(b.batch))
end

# In case KernelCall contains batch GPU array
Adapt.@adapt_structure KernelCall

# Store the state of the reference coordinate system
struct RefState{R,S,T,U,V,W,X,Y}
  t_enter::S          # Reference time at entrance
  beta_gamma_enter::T # Reference energy at entrance
  t_exit::U           # Reference time at exit
  beta_gamma_exit::V  # Reference energy at exit
  L::W                
  g::X                
  ds_step::Y    
  function RefState{R}(t_enter::S, beta_gamma_enter::T, t_exit::U, beta_gamma_exit::V, L::W, g::X, ds_step::Y) where {R,S,T,U,V,W,X,Y}
    return new{R,S,T,U,V,W,X,Y}(t_enter, beta_gamma_enter, t_exit, beta_gamma_exit, L, g, ds_step)
  end
end

function RefState{T}(; t_enter, beta_gamma_enter, t_exit=t_enter, beta_gamma_exit=beta_gamma_enter, L=0, g=(0,0), ds_step=0) where {T}
  return RefState{T}(batch_lower(t_enter), batch_lower(beta_gamma_enter), batch_lower(t_exit), batch_lower(beta_gamma_exit), batch_lower(L), batch_lower(g), batch_lower(ds_step))
end

coordstype(::RefState{T}) where {T} = T

@inline function beval(ref::RefState{T}, i) where {T}
  return RefState{T}(beval(ref.t_enter, i), beval(ref.beta_gamma_enter, i), beval(ref.t_exit, i), beval(ref.beta_gamma_exit, i), beval(ref.L, i), beval(ref.g, i), beval(ref.ds_step, i))
end

# Alias
struct KernelChain{C<:Tuple{Vararg{<:KernelCall}}, S<:RefState, TOUT<:Tuple{Vararg{<:KernelCall}}, TIN<:Tuple{Vararg{<:KernelCall}}}
  chain::C  # The tuple of KernelCalls
  ref::S    # A RefState
  transforms_out::TOUT
  transforms_in::TIN
  function KernelChain(chain, ref, transforms_out=ntuple(t->KernelCall(), Val{3}()), transforms_in=ntuple(t->KernelCall(), Val{3}()))
    new{typeof(chain), typeof(ref), typeof(transforms_out), typeof(transforms_in)}(chain, ref, transforms_out, transforms_in)
  end
end

# In case KernelChain contains batch GPU array
Adapt.@adapt_structure KernelChain

KernelChain(::Val{N}, ref, transforms_out=ntuple(t->KernelCall(), Val{3}()), transforms_in=ntuple(t->KernelCall(), Val{3}())) where {N} = KernelChain(ntuple(t->KernelCall(), Val{N}()), ref, transforms_out, transforms_in)

push(kc::KernelChain, kcall::Nothing) = kc
push_transforms_out(kc::KernelChain, tout::Nothing) = kc
push_transforms_in(kc::KernelChain, tin::Nothing) = kc

push(kc::KernelChain, kcall) = @reset kc.chain = _push(kc.chain, kcall, coordstype(kc.ref))
push_transforms_out(kc::KernelChain, tout) = @reset kc.transforms_out = _push(kc.transforms_out, tout, coordstype(kc.ref))
push_transforms_in(kc::KernelChain, tin) = @reset kc.transforms_in = _push(kc.transforms_in, tin, coordstype(kc.ref))

function _push(chain, kcall, T)
  return __push(chain, KernelCall(kcall.kernel, num_lower(T, kcall.args)))
end

@unroll function __push(chain, kcall)
  i = 0
  @unroll for kcalli in chain
    i += 1
    if kcalli.kernel == blank_kernel!
      return @reset chain[i] = kcall
    end
  end
  error("Unable to push KernelCall to kernel chain: kernel chain is full")
end

# KA does not like Vararg
@kernel function generic_kernel!(coords::Coords, @Const(kc::KernelChain))
  i = @index(Global, Linear)
  @inline _generic_kernel!(i, coords, kc)
end

_generic_kernel!(i, coords, kc) = __generic_kernel!(i, coords, kc.chain, beval(kc.ref, i), kc.transforms_out, kc.transforms_in)

@generated function __generic_kernel!(i, coords, chain::T, ref, transforms_out, transforms_in) where {T}
  N = length(T.parameters)
  if N > 0 && first(T.parameters) <: KernelCall{typeof(reference_momentum_shift!),Tuple{<:Any,TimeFunction}}
    # Static check that everything is ok
    if last(T.parameters) <: KernelCall{typeof(reference_momentum_shift!),Tuple{TimeFunction,TimeFunction}}
      return :(__generic_kernel_ramp!(i, coords, chain, ref, transforms_out, transforms_in))
    else
      error("
        Kernels with time-dependent reference energies must start and end with `reference_momentum_shift!`,
        where the entering one transforms all particles to have individual reference momenta, and the exiting 
        transforms all particles to a uniform reference momentum at the end of the element equal to 
        p_over_q_ref(bunch.t_ref at end).
      ")
    end
  else
    return :(__generic_kernel_noramp!(i, coords, chain, ref, transforms_out, transforms_in))
  end
end

function __generic_kernel_noramp!(i, coords::Coords, chain, ref, transforms_out, transforms_in)
  body_callback = construct_main_callback(coords, transforms_out, transforms_in, ref.t_enter, ref.beta_gamma_enter, ref.ds_step, ref.g)
  body_coords = Coords(coords.state, coords.v, coords.q, coords.weight, body_callback)
  __generic_kernel_noramp_body!(i, body_coords, chain, ref.t_enter, ref.beta_gamma_enter)
  # note: t_ref only used by transforms, can pass 0 for t_ref_transform 
  # beta_gamma is like ds_step and g, passed to callback, so that can't be 0
  exit_callback = construct_main_callback(coords, (), (), 0, ref.beta_gamma_exit, ref.ds_step, ref.g)
  _execute_callbacks(i, coords, exit_callback, ref.L, ref.t_exit)
  return nothing
end


@unroll function __generic_kernel_noramp_body!(i, body_coords, chain, t_ref_enter, beta_gamma_ref_enter)
  @unroll for kcall in chain
    bargs = process_batch_args(i, kcall.args)
    args = process_time_args(i, body_coords, bargs, t_ref_enter, beta_gamma_ref_enter)
    (kcall.kernel)(i, body_coords, args...)
  end
end

# For ramping we need to do something special:
function __generic_kernel_ramp!(i, coords::Coords, chain, ref, transforms_out, transforms_in)
  @assert last(chain).kernel == reference_momentum_shift! 
  @assert last(chain).args[1] isa TimeFunction
  @assert last(chain).args[2] isa TimeFunction
  # Have to store each particles initial time:
  t_initial = compute_time(coords.v[i,ZI], coords.v[i,PZI], ref.t_enter, ref.beta_gamma_enter)
  @inline __generic_kernel_noramp!(i, coords, Base.front(chain), ref, transforms_out, transforms_in)
  # With initial particle's time we now know the dbeta_gamma to evaluate for the last function
  beta_gamma_in_ele = teval(last(chain).args[1], t_initial)
  dbeta_gamma_in_ele = teval(last(chain).args[2], t_initial)
  reference_momentum_shift!(i, coords, beta_gamma_in_ele, dbeta_gamma_in_ele, last(chain).args[3])
  # note: can pass 0 for t_ref_transform because that is not used now 
  # since transforms are empty tuples at the end (back in global frame)
  # However we do now give the user access to reference energy
  exit_callback = construct_main_callback(coords, (), (), 0, ref.beta_gamma_exit, ref.ds_step, ref.g)
  _execute_callbacks(i, coords, exit_callback, ref.L, ref.t_exit)
  return nothing
end

function process_batch_args(i, args)
  if static_batchcheck(args) 
    return beval(args, i)
  else
    return args
  end
end

function process_time_args(i, coords, args, t_ref, beta_gamma_ref)
  if static_timecheck(args) 
    let t = compute_time(coords.v[i,ZI], coords.v[i,PZI], t_ref, beta_gamma_ref)
      return teval(args, t)
    end
  else
    return args
  end
end

# Generic function to launch a kernel on the bunch coordinates matrix
# Matrix v should ALWAYS be in SoA whether for real or as a view via tranpose(v)
@inline function launch!(
  coords::Coords{<:Any,V},
  kc::KernelChain;
  groupsize::Union{Nothing,Integer}=nothing, #backend isa CPU ? floor(Int,REGISTER_SIZE/sizeof(eltype(v))) : 256 
  use_cpu_multithreading::Bool=false,
  use_KA::Bool=!(get_backend(coords.v) isa CPU && isnothing(groupsize)),
  use_explicit_SIMD::Bool=!use_KA #&& (@static VERSION < v"1.11" || Sys.ARCH != :aarch64) # Default to use explicit SIMD on CPU, excepts for Macs above LTS bc SIMD.jl bug
) where {V}
  v = coords.v
  N_particle = size(v, 1)

  if use_KA && use_explicit_SIMD
    error("Cannot use both KernelAbstractions (KA) and explicit SIMD")
  end
#=  
  if !use_KA && backend isa GPU
    error("For GPU parallelized kernel launching, KernelAbstractions (KA) must be used")
  end
=#
  if !use_KA
    if use_explicit_SIMD && V <: SIMD.FastContiguousArray && eltype(V) <: SIMD.ScalarTypes && pick_vector_width(eltype(V)) > 1 # do SIMD
      simd_lane_width = pick_vector_width(eltype(V))
      lane = SIMD.VecRange{Int(simd_lane_width)}(0)
      rmn = rem(N_particle, simd_lane_width)
      N_SIMD = N_particle - rmn
      if use_cpu_multithreading
        Threads.@threads for i in 1:simd_lane_width:N_SIMD
          @assert last(i) <= N_particle "Out of bounds!"  # Use last because SIMD.VecRange SIMD
          _generic_kernel!(lane+i, coords, kc)
        end
      else
        for i in 1:simd_lane_width:N_SIMD
          @assert last(i) <= N_particle "Out of bounds!"  # Use last because SIMD.VecRange SIMD
          _generic_kernel!(lane+i, coords, kc)
        end
      end
      # Do the remainder
      for i in N_SIMD+1:N_particle
        @assert last(i) <= N_particle "Out of bounds!"
        _generic_kernel!(i, coords, kc)
      end
    else
      if use_cpu_multithreading
        Threads.@threads for i in 1:N_particle
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel!(i, coords, kc)
        end
      else
        @simd for i in 1:N_particle
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel!(i, coords, kc)
        end
      end
    end
  else
    backend = get_backend(v)
    if isnothing(groupsize)
      kernel! = generic_kernel!(backend)
    else
      kernel! = generic_kernel!(backend, groupsize)
    end
    kernel!(coords, kc; ndrange=N_particle)
    KernelAbstractions.synchronize(backend)
  end
  return nothing
end

function check_kwargs(mac, kwargs...)
  valid_kwargs = [:(fastgtpsa)=>Bool, :(inbounds)=>Bool]
  for k in kwargs
    if Meta.isexpr(k, :(=))
      pk = Pair(k.args...)
      idx = findfirst(t->t==pk[1], map(t->t[1], valid_kwargs))
      if isnothing(idx)
        error("Unrecognized input to @$(mac) macro: $(pk[1])")
      elseif typeof(pk[2]) != valid_kwargs[idx][2]
        error("Type for keyword argument `$(pk[1])` must be `$(valid_kwargs[idx][2])`")
      end
    else
      error("Unrecognized input to @$(mac) macro: $k")
    end
  end
end

# Also allow launch! on single KernelCalls
@inline launch!(coords::Coords, kcall::KernelCall; kwargs...) = launch!(coords, KernelChain((kcall,), RefState{eltype(coords.v)}(0,0,0,0,0,0,0)); kwargs...)

macro makekernel(args...)
  kwargs = args[1:length(args)-1]
  fcn = last(args)

  fcn.head == :function || error("@makekernel must wrap a function definition")
  body = esc(fcn.args[2])
  signature = fcn.args[1].args

  fcn_name = esc(signature[1])
  args = esc.(signature[2:end])

  # Check if function body contains a return:
  MacroTools.postwalk(body) do x
    !(@capture(x, return _)) || error("Return statement not permitted in a kernel function $(signature[1])")
  end

  check_kwargs(:makekernel, kwargs...)
  kwargnames = map(t->t[1], map(t->Pair(t.args...), kwargs))
  kwargvals = map(t->t[2],map(t->Pair(t.args...), kwargs))

  idx_fastgtpsa = findfirst(t->t==:fastgtpsa, kwargnames)
  idx_inbounds = findfirst(t->t==:inbounds, kwargnames)

   if isnothing(idx_fastgtpsa) || !kwargvals[idx_fastgtpsa] # no fastgtpsa
    if isnothing(idx_inbounds) || kwargvals[idx_inbounds] # inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          @inbounds begin
            $(body)
          end
        end
      end
    else # no inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          $(body)
        end
      end
    end
  else # fastgtpsa
    if isnothing(idx_inbounds) || kwargvals[idx_inbounds] # inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          @inbounds begin @FastGTPSA begin
            $(body)
          end end
        end
      end
    else # no inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          @FastGTPSA begin
            $(body)
          end 
        end
      end

    end
  end

end
