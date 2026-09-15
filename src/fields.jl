"""
    EMField{T}

Electric and magnetic field vectors at one particle location. Electric field
components are in V/m and magnetic field components are in tesla.
"""
struct EMField{T}
  E::SVector{3,T}
  B::SVector{3,T}
end

function EMField(E::SVector{3,TE}, B::SVector{3,TB}) where {TE,TB}
  Ep, Bp = promote(E, B)
  return EMField{eltype(Ep)}(Ep, Bp)
end

function EMField(Ex, Ey, Ez, Bx, By, Bz)
  Ex, Ey, Ez, Bx, By, Bz = promote(Ex, Ey, Ez, Bx, By, Bz)
  return EMField(SVector(Ex, Ey, Ez), SVector(Bx, By, Bz))
end

@inline Base.:+(a::EMField, b::EMField) = EMField(a.E + b.E, a.B + b.B)

"""
    ZeroField()

A callable electromagnetic field source which returns zero fields. Field
sources are called as `source(x, y, z, s)`.
"""
struct ZeroField end

@inline function (::ZeroField)(x, y, z, s)
  v = zero(x)
  zero_vector = SVector(v, v, v)
  return EMField(zero_vector, zero_vector)
end

"""
    MultipoleField(orders, normal, skew)

A callable static magnetic multipole field. `normal` and `skew` contain
non-integrated physical magnetic-field coefficients. Calling the source
returns magnetic fields in tesla. All three arguments must be `SVector`s.
Orders must be unique and ascending.
"""
struct MultipoleField{M,KN,KS}
  orders::M
  normal::KN
  skew::KS

  MultipoleField{M,KN,KS}(orders::M, normal::KN, skew::KS) where {M,KN,KS} =
    new{M,KN,KS}(orders, normal, skew)
end

function MultipoleField(
  orders::M,
  normal::KN,
  skew::KS,
) where {N,M<:SVector{N,<:Integer},KN<:SVector{N},KS<:SVector{N}}
  N > 0 || throw(ArgumentError("use ZeroField for an empty field source"))
  issorted(orders) || throw(ArgumentError("multipole orders must be ascending"))
  allunique(orders) || throw(ArgumentError("multipole orders must be unique"))
  return MultipoleField{M,KN,KS}(orders, normal, skew)
end

@inline function (source::MultipoleField)(x, y, z, s)
  bx, by = normalized_field(source.orders, source.normal, source.skew, x, y, 0)
  zero_field = zero(bx)
  bz = vifelse(source.orders[1] == 0, source.normal[1], zero_field)
  E = SVector(zero_field, zero_field, zero_field)
  return EMField(E, SVector(bx, by, bz))
end

"""
    FunctionalField(evaluator[, parameters])

A callable field source backed by a concrete evaluator. With parameters, the
evaluator is called as `evaluator(x, y, z, s, parameters)`. Without parameters,
it is called as `evaluator(x, y, z, s)`. The evaluator must return `EMField`.
"""
struct FunctionalField{F,P}
  evaluator::F
  parameters::P
end

FunctionalField(evaluator) = FunctionalField(evaluator, nothing)

@inline function (source::FunctionalField{F,Nothing})(x, y, z, s) where {F}
  return source.evaluator(x, y, z, s)
end

@inline function (source::FunctionalField)(x, y, z, s)
  return source.evaluator(x, y, z, s, source.parameters)
end

"""
    SumField(sources...)
    SumField(sources::Tuple)

A callable, statically dispatched sum of electromagnetic field sources.
Nested sums are flattened and `ZeroField` members are removed at construction.
"""
struct SumField{S<:Tuple}
  sources::S

  SumField{S}(sources::S) where {S<:Tuple} = new{S}(sources)
end

@inline _flatten_field_source(::ZeroField) = ()
@inline _flatten_field_source(source::SumField) = source.sources
@inline _flatten_field_source(source) = (source,)

@inline _flatten_field_sources(::Tuple{}) = ()
@inline function _flatten_field_sources(sources::Tuple)
  return (_flatten_field_source(first(sources))..., _flatten_field_sources(Base.tail(sources))...)
end

function SumField(sources::Tuple)
  flattened = _flatten_field_sources(sources)
  isempty(flattened) && return ZeroField()
  length(flattened) == 1 && return first(flattened)
  return SumField{typeof(flattened)}(flattened)
end

SumField(sources...) = SumField(sources)

@generated function _evaluate_field_sum(sources::S, x, y, z, s) where {S<:Tuple}
  N = length(S.parameters)
  N > 0 || return :(ZeroField()(x, y, z, s))
  expression = :(Base.getfield(sources, 1)(x, y, z, s))
  for i in 2:N
    expression = :($expression + Base.getfield(sources, $i)(x, y, z, s))
  end
  return expression
end

@inline function (source::SumField)(x, y, z, s)
  return _evaluate_field_sum(source.sources, x, y, z, s)
end

include("field_parameters.jl")

Adapt.@adapt_structure EMField
# Adaptation preserves the validated orders. Avoid rerunning constructor checks:
# KernelAbstractions also adapts arguments inside GPU kernels (constify).
@inline function Adapt.adapt_structure(to, source::MultipoleField)
  orders = Adapt.adapt(to, source.orders)
  normal = Adapt.adapt(to, source.normal)
  skew = Adapt.adapt(to, source.skew)
  return MultipoleField{typeof(orders),typeof(normal),typeof(skew)}(orders, normal, skew)
end
Adapt.@adapt_structure FunctionalField
Adapt.@adapt_structure SumField
