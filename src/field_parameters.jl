@inline function _rebuild_multipole_field(source::MultipoleField, normal, skew)
  return MultipoleField{typeof(source.orders),typeof(normal),typeof(skew)}(
    source.orders,
    normal,
    skew,
  )
end

@inline batch_lower(source::MultipoleField) =
  _rebuild_multipole_field(source, batch_lower(source.normal), batch_lower(source.skew))
@inline time_lower(source::MultipoleField) =
  _rebuild_multipole_field(source, time_lower(source.normal), time_lower(source.skew))
@inline num_lower(::Type{T}, source::MultipoleField) where {T<:Union{Float32,Float16}} =
  _rebuild_multipole_field(source, num_lower(T, source.normal), num_lower(T, source.skew))
@inline static_batchcheck(source::MultipoleField) =
  static_batchcheck(source.normal) || static_batchcheck(source.skew)
@inline static_timecheck(source::MultipoleField) =
  static_timecheck(source.normal) || static_timecheck(source.skew)
@inline beval(source::MultipoleField, i) =
  _rebuild_multipole_field(source, beval(source.normal, i), beval(source.skew, i))
@inline teval(source::MultipoleField, t) =
  _rebuild_multipole_field(source, teval(source.normal, t), teval(source.skew, t))

@inline _rebuild_functional_field(source::FunctionalField, parameters) =
  FunctionalField(source.evaluator, parameters)

@inline batch_lower(source::FunctionalField) =
  _rebuild_functional_field(source, batch_lower(source.parameters))
@inline time_lower(source::FunctionalField) =
  _rebuild_functional_field(source, time_lower(source.parameters))
@inline num_lower(::Type{T}, source::FunctionalField) where {T<:Union{Float32,Float16}} =
  _rebuild_functional_field(source, num_lower(T, source.parameters))
@inline static_batchcheck(source::FunctionalField) =
  static_batchcheck(source.parameters)
@inline static_timecheck(source::FunctionalField) =
  static_timecheck(source.parameters)
@inline beval(source::FunctionalField, i) =
  _rebuild_functional_field(source, beval(source.parameters, i))
@inline teval(source::FunctionalField, t) =
  _rebuild_functional_field(source, teval(source.parameters, t))

@inline _rebuild_sum_field(source::SumField, sources) =
  SumField{typeof(sources)}(sources)

@inline batch_lower(source::SumField) =
  _rebuild_sum_field(source, batch_lower(source.sources))
@inline time_lower(source::SumField) =
  _rebuild_sum_field(source, time_lower(source.sources))
@inline num_lower(::Type{T}, source::SumField) where {T<:Union{Float32,Float16}} =
  _rebuild_sum_field(source, num_lower(T, source.sources))
@inline static_batchcheck(source::SumField) =
  static_batchcheck(source.sources)
@inline static_timecheck(source::SumField) =
  static_timecheck(source.sources)
@inline beval(source::SumField, i) =
  _rebuild_sum_field(source, beval(source.sources, i))
@inline teval(source::SumField, t) =
  _rebuild_sum_field(source, teval(source.sources, t))
