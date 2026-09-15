# RungeKutta uses the common unpacking, reference-ramp, alignment, aperture, and
# callback path. Only the body field integration is specific to RungeKutta.

@inline _unpack_field_parameter(value::NamedTuple{names}, context) where {names} =
  NamedTuple{names}(_unpack_field_parameter(Tuple(value), context))
@inline _unpack_field_parameter(value::Tuple, context) =
  map(item -> _unpack_field_parameter(item, context), value)
@inline _unpack_field_parameter(value::StaticArray, context) =
  map(item -> _unpack_field_parameter(item, context), value)
@inline _unpack_field_parameter(value, context) = deval(value, context)

@inline _scalarize_field_parameter(value::NamedTuple{names}) where {names} =
  NamedTuple{names}(_scalarize_field_parameter(Tuple(value)))
@inline _scalarize_field_parameter(value::Tuple) =
  map(_scalarize_field_parameter, value)
@inline _scalarize_field_parameter(value::StaticArray) =
  map(_scalarize_field_parameter, value)
@inline _scalarize_field_parameter(value) = scalarize(value)

@inline unpack_field_source(source, context) = source
@inline unpack_field_source(source::MultipoleField, context) =
  BeamTracking._rebuild_multipole_field(
    source,
    _unpack_field_parameter(source.normal, context),
    _unpack_field_parameter(source.skew, context),
  )
@inline unpack_field_source(source::FunctionalField, context) =
  BeamTracking._rebuild_functional_field(
    source,
    _unpack_field_parameter(source.parameters, context),
  )
@inline unpack_field_source(source::SumField, context) =
  BeamTracking._rebuild_sum_field(
    source,
    map(item -> unpack_field_source(item, context), source.sources),
  )

@inline scalarize_field_source(source) = source
@inline scalarize_field_source(source::MultipoleField) =
  BeamTracking._rebuild_multipole_field(
    source,
    _scalarize_field_parameter(source.normal),
    _scalarize_field_parameter(source.skew),
  )
@inline scalarize_field_source(source::FunctionalField) =
  BeamTracking._rebuild_functional_field(
    source,
    _scalarize_field_parameter(source.parameters),
  )
@inline scalarize_field_source(source::SumField) =
  BeamTracking._rebuild_sum_field(
    source,
    map(scalarize_field_source, source.sources),
  )

function unpack_runge_kutta(tm::RungeKutta, context, scalar_params)
  field = unpack_field_source(tm.field, context)
  additional_field = unpack_field_source(tm.additional_field, context)
  if scalar_params
    field = scalarize_field_source(field)
    additional_field = scalarize_field_source(additional_field)
  end
  return RungeKutta(tm.ds_step, tm.n_steps, field, additional_field)
end

@inline function runge_kutta_field(bmultipoleparams, L, p_over_q_ref)
  if !isactive(bmultipoleparams)
    return ZeroField()
  end

  mm = getfield(bmultipoleparams, :order)
  bn, bs = get_physical_strengths(bmultipoleparams, L, p_over_q_ref)
  if mm isa Integer
    return MultipoleField(SA[mm], SA[bn], SA[bs])
  end
  return MultipoleField(mm, bn, bs)
end

@inline function configured_runge_kutta_field(tm::RungeKutta, element_source)
  if !isnothing(tm.field)
    return tm.field
  elseif !isnothing(tm.additional_field)
    return SumField(element_source, tm.additional_field)
  end
  return element_source
end

@inline function runge_kutta_body(
  tm::RungeKutta,
  kc,
  p_over_q_ref,
  bunch,
  bendparams,
  bmultipoleparams,
  patchparams,
  rfparams,
  mapparams,
  fourpotentialparams,
  emultipoleparams,
  L,
)
  L > 0 || error("RungeKutta tracking requires a positive element length")
  !isactive(patchparams) || error("RungeKutta tracking does not support patch elements")
  !isactive(rfparams) || error("RungeKutta tracking does not support RF fields")
  !isactive(mapparams) || error("RungeKutta tracking does not support map elements")
  !isactive(fourpotentialparams) || error("RungeKutta tracking does not support FourPotentialParams")
  !isactive(emultipoleparams) || error("RungeKutta tracking does not support electric multipoles")

  if isactive(bendparams)
    (bendparams.edge1_int == 0 && bendparams.edge2_int == 0) ||
      error("edge1_int and edge2_int not yet handled for tracking")
    (bendparams.e1 == 0 && bendparams.e2 == 0) ||
      error("RungeKutta tracking does not support nonzero bend edge angles e1 or e2 because fringe tracking is not implemented")
    g_ref = bendparams.g_ref
    tilt_ref = bendparams.tilt_ref
    gx = g_ref * cos(tilt_ref)
    gy = g_ref * sin(tilt_ref)
  else
    gx = zero(L)
    gy = zero(L)
  end

  species = bunch.species
  tilde_m, _, beta_0 = BeamTracking.drift_params(species, p_over_q_ref)
  charge = chargeof(species)
  p0c = BeamTracking.R_to_pc(species, p_over_q_ref)
  mc2 = massof(species)
  n_steps, ds_step = BeamTracking.find_steps(tm, L)
  element_source = runge_kutta_field(bmultipoleparams, L, p_over_q_ref)
  source = configured_runge_kutta_field(tm, element_source)

  # Time-dependent values in params are evaluated once, at the particle's
  # element-entrance time, by the common kernel path. They stay fixed during
  # all RK substeps.
  params = (beta_0, tilde_m, charge, p0c, mc2, L, ds_step, n_steps,
            gx, gy, source)
  return push(kc, make_kernel_call(BeamTracking.RungeKuttaTracking.rk4_kernel!, params))
end
