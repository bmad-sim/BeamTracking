# =========== HELPER FUNCTIONS ============= #
@inline function integration_launcher(ker, params, photon_params, tm, edge_params, L)
  order = tm.order
  ds_step = tm.ds_step
  n_steps = tm.n_steps
  if ds_step < 0
    ds_step = L / n_steps
  else
    n_steps = Int(ceil(L / ds_step))
    ds_step = L / n_steps
  end
  fin  = fringe_in(tm.fringe_at)
  fout = fringe_out(tm.fringe_at)
  if order == 2
    return make_kernel_call(BeamTracking.order_two_integrator!, (ker, params, photon_params, ds_step, n_steps, edge_params, fin, fout, Val{tm.use_optimized_schemes}(), L))
  elseif order == 4
    return make_kernel_call(BeamTracking.order_four_integrator!, (ker, params, photon_params, ds_step, n_steps, edge_params, fin, fout, Val{tm.use_optimized_schemes}(), L))
  elseif order == 6
    return make_kernel_call(BeamTracking.order_six_integrator!, (ker, params, photon_params, ds_step, n_steps, edge_params, fin, fout, Val{tm.use_optimized_schemes}(), L))
  elseif order == 8
    return make_kernel_call(BeamTracking.order_eight_integrator!, (ker, params, photon_params, ds_step, n_steps, edge_params, fin, fout, Val{tm.use_optimized_schemes}(), L))
  elseif order == 10
    return make_kernel_call(BeamTracking.order_ten_integrator!, (ker, params, photon_params, ds_step, n_steps, edge_params, fin, fout, Val{tm.use_optimized_schemes}(), L))
  else
    error("Integration order $order not supported")
  end
end

# =========== STRAIGHT ELEMENTS ============= #
# === Thin elements === #
@inline function thin_pure_bdipole(tm::Symplectic, kc, p_over_q_ref, bunch, bm)
  p_over_q_ref = p_over_q_ref
  mm = bm.order
  knl, ksl = get_integrated_strengths(bm, 0, p_over_q_ref)
  if isnothing(bunch.coords.q)
    params = (SA[mm], SA[knl], SA[ksl], 0, 0, 0)
  else
    tilde_m = 1/BeamTracking.R_to_beta_gamma(bunch.species, p_over_q_ref)
    params = (SA[mm], SA[knl], SA[ksl], gyromagnetic_anomaly(bunch.species), 0, tilde_m)
  end
  return push(kc, make_kernel_call(BeamTracking.integrate_thin!, params))
end

@inline function thin_bdipole(tm::Symplectic, kc, p_over_q_ref, bunch, bm)
  p_over_q_ref = p_over_q_ref
  mm = bm.order
  knl, ksl = get_integrated_strengths(bm, 0, p_over_q_ref)
  if isnothing(bunch.coords.q)
    params = (mm, knl, ksl, 0, 0, 0)
  else
    tilde_m = 1/BeamTracking.R_to_beta_gamma(bunch.species, p_over_q_ref)
    params = (mm, knl, ksl, gyromagnetic_anomaly(bunch.species), 0, tilde_m)
  end
  return push(kc, make_kernel_call(BeamTracking.integrate_thin!, params))
end

@inline thin_pure_bquadrupole(tm::Symplectic, kc, p_over_q_ref, bunch, bm) = thin_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm)

@inline thin_bquadrupole(tm::Symplectic, kc, p_over_q_ref, bunch, bm) = thin_bdipole(tm, kc, p_over_q_ref, bunch, bm)

@inline thin_pure_bmultipole(tm::Symplectic, kc, p_over_q_ref, bunch, bm) = thin_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm)

@inline thin_bmultipole(tm::Symplectic, kc, p_over_q_ref, bunch, bm) = thin_bdipole(tm, kc, p_over_q_ref, bunch, bm)


# === Thick elements === #
@inline function drift(tm::Union{Symplectic,DriftKick}, kc, p_over_q_ref, bunch, L)
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  params = (beta_0, gamsqr_0, tilde_m)
  return push(kc, integration_launcher(BeamTracking.exact_drift!, params, nothing, tm, nothing, L))
end

@inline function thick_pure_bsolenoid(tm::Union{Symplectic,SolenoidKick}, kc, p_over_q_ref, bunch, bm, L) 
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = SA[bm.order]
  Ksol, Ksol_skew = get_strengths(bm, L, p_over_q_ref)
  kn = SA[Ksol]
  ks = SA[Ksol_skew]
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, Ksol, nothing, nothing, nothing, nothing, nothing, nothing)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, Ksol, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.sks_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_bsolenoid(tm::Union{Symplectic,SolenoidKick}, kc, p_over_q_ref, bunch, bm, L) 
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  Ksol = kn[1]
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  Kn0 = nothing
  tilt0 = 0
  Kn1 = nothing
  tilt1 = 0
  for j in 1:length(mm)
    if mm[j] == 1
      Kn0 = sqrt(kn[j]^2 + ks[j]^2)
      tilt0 = atan2(ks[j], kn[j])
    elseif mm[j] == 2
      Kn1 = sqrt(kn[j]^2 + ks[j]^2)
      tilt1 = atan2(ks[j], kn[j]) / 2
    end
  end
  if tilt0 ≈ 0
    w0 = nothing
    w0_inv = nothing
  else
    w0 = rot_quaternion(0, 0, tilt0)
    w0_inv = inv_rot_quaternion(0, 0, tilt0)
  end
  if tilt1 ≈ 0
    w1 = nothing
    w1_inv = nothing
  else
    w1 = rot_quaternion(0, 0, tilt1)
    w1_inv = inv_rot_quaternion(0, 0, tilt1)
  end
  edge_params = (a, tilde_m, Ksol, Kn0, w0, w0_inv, Kn1, w1, w1_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, Ksol, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.sks_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_pure_bdipole(tm::DriftKick, kc, p_over_q_ref, bunch, bm, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  Kn0 = nothing
  tilt0 = 0
  Kn1 = nothing
  tilt1 = 0
  for j in 1:length(mm)
    if mm[j] == 1
      Kn0 = sqrt(kn[j]^2 + ks[j]^2)
      tilt0 = atan2(ks[j], kn[j])
    elseif mm[j] == 2
      Kn1 = sqrt(kn[j]^2 + ks[j]^2)
      tilt1 = atan2(ks[j], kn[j]) / 2
    end
  end
  if tilt0 ≈ 0
    w0 = nothing
    w0_inv = nothing
  else
    w0 = rot_quaternion(0, 0, tilt0)
    w0_inv = inv_rot_quaternion(0, 0, tilt0)
  end
  if tilt1 ≈ 0
    w1 = nothing
    w1_inv = nothing
  else
    w1 = rot_quaternion(0, 0, tilt1)
    w1_inv = inv_rot_quaternion(0, 0, tilt1)
  end
  edge_params = (a, tilde_m, nothing, Kn0, w0, w0_inv, Kn1, w1, w1_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, SA[mm], SA[kn], SA[ks])
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, SA[mm], SA[kn], SA[ks])
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.dkd_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_bdipole(tm::Union{Symplectic,DriftKick}, kc, p_over_q_ref, bunch, bm, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  Kn0 = nothing
  tilt0 = 0
  Kn1 = nothing
  tilt1 = 0
  for j in 1:length(mm)
    if mm[j] == 1
      Kn0 = sqrt(kn[j]^2 + ks[j]^2)
      tilt0 = atan2(ks[j], kn[j])
    elseif mm[j] == 2
      Kn1 = sqrt(kn[j]^2 + ks[j]^2)
      tilt1 = atan2(ks[j], kn[j]) / 2
    end
  end
  if tilt0 ≈ 0
    w0 = nothing
    w0_inv = nothing
  else
    w0 = rot_quaternion(0, 0, tilt0)
    w0_inv = inv_rot_quaternion(0, 0, tilt0)
  end
  if tilt1 ≈ 0
    w1 = nothing
    w1_inv = nothing
  else
    w1 = rot_quaternion(0, 0, tilt1)
    w1_inv = inv_rot_quaternion(0, 0, tilt1)
  end
  edge_params = (a, tilde_m, nothing, Kn0, w0, w0_inv, Kn1, w1, w1_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.dkd_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_pure_bdipole(tm::Union{Symplectic,BendKick}, kc, p_over_q_ref, bunch, bm1, L) 
  p_over_q_ref = p_over_q_ref
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm1.order
  kn, ks = get_strengths(bm1, L, p_over_q_ref)
  Kn0 = sqrt(kn^2 + ks^2)
  tilt = atan2(ks, kn)
  if tilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, tilt)
    w_inv = inv_rot_quaternion(0, 0, tilt)
  end
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  E_ref = mc2/tilde_m/beta_0
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, nothing, Kn0, w, w_inv, nothing, nothing, nothing)
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, tilde_m, beta_0, a, 0, w, w_inv, Kn0, SA[mm], SA[kn], SA[ks])
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, SA[mm], SA[kn], SA[ks])
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.bkb_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_bdipole(tm::BendKick, kc, p_over_q_ref, bunch, bm, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  Kn0 = sqrt(kn[1]^2 + ks[1]^2)
  tilt0 = atan2(ks[1], kn[1])
  Kn1 = nothing
  tilt1 = 0
  if mm[2] == 2
    Kn1 = sqrt(kn[2]^2 + ks[2]^2)
    tilt1 = atan2(ks[2], kn[2]) / 2
  end
  if tilt0 ≈ 0
    w0 = nothing
    w0_inv = nothing
  else
    w0 = rot_quaternion(0, 0, tilt0)
    w0_inv = inv_rot_quaternion(0, 0, tilt0)
  end
  if tilt1 ≈ 0
    w1 = nothing
    w1_inv = nothing
  else
    w1 = rot_quaternion(0, 0, tilt1)
    w1_inv = inv_rot_quaternion(0, 0, tilt1)
  end
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, nothing, Kn0, w0, w0_inv, Kn1, w1, w1_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, tilde_m, beta_0, a, 0, w0, w0_inv, Kn0, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.bkb_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_bdipole(tm::MatrixKick, kc, p_over_q_ref, bunch, bm, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  Kn0 = sqrt(kn[1]^2 + ks[1]^2)
  tilt0 = atan2(ks[1], kn[1])
  Kn1 = nothing
  tilt1 = 0
  if mm[2] == 2
    Kn1 = sqrt(kn[2]^2 + ks[2]^2)
    tilt1 = atan2(ks[2], kn[2]) / 2
  end
  if tilt0 ≈ 0
    w0 = nothing
    w0_inv = nothing
  else
    w0 = rot_quaternion(0, 0, tilt0)
    w0_inv = inv_rot_quaternion(0, 0, tilt0)
  end
  if tilt1 ≈ 0
    w1 = nothing
    w1_inv = nothing
  else
    w1 = rot_quaternion(0, 0, tilt1)
    w1_inv = inv_rot_quaternion(0, 0, tilt1)
  end
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, nothing, Kn0, w0, w0_inv, Kn1, w1, w1_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, w1, w1_inv, Kn1, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.mkm_quadrupole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_pure_bquadrupole(tm::Union{Symplectic,MatrixKick}, kc, p_over_q_ref, bunch, bm, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  k1 = sqrt(kn^2 + ks^2)
  tilt = atan2(ks, kn) / 2
  if tilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, tilt)
    w_inv = inv_rot_quaternion(0, 0, tilt)
  end
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, nothing, nothing, nothing, nothing, k1, w, w_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, w, w_inv, k1, SA[mm], SA[kn], SA[ks])
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, SA[mm], SA[kn], SA[ks])
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.mkm_quadrupole!, params, photon_params, tm, edge_params, L))
end

@inline thick_pure_bquadrupole(tm::DriftKick, kc, p_over_q_ref, bunch, bm, L) = 
  thick_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm, L)

@inline function thick_bquadrupole(tm::Union{Symplectic,MatrixKick}, kc, p_over_q_ref, bunch, bm, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  k1 = sqrt(kn[1]^2 + ks[1]^2)
  tilt = atan2(ks[1], kn[1]) / 2
  if tilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, tilt)
    w_inv = inv_rot_quaternion(0, 0, tilt)
  end
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, nothing, nothing, nothing, nothing, k1, w, w_inv)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, w, w_inv, k1, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.mkm_quadrupole!, params, photon_params, tm, edge_params, L))
end

@inline thick_bquadrupole(tm::DriftKick, kc, p_over_q_ref, bunch, bm, L) = thick_bdipole(tm, kc, p_over_q_ref, bunch, bm, L)

@inline thick_pure_bmultipole(tm::Union{Symplectic,DriftKick}, kc, p_over_q_ref, bunch, bm, L) = 
  thick_pure_bdipole(remake(DriftKick, tm), kc, p_over_q_ref, bunch, bm, L)

@inline thick_bmultipole(tm::Union{Symplectic,DriftKick}, kc, p_over_q_ref, bunch, bm, L) = 
  thick_bdipole(remake(DriftKick, tm), kc, p_over_q_ref, bunch, bm, L)


# =========== BENDING ELEMENTS ============= #
@inline function thick_bend_no_field(tm::Union{Symplectic,BendKick}, kc, p_over_q_ref, bunch, bendparams, L)
  g = bendparams.g_ref
  ntilt = -bendparams.tilt_ref
  e1 = bendparams.e1
  e2 = bendparams.e2
  if ntilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, ntilt)
    w_inv = inv_rot_quaternion(0, 0, ntilt)
  end
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  params = (nothing, tilde_m, beta_0, gyromagnetic_anomaly(bunch.species), g, w, w_inv, 0, SA[0], SA[0], SA[0])
  return push(kc, integration_launcher(BeamTracking.bkb_multipole!, params, nothing, tm, nothing, L))
end

@inline function thick_bend_pure_bdipole(tm::Union{Symplectic,BendKick}, kc, p_over_q_ref, bunch, bendparams, bm1, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  g = bendparams.g_ref
  ntilt = -bendparams.tilt_ref
  e1 = bendparams.e1
  e2 = bendparams.e2
  mm = bm1.order
  Kn0, Ks0 = get_strengths(bm1, L, p_over_q_ref)
  Ks0 ≈ 0 || error("A skew dipole field cannot yet be used in a bend")
  if ntilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, ntilt)
    w_inv = inv_rot_quaternion(0, 0, ntilt)
  end
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, Kn0, w, w_inv, e1, e2)
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, tilde_m, beta_0, a, g, w, w_inv, Kn0, SA[mm], SA[Kn0], SA[Ks0])
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, g, ntilt, SA[mm], SA[Kn0], SA[Ks0])
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.bkb_multipole!, params, photon_params, tm, edge_params, L))
end

@inline function thick_bend_bdipole(tm::Union{Symplectic,BendKick}, kc, p_over_q_ref, bunch, bendparams, bm, L)
  @warn "Straight multipoles are being used in a curved reference system. Maxwell's equations in free space are not satisfied." maxlog=1
  p_over_q_ref = p_over_q_ref
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  g = bendparams.g_ref
  ntilt = -bendparams.tilt_ref
  e1 = bendparams.e1
  e2 = bendparams.e2
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  Kn0 = kn[1]
  ks[1] ≈ 0 || error("A skew dipole field cannot yet be used in a bend")
  if ntilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, ntilt)
    w_inv = inv_rot_quaternion(0, 0, ntilt)
  end
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, Kn0, w, w_inv, e1, e2)
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  E_ref = mc2/tilde_m/beta_0
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, tilde_m, beta_0, a, g, w, w_inv, Kn0, mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, g, ntilt, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.bkb_multipole!, params, photon_params, tm, edge_params, L))
end


# =========== RF ============= #
@inline function thick_pure_rf(tm::Union{Symplectic,DriftKick}, kc, p_over_q_ref, bunch, rfparams, beamlineparams, L)
  p_over_q_ref = p_over_q_ref
  omega = rf_omega_calc(rfparams, beamlineparams)
  t_ref = (rf_phi0_calc(rfparams, beamlineparams.beamline.species_ref) - pi/2)/omega
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  E0_normalized = rfparams.voltage/L/p_over_q_ref
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  E_ref = mc2/tilde_m/beta_0
  a = gyromagnetic_anomaly(bunch.species)
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, omega, t_ref, E0_normalized, 0, Val{false}(), SA[], SA[], SA[])
  return push(kc, integration_launcher(BeamTracking.cavity!, params, nothing, tm, nothing, L))
end

@inline function thick_bmultipole_rf(tm::Union{Symplectic,DriftKick,SolenoidKick}, kc, p_over_q_ref, bunch, bm, rfparams, beamlineparams, L)
  p_over_q_ref = p_over_q_ref
  omega = rf_omega_calc(rfparams, beamlineparams)
  t_ref = (rf_phi0_calc(rfparams, beamlineparams.beamline.species_ref) - pi/2) / omega
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  E0_normalized = rfparams.voltage/L/p_over_q_ref
  mm = bm.order
  kn, ks = get_strengths(bm, L, p_over_q_ref)
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  E_ref = mc2/tilde_m/beta_0
  Ksol = nothing
  Kn0 = nothing
  tilt0 = 0
  Kn1 = nothing
  tilt1 = 0
  for j in 1:length(mm)
    if mm[j] == 0
      Ksol = kn[j]
    elseif mm[j] == 1
      Kn0 = sqrt(kn[j]^2 + ks[j]^2)
      tilt0 = atan2(ks[j], kn[j])
    elseif mm[j] == 2
      Kn1 = sqrt(kn[j]^2 + ks[j]^2)
      tilt1 = atan2(ks[j], kn[j]) / 2
    end
  end
  if tilt0 ≈ 0
    w0 = nothing
    w0_inv = nothing
  else
    w0 = rot_quaternion(0, 0, tilt0)
    w0_inv = inv_rot_quaternion(0, 0, tilt0)
  end
  if tilt1 ≈ 0
    w1 = nothing
    w1_inv = nothing
  else
    w1 = rot_quaternion(0, 0, tilt1)
    w1_inv = inv_rot_quaternion(0, 0, tilt1)
  end
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, Ksol, Kn0, w0, w0_inv, Kn1, w1, w1_inv)
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, gamsqr_0, tilde_m, a, omega, t_ref, E0_normalized, Ksol, Val{!isnothing(Ksol)}(), mm, kn, ks)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (get_backend(bunch.coords.v), q, mc2, E_ref, 0, 0, mm, kn, ks)
  else
    photon_params = nothing
  end
  return push(kc, integration_launcher(BeamTracking.cavity!, params, photon_params, tm, edge_params, L))
end


# =========== IMPLICIT ============= #
@inline function implicit(tm::Symplectic, kc, p_over_q_ref, bunch, fpp, bp, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  if !isnothing(bp)
    g = bp.g_ref
    ntilt = -bp.tilt_ref
    (bp.e1 ≈ 0 && bp.e2 ≈ 0) || error("Edge angles are not used in implicit integration")
  else
    g = 0
    ntilt = 0
  end
  potential_and_jac = fpp.four_potential
  potential_params = fpp.four_potential_params
  normalized = Val{fpp.four_potential_normalized}()
  if ntilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, ntilt)
    w_inv = inv_rot_quaternion(0, 0, ntilt)
  end
  a = gyromagnetic_anomaly(bunch.species)
  E_ref = BeamTracking.R_to_E(bunch.species, p_over_q_ref)
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, beta_0, tilde_m, a, g, w, w_inv, potential_and_jac, potential_params, p_over_q_ref, normalized, Val{tm.implicit_use_newton}())
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (BeamTracking.implicit_integrator!, get_backend(bunch.coords.v), q, mc2, E_ref, g, potential_and_jac, potential_params, p_over_q_ref, normalized)
  else
    photon_params = nothing
  end
  kc = push(kc, make_kernel_call(BeamTracking.bmad_to_mad!, (beta_0, tilde_m, 0)))
  kc = push(kc, integration_launcher(BeamTracking.implicit_integrator!, params, photon_params, tm, nothing, L))
  kc = push(kc, make_kernel_call(BeamTracking.mad_to_bmad!, (beta_0, tilde_m, 0)))
  kc = push_transforms_out(kc, make_kernel_call(
      BeamTracking.callback_implicit!, (beta_0, tilde_m, potential_and_jac, potential_params, p_over_q_ref, normalized, Val{false}())
    )
  )
  kc = push_transforms_in(kc, make_kernel_call(
      BeamTracking.callback_implicit!, (beta_0, tilde_m, potential_and_jac, potential_params, p_over_q_ref, normalized, Val{true}())
    )
  )
  return kc
end


# =========== ELECTRIC ============= #
@inline function thick_pure_edipole(tm::Symplectic, kc, p_over_q_ref, bunch, em1, L)
  p_over_q_ref = p_over_q_ref
  tilde_m, _, beta_0 = BeamTracking.drift_params(bunch.species, p_over_q_ref)
  mm = em1.order
  kn, ks = get_e_strengths(em1, L)
  kE = sqrt(kn^2 + ks^2)/(p_over_q_ref * C_LIGHT)
  tilt = atan2(ks, kn)
  if tilt ≈ 0
    w = nothing
    w_inv = nothing
  else
    w = rot_quaternion(0, 0, tilt)
    w_inv = inv_rot_quaternion(0, 0, tilt)
  end
  q = chargeof(bunch.species)
  mc2 = massof(bunch.species)
  E_ref = mc2/tilde_m/beta_0
  a = gyromagnetic_anomaly(bunch.species)
  edge_params = (a, tilde_m, kE, w, w_inv)
  radiation_params = ifelse(tm.radiation_damping_on, (q, mc2, E_ref), nothing)
  params = (radiation_params, kE, beta_0, tilde_m, w, w_inv, a)
  if isprimitivetype(eltype(bunch.coords.v)) && tm.radiation_fluctuations_on
    photon_params = (BeamTracking.exact_elsep!, get_backend(bunch.coords.v), q, mc2, E_ref, kE)
  else
    photon_params = nothing
  end
  kc = push(kc, make_kernel_call(BeamTracking.bmad_to_mad!, (beta_0, tilde_m, 0)))
  kc = push(kc, integration_launcher(BeamTracking.elsep_rad!, params, photon_params, tm, edge_params, L))
  kc = push(kc, make_kernel_call(BeamTracking.mad_to_bmad!, (beta_0, tilde_m, 0)))
  kc = push_transforms_out(kc, make_kernel_call(BeamTracking.callback_electric!, (beta_0, tilde_m, kE, Val{false}())))
  kc = push_transforms_in(kc, make_kernel_call(BeamTracking.callback_electric!, (beta_0, tilde_m, kE, Val{true}())))
  return kc
end