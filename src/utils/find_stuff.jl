function find_steps(tm::BeamTracking.AbstractSymplectic, L) 
  if L == 0
    return (1, L)
  end
  ds_step = tm.ds_step
  n_steps = tm.n_steps
  if ds_step < 0
    ds_step = L / n_steps
    return (n_steps, ds_step)
  else
    return (ceil(Int, L / ds_step), ds_step)
  end
end
find_steps(::Any, L) = (1, L)

function find_steps(tm::RungeKutta, L)
  L > 0 || error("RungeKutta tracking requires a positive element length")
  n_steps = tm.n_steps > 0 ? tm.n_steps : ceil(Int, L / tm.ds_step)
  return n_steps, L / n_steps
end

# Temporary disgusting solution for callbacks - Symplectic
@generated function compute_dt_ref(s, ker::K, params) where {K}
  idx = find_m_tilde(ker)
  if idx == 0
    return quote
      return 0
    end
  else
    return quote
      beta_gamma_ref = 1 / params[$idx]
      return s / beta_gamma_to_v(beta_gamma_ref)
    end
  end
end

find_m_tilde(::Type{K}) where {K<:typeof(implicit_integrator!)} = 3
find_m_tilde(::Type{K}) where {K<:typeof(exact_drift!)} = 3
find_m_tilde(::Type{K}) where {K<:typeof(sks_multipole!)} = 4
find_m_tilde(::Type{K}) where {K<:typeof(dkd_multipole!)} = 4
find_m_tilde(::Type{K}) where {K<:typeof(bkb_multipole!)} = 2
find_m_tilde(::Type{K}) where {K<:typeof(mkm_quadrupole!)} = 4
find_m_tilde(::Type{K}) where {K} = 0 # Fallback