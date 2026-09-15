# ========== Fringe  ===========================
@enumx Fringe NoEnd BothEnds EntranceEnd ExitEnd

# ========== Symplectic ===========================
abstract type AbstractSymplectic end

macro def_integrator_struct(name)
  quote
    struct $(esc(name)) <: AbstractSymplectic
      order::Int
      n_steps::Int 
      ds_step::Float64
      radiation_damping_on::Bool
      radiation_fluctuations_on::Bool
      fringe_at::Fringe.T
      ibs_damping_on::Bool
      ibs_fluctuations_on::Bool
      implicit_use_newton::Bool
      use_optimized_schemes::Bool
  
      function $(esc(name))(; order::Int=4, n_steps::Int=-1, ds_step::Float64=-1.0, radiation_damping_on::Bool=false, radiation_fluctuations_on::Bool=false, fringe_at::Fringe.T=Fringe.BothEnds, ibs_damping_on::Bool=false, ibs_fluctuations_on::Bool=false, implicit_use_newton::Bool=false, use_optimized_schemes::Bool=true)
        _order = order
        _n_steps = n_steps
        _ds_step = ds_step
        if _order ∉ (2, 4, 6, 8, 10)
          error("Symplectic integration only supports orders 2, 4, 6, 8, and 10")
        elseif _n_steps == -1 && _ds_step == -1.0
          _n_steps = 1
        elseif _n_steps > 0 && _ds_step > 0
          error("Only one of n_steps or ds_step should be specified")
        elseif _n_steps < 1 && _ds_step <= 0
          error("Invalid step size")
        elseif _n_steps > 0
          _ds_step = -1.0
        elseif _ds_step > 0
          _n_steps = -1
        end
        return new(_order, _n_steps, _ds_step, radiation_damping_on, radiation_fluctuations_on, fringe_at, ibs_damping_on, ibs_fluctuations_on, implicit_use_newton, use_optimized_schemes)
      end
    end
  end
end

@def_integrator_struct(Symplectic) # Automatically selects split
@def_integrator_struct(MatrixKick)
@def_integrator_struct(BendKick)
@def_integrator_struct(SolenoidKick)
@def_integrator_struct(DriftKick)

function remake(::Type{T}, ytm::AbstractSymplectic) where {T<:AbstractSymplectic}
  return T(
    order = ytm.order,
    n_steps = ytm.n_steps,
    ds_step = ytm.ds_step,
    radiation_damping_on = ytm.radiation_damping_on,
    radiation_fluctuations_on = ytm.radiation_fluctuations_on,
    fringe_at = ytm.fringe_at,
    ibs_damping_on = ytm.ibs_damping_on,
    ibs_fluctuations_on = ytm.ibs_fluctuations_on,
    implicit_use_newton = ytm.implicit_use_newton,
    use_optimized_schemes = ytm.use_optimized_schemes
  )
end



# ========== Exact ===========================

struct Exact
  fringe_at::Fringe.T
  
  function Exact(; fringe_at::Fringe.T=Fringe.BothEnds)
    return new(fringe_at)
  end
end

#---------------------------------------------------------------------------------------------------

struct SaganCavity
  n_cells::Int                # Negative => Use approx half wavelength between cells, Zero => single kick.
  L_active::Float64          # Negative => Use L as the active length.
  radiation_damping_on::Bool
  radiation_fluctuations_on::Bool

  function SaganCavity(; n_cells::Int = -1, L_active::Float64 = -1.0, 
                             radiation_damping_on::Bool=false, radiation_fluctuations_on::Bool=false)
    return new(n_cells, L_active, radiation_damping_on, radiation_fluctuations_on)
  end
end

# ========== Explicit RK4 Tracking ==========
struct RungeKutta{F,A}
  ds_step::Float64
  n_steps::Int
  field::F
  additional_field::A
end

DEFAULT_RK4_DS_STEP = 0.2
function RungeKutta(;
  ds_step::Union{Float64,Nothing}=nothing,
  n_steps::Union{Int,Nothing}=nothing,
  field=nothing,
  additional_field=nothing,
)
  if !isnothing(field) && !isnothing(additional_field)
    error("RungeKutta accepts either field or additional_field")
  end

  # Get actual values (use provided or sentinel)
  _ds_step = ds_step === nothing ? -1.0 : ds_step
  _n_steps = n_steps === nothing ? -1 : n_steps
  
  # Error if both are explicitly set to positive values
  if _ds_step > 0 && _n_steps > 0
    error("Only one of ds_step or n_steps should be specified")
  end
  
  # If user sets n_steps (and it's positive), set ds_step to negative
  if _n_steps > 0
    return RungeKutta(-1.0, _n_steps, field, additional_field)
  end
  
  # If user sets ds_step (and it's positive), set n_steps to -1
  if _ds_step > 0
    return RungeKutta(_ds_step, -1, field, additional_field)
  end
  
  # Fallback: use defaults if both are negative/not set
  return RungeKutta(DEFAULT_RK4_DS_STEP, -1, field, additional_field)
end
