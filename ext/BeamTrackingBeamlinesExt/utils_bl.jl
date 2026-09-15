Base.promote_rule(::Type{DefExpr{T}}, ::Type{TimeDependentParam}) where {T} = DefExpr{TimeDependentParam}

Beamlines.DefExpr{T}(a::TimeDependentParam) where {T} = DefExpr{T}(()->convert(T,a))

"""
    check_bl_bunch!(bunch::Bunch, bl::Beamline, notify::Bool=true)

Ensures the Bunch and Beamline have the same species, by setting the 
Bunch with the Beamline species if not set. 
"""
function check_bl_bunch!(bunch::Bunch, bl::Beamline, notify::Bool=true)
  species_ref = Beamlines.trygetproperty(bl, :species_ref)
  p_over_q_ref = Beamlines.trygetproperty(bl, :p_over_q_ref)
  return _check_bunch!(bunch, species_ref, p_over_q_ref, notify)
end

# Ensures the 
function _check_bunch!(bunch, species_ref, p_over_q_ref, notify::Bool=true)
  if isnullspecies(bunch.species)
    if species_ref isa Beamlines.GetError
      error("Neither the Bunch nor the Beamline has set a species.")
    else
      if notify
        println("Setting bunch.species = $species_ref (reference species from the Beamline)")
      end
      setfield!(bunch, :species, species_ref)
    end
  elseif !(species_ref isa Beamlines.GetError) && species_ref != bunch.species && notify
    error("The species of the bunch currently must equal the reference species of the Beamline.")
  end

  t_ref = bunch.t_ref
  if isnan(bunch.p_over_q_ref)
    if p_over_q_ref isa Beamlines.GetError
      if notify
        println("
          WARNING: Both the bunch and beamline do not have any set reference energy. 
          If any LineElements have unnormalized fields stored as independent variables, 
          or if there is spin tracking, there will be NaNs.
        ")
      end
      p_over_q_ref = bunch.p_over_q_ref
    else
      if notify
        if p_over_q_ref isa TimeDependentParam
          println("Setting bunch.p_over_q_ref = $(p_over_q_ref(t_ref)) (reference p_over_q_ref from the Beamline at t_ref = $t_ref)")
        else
          println("Setting bunch.p_over_q_ref = $p_over_q_ref (reference p_over_q_ref from the Beamline)")
        end
      end
      if p_over_q_ref isa TimeDependentParam
        setfield!(bunch, :p_over_q_ref, typeof(bunch.p_over_q_ref)(p_over_q_ref(t_ref)))
      else
        setfield!(bunch, :p_over_q_ref, typeof(bunch.p_over_q_ref)(p_over_q_ref))
      end
    end
  end
  return bunch.species, p_over_q_ref
end

#---------------------------------------------------------------------------------------------------

get_n_multipoles(::BMultipoleParams{T,N}) where {T,N} = N
get_n_multipoles(::EMultipoleParams{T,N}) where {T,N} = N

make_static(a::StaticArray) = SVector(a)
make_static(a::SizedArray) = SVector(a)
make_static(a) = a

#---------------------------------------------------------------------------------------------------

"""
    get_strengths(bm, L, p_over_q_ref) -> Kn, Ks
Get non-integrated magnetic multipole strength arrays. Also see get_integrated_strengths.

(Kn' + im*Ks') = (Kn + im*Ks)*exp(-im*order*tilt)

# Rotation:
Kn' = Kn*cos(order*tilt) + Ks*sin(order*tilt)
Ks' = Kn*-sin(order*tilt) + Ks*cos(order*tilt)

This works for both BMultipole and BMultipoleParams. Branchless bc SIMD -> basically 
no loss in computing both but benefit of branchless.
"""
@inline function get_strengths(bm, L, p_over_q_ref)
  bmn = getfield(bm, :n)
  bms = getfield(bm, :s)
  bmtilt = getfield(bm, :tilt)
  if isconcretetype(eltype(bmn))
    T = promote_type(eltype(bmn),
                    typeof(L), typeof(p_over_q_ref)
    )
  else
    if bmn isa AbstractArray
      T = promote_type(reduce(promote_type, typeof.(bmn)), 
                      reduce(promote_type, typeof.(bms)),
                      reduce(promote_type, typeof.(bmtilt)),
                      typeof(L), typeof(p_over_q_ref)
      )
    else
      T = promote_type(typeof(bmn), 
                      typeof(bms),
                      typeof(bmtilt),
                      typeof(L), typeof(p_over_q_ref)
      )
    end
  end
  n = T.(make_static(bmn))
  s = T.(make_static(bms))
  tilt = T.(make_static(bmtilt))
  order = getfield(bm, :order)
  normalized = getfield(bm, :normalized)
  integrated = getfield(bm, :integrated)
  np = @. n*cos(order*tilt) + s*sin(order*tilt)
  sp = @. -n*sin(order*tilt) + s*cos(order*tilt)
  np = @. ifelse(!normalized, np/p_over_q_ref, np) 
  sp = @. ifelse(!normalized, sp/p_over_q_ref, sp) 
  np = @. ifelse(integrated, np/L, np)
  sp = @. ifelse(integrated, sp/L, sp)
  return np, sp
end

"""
    get_physical_strengths(bm, L, p_over_q_ref) -> Bn, Bs

Get non-integrated magnetic multipole coefficients in physical units. Normal
and skew coefficients of order `m` are returned in T/m^m. Physical source
values are used directly. Normalized source values are converted once while
preparing the tracking kernel.
"""
@inline function get_physical_strengths(bm, L, p_over_q_ref)
  bmn = getfield(bm, :n)
  bms = getfield(bm, :s)
  bmtilt = getfield(bm, :tilt)
  normalized = getfield(bm, :normalized)

  if isconcretetype(eltype(bmn))
    T = promote_type(eltype(bmn), typeof(L))
  elseif bmn isa AbstractArray
    T = promote_type(
      reduce(promote_type, typeof.(bmn)),
      reduce(promote_type, typeof.(bms)),
      reduce(promote_type, typeof.(bmtilt)),
      typeof(L),
    )
  else
    T = promote_type(typeof(bmn), typeof(bms), typeof(bmtilt), typeof(L))
  end
  if any(normalized)
    T = promote_type(T, typeof(p_over_q_ref))
  end

  n = T.(make_static(bmn))
  skew = T.(make_static(bms))
  tilt = T.(make_static(bmtilt))
  order = getfield(bm, :order)
  integrated = getfield(bm, :integrated)

  np = @. n*cos(order*tilt) + skew*sin(order*tilt)
  sp = @. -n*sin(order*tilt) + skew*cos(order*tilt)
  np = @. ifelse(normalized, np*p_over_q_ref, np)
  sp = @. ifelse(normalized, sp*p_over_q_ref, sp)
  np = @. ifelse(integrated, np/L, np)
  sp = @. ifelse(integrated, sp/L, sp)
  return np, sp
end

@inline function get_integrated_strengths(bm, L, p_over_q_ref)
  bmn = getfield(bm, :n)
  bms = getfield(bm, :s)
  bmtilt = getfield(bm, :tilt)
  if isconcretetype(eltype(bmn))
    T = promote_type(eltype(bmn),
                    typeof(L), typeof(p_over_q_ref)
    )
  else
    if bmn isa AbstractArray
      T = promote_type(reduce(promote_type, typeof.(bmn)), 
                      reduce(promote_type, typeof.(bms)),
                      reduce(promote_type, typeof.(bmtilt)),
                      typeof(L), typeof(p_over_q_ref)
      )
    else
      T = promote_type(typeof(bmn), 
                      typeof(bms),
                      typeof(bmtilt),
                      typeof(L), typeof(p_over_q_ref)
      )
    end
  end
  n = T.(make_static(bmn))
  s = T.(make_static(bms))
  tilt = T.(make_static(bmtilt))
  order = getfield(bm, :order)
  normalized = getfield(bm, :normalized)
  integrated = getfield(bm, :integrated)
  np = @. n*cos(order*tilt) + s*sin(order*tilt)
  sp = @. -n*sin(order*tilt) + s*cos(order*tilt)
  np = @. ifelse(!normalized, np/p_over_q_ref, np) 
  sp = @. ifelse(!normalized, sp/p_over_q_ref, sp) 
  np = @. ifelse(!integrated, np*L, np)
  sp = @. ifelse(!integrated, sp*L, sp)
  return np, sp
end

@inline function get_e_strengths(em, L)
  emn = getfield(em, :n)
  ems = getfield(em, :s)
  emtilt = getfield(em, :etilt)
  if isconcretetype(eltype(emn))
    T = promote_type(eltype(emn),
                    typeof(L)
    )
  else
    if emn isa AbstractArray
      T = promote_type(reduce(promote_type, typeof.(emn)), 
                      reduce(promote_type, typeof.(ems)),
                      reduce(promote_type, typeof.(emtilt)),
                      typeof(L)
      )
    else
      T = promote_type(typeof(emn), 
                      typeof(ems),
                      typeof(emtilt),
                      typeof(L)
      )
    end
  end
  n = T.(make_static(emn))
  s = T.(make_static(ems))
  tilt = T.(make_static(emtilt))
  order = getfield(em, :order)
  integrated = getfield(em, :integrated)
  np = @. n*cos(order*tilt) + s*sin(order*tilt)
  sp = @. -n*sin(order*tilt) + s*cos(order*tilt)
  np = @. ifelse(integrated, np/L, np)
  sp = @. ifelse(integrated, sp/L, sp)
  return np, sp
end

#---------------------------------------------------------------------------------------------------

function rf_omega_calc(rfparams, beamlineparams)
  if rfparams.rate_meaning == RateMeaning.Harmon
    beamline = beamlineparams.beamline
    p_over_q_ref = beamline.p_over_q_ref
    species = beamline.species_ref
    circumference = beamline.line[end].s_downstream
    v = R_to_v(species, p_over_q_ref)
    return 2*pi*rfparams.rate*v/circumference
  else
    return 2*pi*rfparams.rate
  end
end

#---------------------------------------------------------------------------------------------------

function rf_phi0_calc(rfparams, species)
  dphi = chargeof(species) > 0 ? 0 : pi

  if rfparams.zero_phase == PhaseRef.BelowTransition
    return rfparams.phi0 + pi/2 + dphi
  elseif rfparams.zero_phase == PhaseRef.AboveTransition
    return rfparams.phi0 - pi/2 + dphi
  elseif rfparams.zero_phase == PhaseRef.Accelerating
    return rfparams.phi0 + dphi
  else
    error("RF parameter zero_phase value not set correctly.")
  end
end

#---------------------------------------------------------------------------------------------------

function fringe_in(f::Fringe.T)
  if f == Fringe.BothEnds || f == Fringe.EntranceEnd
    return Val{true}()
  else
    return Val{false}()
  end
end

function fringe_out(f::Fringe.T)
  if f == Fringe.BothEnds || f == Fringe.ExitEnd
    return Val{true}()
  else
    return Val{false}()
  end
end

#---------------------------------------------------------------------------------------------------
"""
    rf_step_calc(n_cells, L_active, rf_omega, L) -> n_cells_out, L_active_out

For an element of length `L`, calculate the number of RF cells (kicks) `n_cells_out` and the active length 
`L_active_out` given the input number of cells `n_cells` and the active length length `L_active`.

If `L_active` is negative, `L_active_out` is set to the element length `L`.
If `n_cells` is negative, `n_cells_out` is set so that the cell length is near half a wavelength.
If `L_active` is zero, `n_cells_out` is set to zero.
"""
function rf_step_calc(n_cells, L_active, rf_omega, L)  
  L_active < 0 ? L_act = L : L_act = L_active

  if n_cells < 0
    return round(rf_omega * L_act / (pi * C_LIGHT)), L_act
  elseif L_active == 0
    return 0, L_active
  else
    return n_cells, L_act
  end
end
