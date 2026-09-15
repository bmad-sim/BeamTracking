using Test,
      AtomicAndPhysicalConstants,
      BeamTracking,
      Beamlines,
      JET,
      BenchmarkTools,
      GTPSA,
      StaticArrays,
      ReferenceFrameRotations,
      SIMD,
      KernelAbstractions,
      ForwardDiff

using BeamTracking: Coords, KernelCall, make_kernel_call, Q0, QX, QY, QZ, STATE_ALIVE, STATE_LOST, C_LIGHT,
      STATE_LOST_NEG_X, STATE_LOST_POS_X, STATE_LOST_NEG_Y, STATE_LOST_POS_Y, STATE_LOST_PZ, STATE_LOST_Z,
      rot_quaternion, inv_rot_quaternion, atan2, sincu, sinhcu, sincuc, expq, atan2,
      quat_mul, quat_rotate, gaussian_random
using Beamlines: isactive

@show BeamTracking.REGISTER_SIZE
@show Sys.ARCH
@show Threads.nthreads()

BenchmarkTools.DEFAULT_PARAMETERS.gctrial = false
BenchmarkTools.DEFAULT_PARAMETERS.evals = 2

const D1 = Descriptor(6, 1)   # 6 variables, 1st order
const D1_1 = Descriptor(6, 1, 1, 1) # 6 variables and 1 parameter, 1st order
const D1_2 = Descriptor(6, 1, 2, 1) # 6 variables and 2 parameters, 1st order
const D10 = Descriptor(6, 10) # 6 variables, 10th order

function test_matrix(
  M_expected,    # Expected matrix
  kernel_call;
  type_stable=VERSION >= v"1.11", 
  no_scalar_allocs=!(any(t->eltype(t) <: TPS, kernel_call.args)), # only for non-parametric 
  rtol=nothing, 
  atol=nothing,
  printit = false
)
  # Initialize bunch without spin
  v = transpose(@vars(D1))
  state = similar(v, UInt8, 1)
  state .= STATE_ALIVE
  coords = Coords(state, v, nothing, nothing, ())

  # Set up kernel chain and launch!
  BeamTracking.launch!(coords, kernel_call)

  # Set up tolerance kwargs
  kwargs = ()
  if !isnothing(atol)
    kwargs = pairs((;kwargs..., atol=atol))
  end
  if !isnothing(rtol)
    kwargs = pairs((;kwargs..., rtol=rtol))
  end

  # 1) Correctness
  if printit; println(GTPSA.jacobian(coords.v)[1:6,1:6]); end
  @test isapprox(GTPSA.jacobian(coords.v)[1:6,1:6], scalar.(M_expected); kwargs...)

  # 2) Type stability
  if type_stable
    @test_opt kernel_call.kernel(1, coords, kernel_call.args...)
  end

  # 3) No scalar allocations
  if no_scalar_allocs
    v = repeat([0.01 0.02 0.03 0.04 0.05 0.06], 2)
    q = repeat([1.0 0.0 0.0 0.0], 2)
    state = [STATE_ALIVE STATE_ALIVE]
    @test @ballocated(BeamTracking.launch!(coords, $kernel_call; use_KA=false), 
    setup=(coords = Coords(copy($state), copy($v), copy($q), nothing, ()))) == 0
  end
end

function read_map(bmad_map_file::AbstractString)
  # Load reference data from file in isolated module to avoid polluting global namespace
  mod = Module()
  Base.include(mod, bmad_map_file)
  getd_z() = getfield(mod, :d_z)
  getv_z() = getfield(mod, :v_z)
  d_z = invokelatest(getd_z)
  v_z = invokelatest(getv_z)
  d_z == D10 || error("Please use a 10th order map for test_map")
  return v_z
end

function read_spin_orbit_map(bmad_map_file::AbstractString)
  # Load reference data from file in isolated module to avoid polluting global namespace
  mod = Module()
  Base.include(mod, bmad_map_file)
  getd_z() = getfield(mod, :d_z)
  getv_z() = getfield(mod, :v_z)
  getq_z() = getfield(mod, :q_z)
  d_z = invokelatest(getd_z)
  v_z = invokelatest(getv_z)
  q_z = invokelatest(getq_z)
  d_z == D10 || error("Please use a 10th order map for test_map")
  return (v_z, q_z)
end

function test_map(
  bmad_map_file::AbstractString,
  kernel_call;
  type_stable=VERSION >= v"1.11", 
  no_scalar_allocs=!(any(t->eltype(t) <: TPS, kernel_call.args)), # only for non-parametric 
  tol=1e-8
)
  v_expected = read_map(bmad_map_file)

  # Initialize bunch without spin
  v = transpose(@vars(D10))
  q = TPS64{D10}[1 0 0 0]
  state = similar(v, UInt8, 1)
  state .= STATE_ALIVE
  coords = Coords(state, v, q, nothing, ())

  # Set up kernel chain and launch!
  BeamTracking.launch!(coords, kernel_call)

  # 1) Correctness
  @test coeffs_approx_equal(v_expected, coords.v, tol)
  # 2) Type stability
  if type_stable
    @test_opt kernel_call.kernel(1, coords, kernel_call.args...)
  end
  # 3) No scalar allocations
  if no_scalar_allocs
    v = repeat([0.01 0.02 0.03 0.04 0.05 0.06], 2)
    q = repeat([1.0 0.0 0.0 0.0], 2)
    state = [STATE_ALIVE STATE_ALIVE]
    @test @ballocated(BeamTracking.launch!(coords, $kernel_call; use_KA=false), 
    setup=(coords = Coords(copy($state), copy($v), copy($q), nothing, ()))) == 0
  end


  #= LineElement tracking test
    if haskey(kwargs, :p_over_q_ref) && haskey(kwargs, :species)
      p_over_q_ref = kwargs[:p_over_q_ref]
    elseif haskey(kwargs, :E) && haskey(kwargs, :species)
      p_over_q_ref = BeamTracking.E_to_R(kwargs[:species], kwargs[:E])
    elseif haskey(kwargs, :p0c) && haskey(kwargs, :species)
      p_over_q_ref = BeamTracking.E_to_R(kwargs[:species], sqrt(kwargs[:p0c]^2 + massof(kwargs[:species])^2))
    else
      error("`p_over_q_ref`, `E` or `p0c`, as well as `species` must both be provided as keyword arguments")
    end
    
    if !haskey(kwargs, :ele)
      error("ele must be provided as a keyword argument")
    else
      coords = Bunch(v, species=kwargs[:species], p_over_q_ref=p_over_q_ref)
      v = track!(coords, kwargs[:ele]).v
      @test coeffs_approx_equal(v_expected, v, tol)
    end

  =#
end

# Coefficient-wise approximate equality
function coeffs_approx_equal(v_expected, v_calculated, ϵ)
  n = GTPSA.numcoefs(v_expected[1])
  all_ok = true
  for i in 1:length(v_expected)
      for j in 0:n-1
          c1, c2 = v_expected[i][j], v_calculated[i][j]
          if abs(c1 - c2) > max(ϵ, ϵ * (abs(c1) + abs(c2)))
              println("Coefficients not equal: v_expected[$i][$j] = $c1, v_calculated[$i][$j] = $c2")
              println("Difference: $(abs(c1 - c2))")
              println("Tolerance:  $(max(ϵ, ϵ * (abs(c1) + abs(c2))))")
              all_ok = false
              break
          end
      end
      if !all_ok
          break
      end
  end
  return all_ok
end


function quaternion_coeffs_approx_equal(q_expected, q_calculated, ϵ)
  sgn = ifelse(q_expected.q0[[0,0,0,0,0,0]] * q_calculated.q0[[0,0,0,0,0,0]] >= 0, 1, -1)
  components = (:q0, :q1, :q2, :q3)
  n = GTPSA.numcoefs(q_expected.q0)
  all_ok = true
    for cname in components
      v_expected = getfield(q_expected, cname)
      v_calculated = sgn * getfield(q_calculated, cname)
      for j in 0:n-1
          c1, c2 = v_expected[j], v_calculated[j]
          if abs(c1 - c2) > max(ϵ, ϵ * (abs(c1) + abs(c2)))
              println("Coefficients not equal: expected $cname[$j] = $c1, got $cname[$j] = $c2")
              println("Difference: $(abs(c1 - c2))")
              println("Tolerance:  $(max(ϵ, ϵ * (abs(c1) + abs(c2))))")
              all_ok = false
              break
          end
      end
      if !all_ok
          break
      end
  end
  return all_ok
end

# The suite is sharded across CI jobs (see .github/workflows/CI.yml), which set
# BEAMTRACKING_TEST_GROUP. With the variable unset -- `Pkg.test()`, or running
# this file directly -- every group runs, exactly as before.
const TEST_GROUPS = Dict(
  "core" => ["miscellaneous_test.jl",
             "FieldSources_test.jl",
             "sagan_cavity_tracking_test.jl",
             "BeamlinesExt_test.jl",
             "alignment_tracking_test.jl",
             "aperture_tracking_test.jl",
             "ExactTracking_test.jl",
             "IntegrationTracking_test.jl",
             "collective_test.jl",
             "callback_test.jl",
             "ImplicitTracking_test.jl",
             "RungeKuttaTracking_test.jl"],
  # By far the longest-running group; kept on its own so it does not set the
  # wall-clock floor for everything else.
  "symplectic" => ["BeamlinesExt_symplectic_test.jl"],
  "batch_time" => ["batch_test.jl",
                   "time_test.jl"],
)

const GROUP_ORDER = ["core", "symplectic", "batch_time"]

# A test file that belongs to no group would silently stop running in CI.
let assigned = reduce(vcat, (TEST_GROUPS[g] for g in GROUP_ORDER)),
    on_disk = filter(f -> endswith(f, "_test.jl"), readdir(@__DIR__))
  unassigned = setdiff(on_disk, assigned)
  isempty(unassigned) ||
    error("test file(s) not assigned to a group in runtests.jl: " * join(sort(unassigned), ", "))
end

const GROUP = get(ENV, "BEAMTRACKING_TEST_GROUP", "all")
const TEST_FILES = if GROUP == "all"
  reduce(vcat, (TEST_GROUPS[g] for g in GROUP_ORDER))
else
  haskey(TEST_GROUPS, GROUP) ||
    error("unknown BEAMTRACKING_TEST_GROUP=$GROUP; expected \"all\" or one of " * join(GROUP_ORDER, ", "))
  TEST_GROUPS[GROUP]
end

@info "Running test group \"$GROUP\"" TEST_FILES
foreach(include, TEST_FILES)
