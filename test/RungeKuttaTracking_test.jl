function rk_test_uniform_field(x, y, z, s, parameters)
  carrier = zero(x)
  return EMField(
    carrier + parameters.Ex,
    carrier + parameters.Ey,
    carrier + parameters.Ez,
    carrier + parameters.Bx,
    carrier + parameters.By,
    carrier + parameters.Bz,
  )
end

function rk_test_parameter_free_field(x, y, z, s)
  carrier = zero(x)
  return EMField(carrier, carrier, carrier, carrier, carrier, carrier + 1)
end

struct RKCustomField{T}
  strength::T
end

function (source::RKCustomField)(x, y, z, s)
  v = zero(x)
  return EMField(v, v, v, v, v + source.strength, v)
end

@testset "RungeKuttaTracking" begin
  using BeamTracking
  using BeamTracking: Species, massof, chargeof, R_to_beta_gamma, R_to_pc, pc_to_R,
                      RungeKuttaTracking, Bunch, STATE_ALIVE, STATE_LOST_PZ, E_CHARGE, C_LIGHT
  using StaticArrays

  # Match the batch tests and the compiler-bug guard in beval: batched
  # explicit SIMD is unsupported on Julia < 1.11 with x86_64 CPUs.
  batch_simd_supported = !(VERSION < v"1.11" && Sys.ARCH == :x86_64)

  # Helper function to setup tracking parameters
  function setup_particle(pc=1e9)  # pc in eV, default corresponds to 1 GeV
    species = Species("electron")
    mc2 = massof(species)  # eV
    p_over_q_ref = pc_to_R(species, pc)

    # Calculate tracking parameters
    beta_gamma_0 = R_to_beta_gamma(species, p_over_q_ref)
    tilde_m = 1 / beta_gamma_0
    gamsqr_0 = 1 + beta_gamma_0^2
    beta_0 = beta_gamma_0 / sqrt(gamsqr_0)
    charge = chargeof(species)
    p0c = R_to_pc(species, p_over_q_ref)

    return species, p_over_q_ref, beta_0, gamsqr_0, tilde_m, charge, p0c, mc2
  end

  @testset "On-axis electric acceleration" begin
    # Analytic on-axis electric acceleration, including the beta-dependent z term.
    m, beta0, Ez, z = 2.0, 1/sqrt(5.0), 0.03, 0.2
    rhs = RungeKuttaTracking.kick_vector(0., 0., 0., 0., z, 0., 0.,
      EMField(0., 0., Ez, 0., 0., 0.), 1., m, beta0, 0., 0., 1., m)
    @test rhs[6] ≈ Ez / beta0
    @test rhs[5] ≈ m^2 * beta0 * Ez * z
  end

  @testset "RungeKutta constructor" begin
    using BeamTracking: RungeKutta

    # Test default constructor (no arguments)
    rk_default = RungeKutta()
    @test rk_default.ds_step == 0.2
    @test rk_default.n_steps == -1

    # Test constructor with ds_step only
    rk_ds = RungeKutta(ds_step=0.1)
    @test rk_ds.ds_step == 0.1
    @test rk_ds.n_steps == -1

    # Test constructor with n_steps only
    rk_ns = RungeKutta(n_steps=50)
    @test rk_ns.ds_step == -1.0
    @test rk_ns.n_steps == 50

    # Test constructor with both ds_step and n_steps (should error)
    @test_throws ErrorException RungeKutta(ds_step=0.1, n_steps=50)

    # Test constructor with explicit nothing values (should use defaults)
    rk_nothing = RungeKutta(ds_step=nothing, n_steps=nothing)
    @test rk_nothing.ds_step == 0.2
    @test rk_nothing.n_steps == -1

    replacement = MultipoleField(SA[1], SA[0.01], SA[0.0])
    rk_replacement = RungeKutta(field=replacement)
    @test rk_replacement.field === replacement
    @test isnothing(rk_replacement.additional_field)

    additional = FunctionalField(rk_test_parameter_free_field)
    rk_additional = RungeKutta(additional_field=additional)
    @test rk_additional.additional_field === additional
    @test isnothing(rk_additional.field)
  end

  @testset "Pure drift" begin
    species, p_over_q_ref, beta_0, gamsqr_0, tilde_m, charge, p0c, mc2 = setup_particle()

    # Create bunch with small transverse momentum
    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch.coords.v[1, BeamTracking.PXI] = 0.01

    L = 1.0
    ds_step = 0.01
    n_steps = 100
    gx = 0.0
    gy = 0.0
    
    source = ZeroField()

    RungeKuttaTracking.rk4_kernel!(1, bunch.coords, beta_0, tilde_m,
                                   charge, p0c, mc2, L, ds_step, n_steps, gx, gy,
                                   source)

    # Regression test
    solution = [0.0100005  0.01  0.0  0.0  -5.00038e-5  0.0]
    @test isapprox(bunch.coords.v, solution, rtol=1e-6)
    @test bunch.coords.state[1] == STATE_ALIVE
  end

  @testset "Solenoid" begin
    species, p_over_q_ref, beta_0, gamsqr_0, tilde_m, charge, p0c, mc2 = setup_particle(1e9)

    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch.coords.v[1, BeamTracking.PXI] = 0.01

    L = 1.0
    ds_step = 0.01
    n_steps = 100
    gx = 0.0
    gy = 0.0
    
    # Solenoid field
    Bz_physical = 0.01  # Tesla
    source = MultipoleField(SA[0], SA[Bz_physical], SA[0.0])

    RungeKuttaTracking.rk4_kernel!(1, bunch.coords, beta_0, tilde_m,
                                   charge, p0c, mc2, L, ds_step, n_steps, gx, gy,
                                   source)

    # In uniform B-field, particle should follow circular path
    # Total transverse momentum should be conserved
    pt2 = bunch.coords.v[1, 2]^2 + bunch.coords.v[1, 4]^2
    @test isapprox(pt2, 0.01^2, rtol=1e-4)
    # Regression test
    solution = [0.010000485056009705 0.009999955057780502 1.4991110783291216e-5 2.9980699961334158e-5 -5.000375031233078e-5 0.0]
    @test isapprox(bunch.coords.v, solution, rtol=1e-6)
    @test bunch.coords.state[1] == STATE_ALIVE
  end

  @testset "Dipole" begin
    species, p_over_q_ref, beta_0, gamsqr_0, tilde_m, charge, p0c, mc2 = setup_particle(1e9)

    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch.coords.v[1, BeamTracking.PXI] = 0.01

    L = 1.0
    ds_step = 0.01
    n_steps = 100
    gx = 0.0
    gy = 0.0
    
    # Dipole field
    By_physical = 0.01  # Tesla
    source = MultipoleField(SA[1], SA[By_physical], SA[0.0])

    RungeKuttaTracking.rk4_kernel!(1, bunch.coords, beta_0, tilde_m,
                                   charge, p0c, mc2, L, ds_step, n_steps, gx, gy,
                                   source)

    # Regression test
    solution = [0.011499735519796054 0.012997924579999955 0.0 0.0 -6.649432859025015e-5 0.0]
    @test isapprox(bunch.coords.v, solution, rtol=1e-6)
    @test bunch.coords.state[1] == STATE_ALIVE
  end

  @testset "Particle loss detection" begin
    species, p_over_q_ref, beta_0, gamsqr_0, tilde_m, charge, p0c, mc2 = setup_particle(1e9)

    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch.coords.v[1, BeamTracking.PXI] = 1.5 # Unphysical initial momentum

    L = 1.0
    ds_step = 0.1  # 10 cm step size
    n_steps = 10
    gx = 0.0
    gy = 0.0
    
    source = ZeroField()

    RungeKuttaTracking.rk4_kernel!(1, bunch.coords, beta_0, tilde_m,
                                   charge, p0c, mc2, L, ds_step, n_steps, gx, gy,
                                   source)

    # Particle should not track
    solution = [0.0  1.5  0.0  0.0  0.0  0.0]
    @test isapprox(bunch.coords.v, solution, rtol=1e-6)
    @test bunch.coords.state[1] == STATE_LOST_PZ
  end

  @testset "Invalid RK momenta and rejected steps" begin
    species, R, beta0, _, m, charge, pc, mc2 = setup_particle()
    # Include healthy and previously lost particles in the same SIMD lanes.
    initial = [0.0 0.0 0.0 0.0 0.0 -1.0;
               0.0 0.0 0.0 0.0 0.0 -2.0;
               0.0 0.0 0.0 0.0 0.0  0.0;
               0.0 0.1 0.0 0.0 0.0  0.0;
               0.0 1.0 0.0 0.0 0.0  0.0;
               0.0 NaN 0.0 0.0 0.0  0.0;
               0.0 0.0 0.0 0.0 0.0  Inf;
               0.0 0.0 0.0 0.0 0.0  0.0]
    for (use_KA, use_explicit_SIMD) in ((false, false), (false, true), (true, false))
      bunch = Bunch(copy(initial); species, p_over_q_ref=R)
      bunch.coords.state[8] = BeamTracking.STATE_LOST_POS_X
      call = BeamTracking.make_kernel_call(RungeKuttaTracking.rk4_kernel!,
        (beta0, m, charge, pc, mc2, 1.0, 1.0, 1, 0.0, 0.0, ZeroField()))
      BeamTracking.launch!(bunch.coords, call; use_KA, use_explicit_SIMD)
      @test bunch.coords.state == [STATE_LOST_PZ, STATE_LOST_PZ, STATE_ALIVE,
        STATE_ALIVE, STATE_LOST_PZ, STATE_LOST_PZ, STATE_LOST_PZ, BeamTracking.STATE_LOST_POS_X]
      @test isequal(bunch.coords.v[[1, 2, 5, 6, 7, 8], :], initial[[1, 2, 5, 6, 7, 8], :])
      @test bunch.coords.v[4, 1] ≈ 0.1 / sqrt(1 - 0.1^2)
    end

    # A bad intermediate stage must reject the step even if zeroing its
    # derivative would make the weighted final momentum appear valid.
    intermediate_loss = MultipoleField(SA[1], SA[1.1 * R], SA[0.0])
    # All stage input momenta are valid here, but k4 makes the final px = -2.
    final_loss = FunctionalField((x, y, z, s, R) -> begin
      v = zero(x)
      by = v + ifelse(s == 1, 12 * R, zero(R))
      EMField(v, v, v, v, by, v)
    end, R)
    for source in (intermediate_loss, final_loss)
      for (use_KA, use_explicit_SIMD) in ((false, false), (false, true), (true, false))
        bunch = Bunch(zeros(8, 6); species, p_over_q_ref=R)
        call = BeamTracking.make_kernel_call(RungeKuttaTracking.rk4_kernel!,
          (beta0, m, charge, pc, mc2, 1.0, 1.0, 1, 0.0, 0.0, source))
        BeamTracking.launch!(bunch.coords, call; use_KA, use_explicit_SIMD)
        @test all(==(STATE_LOST_PZ), bunch.coords.state)
        @test iszero(bunch.coords.v)
      end
      # Direct users of rk4_step! need the same loss handling as the kernel.
      bunch = Bunch(zeros(1, 6); species, p_over_q_ref=R)
      RungeKuttaTracking.rk4_step!(bunch.coords, 1, 0.0, 1.0, source,
        charge, m, beta0, 0.0, 0.0, pc, mc2)
      @test bunch.coords.state[1] == STATE_LOST_PZ
      @test iszero(bunch.coords.v)
    end

    # Electric deceleration may also make total momentum nonpositive midstep.
    source = FunctionalField((x, y, z, s) -> begin
      v = zero(x)
      EMField(v, v, v + 4 * pc, v, v, v)
    end)
    bunch = Bunch(zeros(1, 6); species, p_over_q_ref=R)
    RungeKuttaTracking.rk4_step!(bunch.coords, 1, 0.0, 1.0, source,
      charge, m, beta0, 0.0, 0.0, pc, mc2)
    @test bunch.coords.state[1] == STATE_LOST_PZ
    @test iszero(bunch.coords.v)
  end

  @testset "Convergence test" begin
    species, p_over_q_ref, beta_0, gamsqr_0, tilde_m, charge, p0c, mc2 = setup_particle(1e9)

    bunch1 = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch2 = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch1.coords.v[1, BeamTracking.PXI] = 0.01
    bunch2.coords.v[1, BeamTracking.PXI] = 0.01

    L = 1.0
    gx = 0.0
    gy = 0.0
    
    source = ZeroField()

    # Track with different step sizes
    RungeKuttaTracking.rk4_kernel!(1, bunch1.coords, beta_0, tilde_m,
                                   charge, p0c, mc2, L, 0.1, 10, gx, gy,
                                   source)
    RungeKuttaTracking.rk4_kernel!(1, bunch2.coords, beta_0, tilde_m,
                                   charge, p0c, mc2, L, 0.05, 20, gx, gy,
                                   source)

    # Results should be identical
    @test isapprox(bunch1.coords.v, bunch2.coords.v, rtol=1e-2)
  end

  @testset "Beamlines integration - Drift" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch.coords.v[1, BeamTracking.PXI] = 0.01

    drift_ele = Drift(L=1.0)
    drift_ele.tracking_method = RungeKutta()
    drift_line = Beamline([drift_ele], p_over_q_ref=p_over_q_ref, species_ref=species)

    track!(bunch, drift_line)

    # Regression test
    solution = [0.0100005  0.01  0.0  0.0  -5.00038e-5  0.0]
    @test isapprox(bunch.coords.v, solution, rtol=1e-6)
  end

  @testset "Beamlines integration - SBend" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch.coords.v[1, BeamTracking.PXI] = 0.01

    sbend_ele = SBend(L=1.0, angle=pi/132)
    sbend_ele.tracking_method = RungeKutta()
    sbend_line = Beamline([sbend_ele], p_over_q_ref=p_over_q_ref, species_ref=species)

    track!(bunch, sbend_line)

    # Regression test
    solution = [0.010000150630002367 0.009995978032305387 0.0 0.0 -0.00016899908120890584 0.0]
    @test isapprox(bunch.coords.v, solution, rtol=1e-6)
  end

  @testset "Beamlines physical and normalized multipoles" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    initial = [0.001 0.01 -0.002 0.003 0.0 0.0]
    normalized_strength = 0.2
    physical_strength = normalized_strength * p_over_q_ref

    normalized_element = Quadrupole(
      L=0.5,
      Kn1=normalized_strength,
      tracking_method=RungeKutta(n_steps=5),
    )
    physical_element = Quadrupole(
      L=0.5,
      Bn1=physical_strength,
      tracking_method=RungeKutta(n_steps=5),
    )
    normalized_line = Beamline(
      [normalized_element],
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    physical_line = Beamline(
      [physical_element],
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    normalized_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    physical_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)

    track!(normalized_bunch, normalized_line)
    track!(physical_bunch, physical_line)

    @test normalized_bunch.coords.v ≈ physical_bunch.coords.v
  end

  @testset "Beamlines configured field sources" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    initial = [0.001 0.01 -0.002 0.003 0.0 0.0]
    element_strength = 0.2
    external = FunctionalField(
      rk_test_uniform_field,
      (Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=0.004, Bz=0.0),
    )

    element = Quadrupole(
      L=0.5,
      Kn1=element_strength,
      tracking_method=RungeKutta(additional_field=external, n_steps=5),
    )
    composed = SumField(
      MultipoleField(SA[2], SA[element_strength * p_over_q_ref], SA[0.0]),
      external,
    )
    reference = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=composed, n_steps=5),
    )

    element_line = Beamline([element], p_over_q_ref=p_over_q_ref, species_ref=species)
    reference_line = Beamline([reference], p_over_q_ref=p_over_q_ref, species_ref=species)
    element_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    reference_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)

    track!(element_bunch, element_line)
    track!(reference_bunch, reference_line)

    @test element_bunch.coords.v ≈ reference_bunch.coords.v

    replacement_element = Quadrupole(
      L=0.5,
      Kn1=5 * element_strength,
      tracking_method=RungeKutta(field=external, n_steps=5),
    )
    replacement_reference = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=external, n_steps=5),
    )
    replacement_line = Beamline(
      [replacement_element],
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    replacement_reference_line = Beamline(
      [replacement_reference],
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    replacement_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    replacement_reference_bunch = Bunch(
      copy(initial),
      p_over_q_ref=p_over_q_ref,
      species=species,
    )

    track!(replacement_bunch, replacement_line)
    track!(replacement_reference_bunch, replacement_reference_line)

    @test replacement_bunch.coords.v ≈ replacement_reference_bunch.coords.v
  end

  @testset "Beamlines field-source context" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    initial = [0.001 0.01 -0.002 0.003 0.0 0.0]
    context = Context(dipole=0.003, external=0.004)
    context_source = SumField(
      MultipoleField(
        SA[1],
        SA[DefExpr{Float64}(c -> c.dipole)],
        SA[DefExpr{Float64}(c -> 0.0)],
      ),
      FunctionalField(
        rk_test_uniform_field,
        (
          Ex=0.0,
          Ey=0.0,
          Ez=0.0,
          Bx=0.0,
          By=DefExpr{Float64}(c -> c.external),
          Bz=0.0,
        ),
      ),
    )
    fixed_source = SumField(
      MultipoleField(SA[1], SA[context.dipole], SA[0.0]),
      FunctionalField(
        rk_test_uniform_field,
        (Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=context.external, Bz=0.0),
      ),
    )
    context_element = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=context_source, n_steps=5),
    )
    fixed_element = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=fixed_source, n_steps=5),
    )
    context_line = Beamline(
      [context_element],
      context=context,
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    fixed_line = Beamline(
      [fixed_element],
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    context_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    fixed_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)

    track!(context_bunch, context_line)
    track!(fixed_bunch, fixed_line)

    @test context_bunch.coords.v ≈ fixed_bunch.coords.v
  end

  @testset "Functional field-source tracking" begin
    species, p_over_q_ref, beta_0, _, tilde_m, charge, p0c, mc2 = setup_particle()
    context = Context(strength=0.004)
    initial = repeat([0.001 0.01 -0.002 0.003 0.0 0.0], 8, 1)
    initial[:, 5] .= range(0.0, -0.2, length=8)
    beta_gamma_ref = R_to_beta_gamma(species, p_over_q_ref)
    times = [BeamTracking.compute_time(initial[i, 5], initial[i, 6], 0.0, beta_gamma_ref)
             for i in axes(initial, 1)]
    cases = (
      (DefExpr{Float64}(c -> c.strength), fill(context.strength, 8), false),
      (DefExpr{BatchParam}(c -> BatchParam([c.strength, 2*c.strength])),
       repeat([context.strength, 2*context.strength], 4), false),
      (DefExpr{TimeDependentParam}(c -> c.strength + 1e6*Time()),
       context.strength .+ 1e6 .* times, false),
      (ForwardDiff.Dual(context.strength, 1.0), fill(context.strength, 8), true),
    )
    for (strength, expected_strengths, scalar_params) in cases
      functional = FunctionalField(
        (x, y, z, s, p) -> RKCustomField(p.strength)(x, y, z, s),
        (strength=strength,),
      )
      sources = (
        functional,
        SumField(functional, RKCustomField(0.0)),
      )
      expected = similar(initial)
      for i in axes(initial, 1)
        fixed = MultipoleField(SA[1], SA[expected_strengths[i]], SA[0.0])
        line = Beamline([Drift(L=0.5, tracking_method=RungeKutta(field=fixed, n_steps=5))],
                        p_over_q_ref=p_over_q_ref, species_ref=species)
        bunch = Bunch(copy(initial[i:i, :]), p_over_q_ref=p_over_q_ref, species=species)
        track!(bunch, line; use_KA=false, use_explicit_SIMD=false)
        expected[i, :] .= bunch.coords.v[1, :]
      end
      for source in sources, (use_KA, use_explicit_SIMD) in ((false, false), (false, true), (true, false))
        if strength isa DefExpr{BatchParam} && use_explicit_SIMD && !batch_simd_supported
          continue
        end
        line = Beamline([Drift(L=0.5, tracking_method=RungeKutta(field=source, n_steps=5))],
                        context=context, p_over_q_ref=p_over_q_ref, species_ref=species)
        bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
        track!(bunch, line; scalar_params, use_KA, use_explicit_SIMD)
        @test bunch.coords.v ≈ expected
      end
    end

    source = FunctionalField(
      (x, y, z, s, p) -> RKCustomField(p.strength)(x, y, z, s),
      (strength=BatchParam([0.002, 0.004]),),
    )
    call = BeamTracking.make_kernel_call(RungeKuttaTracking.rk4_kernel!, (
      beta_0, tilde_m, charge, p0c, mc2, 0.5, 0.1, 5, 0.0, 0.0, source,
    ))
    bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    for simd in (batch_simd_supported ? (false, true) : (false,))
      @test @ballocated(BeamTracking.launch!($bunch.coords, $call;
                       use_KA=false, use_explicit_SIMD=$simd)) == 0
    end
  end

  @testset "Float32 field-source tracking" begin
    species, R, = setup_particle()
    # Check every RK stage, including SIMD lanes, not just storage.
    evaluator = (x, y, z, s, p) -> begin
      @assert eltype(x) === eltype(y) === eltype(z) === eltype(s) === Float32
      @assert eltype(p.By) === Float32
      v = zero(x)
      EMField(v, v, v, v, v + p.By, v)
    end
    for (use_KA, use_explicit_SIMD) in ((false, false), (false, true), (true, false))
      # Float32 batch gathers currently hit an upstream SIMD pointer-cast bug.
      # Cover SIMD with static parameters and batches on scalar/KA paths.
      strength = use_explicit_SIMD ? BatchParam(0.002) : BatchParam([0.002, 0.004])
      source = SumField(
        MultipoleField(SA[1], SA[0.001], SA[0.0]),
        FunctionalField(evaluator, (By=strength,)),
      )
      line = Beamline([Drift(L=0.5, tracking_method=RungeKutta(field=source, n_steps=5))],
                      p_over_q_ref=R, species_ref=species)
      reference = Beamline([Drift(L=0.5, tracking_method=RungeKutta(
        field=MultipoleField(SA[1], SA[strength + 0.001], SA[BatchParam(0.0)]),
        n_steps=5))], p_over_q_ref=R, species_ref=species)
      bunch = Bunch(zeros(Float32, 16, 6); species, p_over_q_ref=R)
      expected = Bunch(zeros(16, 6); species, p_over_q_ref=R)
      track!(expected, reference; use_KA=false, use_explicit_SIMD=false)
      track!(bunch, line; use_KA, use_explicit_SIMD)
      # Near-zero z includes cancellation of order-one terms in Float32.
      @test all(isapprox.(bunch.coords.v, expected.coords.v; rtol=1e-5, atol=eps(Float32)))
    end
  end

  @testset "Batch field-source tracking" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    batch_fields = [0.002, 0.004]
    initial_particle = [0.001 0.01 -0.002 0.003 0.0 0.0]
    initial = repeat(initial_particle, 8, 1)
    source = FunctionalField(
      rk_test_uniform_field,
      (Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=BatchParam(batch_fields), Bz=0.0),
    )
    element = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=source, n_steps=5),
    )
    line = Beamline([element], p_over_q_ref=p_over_q_ref, species_ref=species)
    simd_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    ka_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)

    track!(simd_bunch, line; use_KA=false, use_explicit_SIMD=batch_simd_supported)
    track!(ka_bunch, line; use_KA=true, use_explicit_SIMD=false)

    expected = similar(initial)
    for i in axes(initial, 1)
      fixed_source = FunctionalField(
        rk_test_uniform_field,
        (
          Ex=0.0,
          Ey=0.0,
          Ez=0.0,
          Bx=0.0,
          By=batch_fields[mod1(i, length(batch_fields))],
          Bz=0.0,
        ),
      )
      fixed_element = Drift(
        L=0.5,
        tracking_method=RungeKutta(field=fixed_source, n_steps=5),
      )
      fixed_line = Beamline(
        [fixed_element],
        p_over_q_ref=p_over_q_ref,
        species_ref=species,
      )
      fixed_bunch = Bunch(
        copy(initial[i:i, :]),
        p_over_q_ref=p_over_q_ref,
        species=species,
      )
      track!(fixed_bunch, fixed_line; use_KA=false, use_explicit_SIMD=false)
      expected[i, :] .= fixed_bunch.coords.v[1, :]
    end

    @test simd_bunch.coords.v ≈ expected
    @test ka_bunch.coords.v ≈ expected
  end

  @testset "Scalarized field-source tracking" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    initial = [0.001 0.01 -0.002 0.003 0.0 0.0]
    dual_field = ForwardDiff.Dual(0.004, 1.0)
    source = FunctionalField(
      rk_test_uniform_field,
      (Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=dual_field, Bz=0.0),
    )
    fixed_source = FunctionalField(
      rk_test_uniform_field,
      (Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=ForwardDiff.value(dual_field), Bz=0.0),
    )
    element = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=source, n_steps=5),
    )
    fixed_element = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=fixed_source, n_steps=5),
    )
    line = Beamline([element], p_over_q_ref=p_over_q_ref, species_ref=species)
    fixed_line = Beamline(
      [fixed_element],
      p_over_q_ref=p_over_q_ref,
      species_ref=species,
    )
    bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)
    fixed_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)

    track!(bunch, line; scalar_params=true)
    track!(fixed_bunch, fixed_line)

    @test bunch.coords.v ≈ fixed_bunch.coords.v
  end

  @testset "Time-dependent field-source tracking" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    beta_gamma_ref = R_to_beta_gamma(species, p_over_q_ref)
    initial = [
      0.001 0.01 -0.002 0.003  0.00  0.0
      0.001 0.01 -0.002 0.003 -0.05  0.0
      0.001 0.01 -0.002 0.003 -0.10  0.0
      0.001 0.01 -0.002 0.003 -0.15  0.0
    ]
    field_at_time = 0.002 + 1.0e6 * Time()
    source = FunctionalField(
      rk_test_uniform_field,
      (Ex=0.0, Ey=0.0, Ez=0.0, Bx=0.0, By=field_at_time, Bz=0.0),
    )
    element = Drift(
      L=0.5,
      tracking_method=RungeKutta(field=source, n_steps=5),
    )
    line = Beamline([element], p_over_q_ref=p_over_q_ref, species_ref=species)
    dynamic_bunch = Bunch(copy(initial), p_over_q_ref=p_over_q_ref, species=species)

    track!(dynamic_bunch, line; use_KA=false, use_explicit_SIMD=true)

    expected = similar(initial)
    for i in axes(initial, 1)
      particle_time = BeamTracking.compute_time(
        initial[i, BeamTracking.ZI],
        initial[i, BeamTracking.PZI],
        0.0,
        beta_gamma_ref,
      )
      fixed_source = FunctionalField(
        rk_test_uniform_field,
        (
          Ex=0.0,
          Ey=0.0,
          Ez=0.0,
          Bx=0.0,
          By=field_at_time(particle_time),
          Bz=0.0,
        ),
      )
      fixed_element = Drift(
        L=0.5,
        tracking_method=RungeKutta(field=fixed_source, n_steps=5),
      )
      fixed_line = Beamline(
        [fixed_element],
        p_over_q_ref=p_over_q_ref,
        species_ref=species,
      )
      fixed_bunch = Bunch(
        copy(initial[i:i, :]),
        p_over_q_ref=p_over_q_ref,
        species=species,
      )
      track!(fixed_bunch, fixed_line; use_KA=false, use_explicit_SIMD=false)
      expected[i, :] .= fixed_bunch.coords.v[1, :]
    end

    @test dynamic_bunch.coords.v ≈ expected
  end

  @testset "RungeKutta with different step configurations" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()

    # Test with ds_step
    drift_ds = Drift(L=1.0)
    drift_ds.tracking_method = RungeKutta(ds_step=0.1)
    line_ds = Beamline([drift_ds], p_over_q_ref=p_over_q_ref, species_ref=species)
    bunch_ds = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch_ds.coords.v[1, BeamTracking.PXI] = 0.01
    track!(bunch_ds, line_ds)

    # Test with n_steps
    drift_ns = Drift(L=1.0)
    drift_ns.tracking_method = RungeKutta(n_steps=10)
    line_ns = Beamline([drift_ns], p_over_q_ref=p_over_q_ref, species_ref=species)
    bunch_ns = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    bunch_ns.coords.v[1, BeamTracking.PXI] = 0.01
    track!(bunch_ns, line_ns)

    # Both should give the same results
    @test isapprox(bunch_ds.coords.v, bunch_ns.coords.v, rtol=1e-2)
  end

  @testset "RungeKutta step selection" begin
    @test BeamTracking.find_steps(RungeKutta(ds_step=0.3), 1.0) == (4, 0.25)
    @test BeamTracking.find_steps(RungeKutta(n_steps=4), 1.0) == (4, 0.25)
  end

  @testset "Tilted reference curvature" begin
    _, p_over_q_ref, beta_0, _, tilde_m, charge, p0c, mc2 = setup_particle()
    zero_field = ntuple(_ -> 0.0, 6)

    horizontal = RungeKuttaTracking.kick_vector(
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero_field...,
      charge, tilde_m, beta_0, 0.1, 0.0, p0c, mc2,
    )
    vertical = RungeKuttaTracking.kick_vector(
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero_field...,
      charge, tilde_m, beta_0, 0.0, 0.1, p0c, mc2,
    )

    @test horizontal[2] ≈ 0.1
    @test horizontal[4] ≈ 0.0
    @test vertical[2] ≈ 0.0
    @test vertical[4] ≈ 0.1
  end

  @testset "RungeKutta callbacks" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    s_values = Float64[]
    ds_values = Float64[]
    function save_position!(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref,
                            last_ds_step, last_g, transforms_out!, transforms_in!)
      push!(s_values, cur_s)
      push!(ds_values, last_ds_step)
    end

    ele = Drift(L=1.0, tracking_method=RungeKutta(n_steps=4))
    line = Beamline([ele], p_over_q_ref=p_over_q_ref, species_ref=species)
    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species,
                  callbacks=(save_position!,))

    track!(bunch, line)

    # Three internal callbacks follow completed non-final RK steps. The common
    # tracking path supplies the fourth callback after element-exit processing.
    @test s_values ≈ [0.25, 0.5, 0.75, 1.0]
    @test ds_values ≈ fill(0.25, 4)
  end

  @testset "Per-particle reference ramp" begin
    using Beamlines

    species = Species("electron")
    E_ref = TimeDependentParam(t -> 1e9 * (1 + 1e6 * t), false)
    ele = Drift(L=1.0, tracking_method=RungeKutta(n_steps=4))
    line = Beamline([ele], E_ref=E_ref, species_ref=species)
    p_over_q_ref = line.p_over_q_ref(0.0)
    beta_gamma_ref = BeamTracking.R_to_beta_gamma(species, p_over_q_ref)
    expected_t_ref = 1.0 / BeamTracking.beta_gamma_to_v(beta_gamma_ref)
    expected_p_over_q_ref = line.p_over_q_ref(expected_t_ref)
    bunch = Bunch([0.0 0.01 0.0 0.0 0.0 0.0;
                   0.0 0.01 0.0 0.0 -1.0 0.0],
                  p_over_q_ref=p_over_q_ref, species=species)

    track!(bunch, line; ramp_update_each_particle=true)

    @test bunch.t_ref ≈ expected_t_ref
    @test bunch.p_over_q_ref ≈ expected_p_over_q_ref
    @test all(isfinite, bunch.coords.v)
  end

  @testset "Unsupported bend edge angles" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()
    bunch = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)

    entrance_edge = SBend(L=1.0, g_ref=0.1, e1=0.01,
                          tracking_method=RungeKutta())
    entrance_line = Beamline([entrance_edge], p_over_q_ref=p_over_q_ref,
                             species_ref=species)
    @test_throws ErrorException track!(bunch, entrance_line)

    exit_edge = SBend(L=1.0, g_ref=0.1, e2=0.01,
                      tracking_method=RungeKutta())
    exit_line = Beamline([exit_edge], p_over_q_ref=p_over_q_ref,
                         species_ref=species)
    @test_throws ErrorException track!(bunch, exit_line)
  end

  @testset "Zero-length elements" begin
    using Beamlines

    species, p_over_q_ref, _, _, _, _, _, _ = setup_particle()

    # Test zero-length drift should throw an error
    drift_zero = Drift(L=0.0)
    drift_zero.tracking_method = RungeKutta()
    line_zero = Beamline([drift_zero], p_over_q_ref=p_over_q_ref, species_ref=species)
    bunch_drift = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    
    @test_throws ErrorException track!(bunch_drift, line_zero)

    # Test negative length should also throw an error
    drift_negative = Drift(L=-0.1)
    drift_negative.tracking_method = RungeKutta()
    line_negative = Beamline([drift_negative], p_over_q_ref=p_over_q_ref, species_ref=species)
    bunch_negative = Bunch(zeros(1, 6), p_over_q_ref=p_over_q_ref, species=species)
    
    @test_throws ErrorException track!(bunch_negative, line_negative)
  end

end
