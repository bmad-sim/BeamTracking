# RK4 solenoid benchmark: 100 steps over 1 m, Float64, no spin.
# Run in an environment with BeamTracking, BenchmarkTools, and StaticArrays:
#   julia --project=benchmark --threads=8 benchmark/rk4_kernel_benchmark.jl [--gpu=auto|off|cuda] [results.csv [particle_counts...]]
# CUDA is optional. Auto mode uses it when available; off never imports it;
# cuda requires a working device. See --help for output and timing details.

using BeamTracking
using BeamTracking: Species, massof, chargeof, R_to_beta_gamma, R_to_pc, pc_to_R,
                    RungeKuttaTracking, Bunch, STATE_ALIVE
using StaticArrays, BenchmarkTools, Random, Printf, TOML, Dates, Sockets

const DEFAULT_PARTICLE_COUNTS = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)

function parse_options(args)
    gpu = "auto"
    positional = String[]

    for arg in args
        if arg in ("--help", "-h")
            print("""
                Usage:
                  julia --project=benchmark benchmark/rk4_kernel_benchmark.jl
                    [--gpu=auto|off|cuda] [results.csv [particle_counts...]]

                Outputs:
                  CSV timings and a companion .metadata.toml file describing
                  the machine and settings.

                Timing:
                  Total time for the entire bunch through 100 RK4 steps.
                  Includes launch and synchronization.
                  Excludes compilation, resets, and transfers.
                """)
            return nothing
        elseif startswith(arg, "--gpu=")
            gpu = split(arg, '='; limit=2)[2]
            gpu in ("auto", "off", "cuda") || error("--gpu must be auto, off, or cuda")
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
        end
    end

    output = isempty(positional) ? "rk4_results.csv" : positional[1]
    counts = length(positional) > 1 ?
        parse.(Int, positional[2:end]) : collect(DEFAULT_PARTICLE_COUNTS)
    all(>(0), counts) || error("Particle counts must be positive")
    length(unique(counts)) == length(counts) || error("Particle counts must be unique")

    return (; gpu, output, counts)
end

function select_gpu(mode)
    mode == "off" && return (false, "disabled by --gpu=off")

    reason = if isnothing(Base.find_package("CUDA"))
        "CUDA.jl is not installed in the active environment"
    else
        try
            @eval import CUDA
            # CUDA methods were loaded dynamically in this function.
            Base.invokelatest(CUDA.functional) && return (true, "CUDA is functional")
            "CUDA.jl is installed, but no functional CUDA device is available"
        catch err
            err isa InterruptException && rethrow()
            "CUDA initialization failed: " * sprint(showerror, err)
        end
    end

    mode == "cuda" && error("GPU requested with --gpu=cuda: $reason")
    return (false, reason)
end

function setup_particle(pc=1e9)
    species = Species("electron")
    mc2 = massof(species)
    p_over_q_ref = pc_to_R(species, pc)

    beta_gamma_0 = R_to_beta_gamma(species, p_over_q_ref)
    tilde_m = 1 / beta_gamma_0
    beta_0 = beta_gamma_0 / sqrt(1 + beta_gamma_0^2)
    charge = chargeof(species)
    p0c = R_to_pc(species, p_over_q_ref)

    return species, p_over_q_ref, beta_0, tilde_m, charge, p0c, mc2
end

function setup_multi_particle(n_particles)
    species, p_over_q_ref, beta_0, tilde_m, charge, p0c, mc2 = setup_particle(1e9)

    rng = MersenneTwister(1234)
    bunch = Bunch(
        randn(rng, n_particles, 6) * 0.001,
        p_over_q_ref=p_over_q_ref,
        species=species,
    )

    L = 1.0
    ds_step = 0.01
    n_steps = 100
    gx = 0.0
    gy = 0.0

    Bz_physical = 0.01
    source = MultipoleField(SA[0], SA[Bz_physical], SA[0.0])

    return bunch, beta_0, tilde_m, charge, p0c, mc2, L, ds_step, n_steps,
           gx, gy, source
end

function track_all_particles!(
    bunch, beta_0, tilde_m, charge, p0c, mc2,
    L, ds_step, n_steps, gx, gy, source,
)
    n = size(bunch.coords.v, 1)
    for i in 1:n
        RungeKuttaTracking.rk4_kernel!(
            i, bunch.coords, beta_0, tilde_m,
            charge, p0c, mc2, L, ds_step, n_steps,
            gx, gy, source,
        )
    end
    return nothing
end

function reset_scaling!(bunch, initial, gpu)
    copyto!(bunch.coords.v, initial)
    fill!(bunch.coords.state, STATE_ALIVE)
    gpu && CUDA.synchronize()
    return nothing
end

function run_scaling!(bunch, call, args, mode)
    if mode == :scalar
        track_all_particles!(bunch, args...)
    else
        BeamTracking.launch!(
            bunch.coords, call;
            use_KA=mode in (:ka_cpu, :cuda),
            use_explicit_SIMD=mode in (:simd, :threaded),
            use_cpu_multithreading=mode == :threaded,
        )
    end
    mode == :cuda && CUDA.synchronize()
    return nothing
end

function scaling_main(options, gpu_enabled, gpu_reason)
    output, particle_counts = options.output, options.counts
    modes = gpu_enabled ?
        (:scalar, :simd, :threaded, :ka_cpu, :cuda) :
        (:scalar, :simd, :threaded, :ka_cpu)

    println("GPU: ", gpu_reason)
    println("CPU: ", Sys.CPU_NAME, "; Julia threads: ", Threads.nthreads())
    if gpu_enabled
        CUDA.allowscalar(false)
        CUDA.versioninfo()
    end
    flush(stdout)

    metadata = Dict{String,Any}(
        "timestamp_utc" => string(now(UTC)),
        "hostname" => gethostname(),
        "cpu" => Sys.CPU_NAME,
        "threads" => Threads.nthreads(),
        "julia_version" => string(VERSION),
        "beamtracking_path" => pathof(BeamTracking),
        "beamtracking_version" => string(Base.pkgversion(BeamTracking)),
        "benchmarktools_version" => string(Base.pkgversion(BenchmarkTools)),
        "kernelabstractions_version" => string(Base.pkgversion(BeamTracking.KernelAbstractions)),
        "gpu_requested" => options.gpu,
        "gpu_enabled" => gpu_enabled,
        "gpu_status" => gpu_reason,
        "modes" => string.(collect(modes)),
        "particle_counts" => particle_counts,
        "steps" => 100,
        "length_m" => 1.0,
        "solenoid_tesla" => 0.01,
        "momentum_eV_c" => 1e9,
        "precision" => "Float64",
        "max_samples" => 15,
        "seconds_per_case" => 120,
        "evals_per_sample" => 1,
        "timing" => "Includes launch and synchronization; excludes compilation, reset, and transfers",
    )
    if gpu_enabled
        metadata["cuda_version"] = string(Base.pkgversion(CUDA))
        metadata["gpu_name"] = CUDA.name(CUDA.device())
    end
    open(splitext(output)[1] * ".metadata.toml", "w") do io
        TOML.print(io, metadata; sorted=true)
    end

    open(output, "w") do io
        println(
            io, "particles,steps,mode,threads,samples,median_ms,min_ms,max_ms,host_bytes,host_allocs,particles_per_second,max_abs_error",
        )
        for n in particle_counts
            cpu_bunch, args... = setup_multi_particle(n)
            initial = copy(cpu_bunch.coords.v)
            call = BeamTracking.make_kernel_call(RungeKuttaTracking.rk4_kernel!, Tuple(args))

            track_all_particles!(cpu_bunch, args...)
            expected = copy(cpu_bunch.coords.v)
            all(isfinite, expected) || error("Nonfinite scalar reference for N=$n")
            all(==(STATE_ALIVE), cpu_bunch.coords.state) ||
                error("Particle loss in scalar reference for N=$n")

            for mode in modes
                gpu = mode == :cuda
                bunch = gpu ? Bunch(
                    CUDA.CuArray(initial);
                    species=cpu_bunch.species,
                    p_over_q_ref=cpu_bunch.p_over_q_ref,
                ) : cpu_bunch
                saved = gpu ? CUDA.CuArray(initial) : initial

                # Warm up each specialization, including the reset and launch.
                for _ in 1:2
                    reset_scaling!(bunch, saved, gpu)
                    run_scaling!(bunch, call, Tuple(args), mode)
                end

                actual = Array(bunch.coords.v)
                all(isapprox.(actual, expected; atol=5e-13, rtol=5e-11)) ||
                    error("Coordinate mismatch: N=$n, mode=$mode")
                all(==(STATE_ALIVE), Array(bunch.coords.state)) ||
                    error("Particle loss: N=$n, mode=$mode")
                max_error = maximum(abs, actual - expected)

                # Exactly one tracking pass per sample. Reset/transfers/compilation
                # are outside timing; GPU completion is inside timing.
                trial = @benchmark(
                    run_scaling!($bunch, $call, $(Tuple(args)), $mode),
                    setup=(reset_scaling!($bunch, $saved, $gpu)),
                    evals=1,
                    samples=15,
                    seconds=120,
                    gctrial=false,
                )
                med = median(trial)
                @printf(
                    io, "%d,100,%s,%d,%d,%.9f,%.9f,%.9f,%d,%d,%.6f,%.6e\n",
                    n, mode, Threads.nthreads(), length(trial), med.time / 1e6,
                    minimum(trial).time / 1e6, maximum(trial).time / 1e6,
                    med.memory, med.allocs, n * 1e9 / med.time, max_error,
                )
                flush(io)
                @printf(
                    "N=%7d %-8s median=%10.4f ms min=%10.4f ms samples=%d host_allocs=%d\n",
                    n, mode, med.time / 1e6, minimum(trial).time / 1e6,
                    length(trial), med.allocs,
                )
                flush(stdout)

                gpu && CUDA.unsafe_free!(saved)
                gpu && CUDA.unsafe_free!(bunch.coords.v)
                gpu && CUDA.unsafe_free!(bunch.coords.state)
            end
        end
    end
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return
    enabled, reason = select_gpu(options.gpu)
    # Enter the latest world after the optional CUDA import.
    Base.invokelatest(scaling_main, options, enabled, reason)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
