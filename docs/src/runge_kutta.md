# Runge-Kutta Tracking

`RungeKutta` tracks particles through electromagnetic field sources with a
classical fourth-order Runge-Kutta (RK4) integrator. It uses the common tracking
path for reference-momentum updates, alignment, aperture checks, and callbacks.

## Configuration

Set either `ds_step` or `n_steps`:

- `ds_step` is the maximum requested step length in meters. The number of steps
  is `ceil(Int, L / ds_step)`, and the actual equal step length is `L / n_steps`.
- `n_steps` is the exact number of equal steps. The step length is
  `L / n_steps`.

Only one option can be positive. If neither option is set, the default is
`ds_step=0.2`. RK tracking requires a positive element length.

```julia
ele.tracking_method = RungeKutta()
ele.tracking_method = RungeKutta(ds_step=0.1)
ele.tracking_method = RungeKutta(n_steps=50)
```

## Field sources

A field source is a concrete callable object with the interface:

```julia
source(x, y, z, s) -> EMField
```

`EMField.E` is an `SVector` in V/m and `EMField.B` is an `SVector` in tesla.
RK tracking provides `ZeroField`, `MultipoleField`, `FunctionalField`, and
`SumField` source types.

- `ZeroField()` returns zero electric and magnetic fields.
- `MultipoleField(orders, normal, skew)` stores static multipole orders and
  non-integrated physical magnetic coefficients. Its field values are in
  tesla.
- `FunctionalField(evaluator, parameters)` stores the evaluator type and the
  parameter type in the source type. The evaluator receives
  `(x, y, z, s, parameters)` and returns an `EMField`.
- `FunctionalField(evaluator)` calls the evaluator with `(x, y, z, s)`.
- `SumField(sources...)` stores a tuple of concrete sources and evaluates the
  sum with static dispatch.

`field` sets the complete body field:

```julia
function uniform_field(x, y, z, s, parameters)
  v = zero(x)
  return EMField(
    v, v, v,
    v + parameters.Bx, v + parameters.By, v + parameters.Bz,
  )
end

source = FunctionalField(uniform_field, (Bx=0.0, By=0.1, Bz=0.0))
ele.tracking_method = RungeKutta(field=source, n_steps=20)
```

`additional_field` adds a source to the magnetic multipoles stored on the
element:

```julia
ele.tracking_method = RungeKutta(additional_field=source, n_steps=20)
```

The configured sources and their parameter types remain concrete in the RK
kernel.

### Custom sources

Custom callable objects can be passed directly to `field` or
`additional_field` when their fields are already concrete and do not require
parameter preparation:

```julia
struct UniformMagneticField{T}
  By::T
end

function (source::UniformMagneticField)(x, y, z, s)
  v = zero(x)
  return EMField(v, v, v, v, v + source.By, v)
end

ele.tracking_method = RungeKutta(field=UniformMagneticField(0.1))
```

## Beamlines usage

The element must get its reference data from a `Beamline`, in the same way as
the upstream tracking methods.

```julia
using BeamTracking, Beamlines

species = Species("electron")
ele = Quadrupole(L=1.0, Kn1=0.1,
                 tracking_method=RungeKutta(ds_step=0.1))
line = Beamline([ele], p_over_q_ref=-3.0, species_ref=species)
bunch = Bunch(zeros(100, 6), p_over_q_ref=line.p_over_q_ref,
              species=species)

track!(bunch, line)
```

The RK body kernel tracks static electric and magnetic fields. Beamlines
magnetic multipoles are represented by `MultipoleField`, including solenoid,
normal, and skew terms. Bend body tracking uses reference curvature with zero
edge angles.

## Time-dependent values and reference ramping

`ramp_update_each_particle=true` uses the upstream per-particle reference-ramp
path. Time-dependent values are evaluated once for each particle, using that
particle's time at the element entrance. Each evaluated value stays fixed
during every RK substep and during the `k1` through `k4` stages.

## Callbacks

RK calls internal callbacks after each completed non-final substep. Each RK
step completes its `k1`, `k2`, `k3`, and `k4` stages before the callback. The
common tracking path performs the final callback after element-exit processing.

## Low-level kernel

The low-level entry point is:

```julia
rk4_kernel!(i, coords, beta_0, tilde_m, charge, p0c, mc2,
            L, ds_step, n_steps, gx, gy, source)
```

`source` is a concrete callable field source. The kernel marks a particle as
`STATE_LOST_PZ` when its transverse velocity is unphysical.
