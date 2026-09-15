module BeamTracking
using GTPSA,
      ReferenceFrameRotations,
      StaticArrays, 
      SIMD,
      SIMDMathFunctions,
      HostCPUFeatures,
      Unrolled,
      MacroTools,
      Adapt,
      Accessors,
      SpecialFunctions,
      AtomicAndPhysicalConstants,
      Random,
      EnumX,
      Statistics,
      LinearAlgebra,
      TPSAInterface,
      ForwardDiff

using KernelAbstractions

import GTPSA: sincu, sinhcu

export Species
export Bunch, State, ParticleView, Time, TimeDependentParam, BatchParam
export Symplectic, MatrixKick, BendKick, SolenoidKick, DriftKick, Exact, RungeKutta
export EMField, ZeroField, MultipoleField, FunctionalField, SumField
export Fringe, SaganCavity, track!

include("utils/float32.jl")
include("utils/coord_transforms.jl")
include("utils/energy.jl")
include("utils/math_simd.jl")
include("utils/quaternions.jl")
include("utils/z_to_time.jl")
include("utils/beam_statistics.jl")
include("utils/ibs_integrals.jl")
include("utils/random.jl")

include("types.jl")
include("time.jl")
include("batch.jl")
include("callbacks.jl")
include("kernel.jl")
include("tracking_methods.jl")

include("kernels/kernel_utils.jl")
include("kernels/alignment.jl")
include("kernels/aperture.jl")
include("kernels/bend_kick.jl")
include("kernels/drift_kick.jl")
include("kernels/map.jl")
include("kernels/multipole.jl")
include("kernels/patch.jl")
include("kernels/quadrupole_kick.jl")
include("kernels/radiation.jl")
include("kernels/ramp_P0.jl")
include("kernels/rfcavity_kick.jl")
include("kernels/sagan_cavity.jl")
include("kernels/solenoid_kick.jl")
include("kernels/spin.jl")
include("kernels/transforms.jl")
include("kernels/integrators.jl")
include("kernels/ibs_kick.jl")
include("kernels/implicit.jl")
include("kernels/thin.jl")
include("kernels/fringe.jl")
include("kernels/elsep.jl")

include("fields.jl")

include("utils/find_stuff.jl")
include("modules/RungeKuttaTracking.jl")

# Empty tracking method to be imported+implemented by package extensions
function track! end

end
