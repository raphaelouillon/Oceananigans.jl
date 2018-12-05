module OceanDispatch

export
    ConstantsCollection,
    Grid,
    Field,
    FieldCollection,
    TimeStepper,

    PlanetaryConstants,
    EarthConstants,

    RegularCartesianGrid,

    ZoneField,
    FaceField,
    Fields,
    SourceTermFields,
    ForcingFields,

    laplacian3d_ppn,

    solve_poisson_1d_pbc,
    solve_poisson_1d_nbc,
    solve_poisson_2d_pbc,
    solve_poisson_2d_mbc,
    solve_poisson_3d_pbc,
    solve_poisson_3d_mbc,
    solve_poisson_3d_ppn

using
    FFTW

abstract type ConstantsCollection end
abstract type EquationOfStateParameters <: ConstantsCollection end
abstract type Grid end
abstract type Field end
# abstract type Field <: AbstractArray end
abstract type FieldCollection end
abstract type TimeStepper end

include("planetary_constants.jl")
include("grid.jl")
include("fields.jl")
include("operators.jl")
include("equation_of_state_future.jl")
include("time_steppers.jl")
include("spectral_solvers.jl")

end # module
