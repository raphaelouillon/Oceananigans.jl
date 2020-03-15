"""
    AdamsBashforthTimeStepper(float_type, arch, grid, tracers, χ=0.125;
                              Gⁿ = TendencyFields(arch, grid, tracers),
                              G⁻ = TendencyFields(arch, grid, tracers))

Return an AdamsBashforthTimeStepper object with tendency fields on `arch` and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
struct AdamsBashforthTimeStepper{T, TG, P} <: AbstractTimeStepper
     χ :: T
    Gⁿ :: TG
    G⁻ :: TG
    predictor_velocities :: P
end

function AdamsBashforthTimeStepper(float_type, arch, grid, velocities, tracers, χ=0.1;
                                   Gⁿ = TendencyFields(arch, grid, tracers),
                                   G⁻ = TendencyFields(arch, grid, tracers))

    u★ = Field{Face, Cell, Cell}(data(velocities.u), grid, velocities.u.boundary_conditions)
    v★ = Field{Cell, Face, Cell}(data(velocities.v), grid, velocities.v.boundary_conditions)
    w★ = Field{Cell, Cell, Face}(data(velocities.w), grid, velocities.w.boundary_conditions)

    U★ = (u=u★, v=v★, w=w★)

    return AdamsBashforthTimeStepper{float_type, typeof(Gⁿ), typeof(U★)}(χ, Gⁿ, G⁻, U★)
end

#####
##### Time steppping
#####

"""
    time_step!(model::IncompressibleModel{<:AdamsBashforthTimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::IncompressibleModel{<:AdamsBashforthTimeStepper}, Δt; euler=false)
    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Convert NamedTuples of Fields to NamedTuples of OffsetArrays
    velocities, tracers, pressures, diffusivities, Gⁿ, G⁻, predictor_velocities =
        datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities,
                   model.timestepper.Gⁿ, model.timestepper.G⁻, model.timestepper.predictor_velocities)

    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

    calculate_tendencies!(Gⁿ, velocities, tracers, pressures, diffusivities, model)

    ab2_time_step_tracers!(tracers, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    # Fractional step. Note that predictor velocities share memory space with velocities.
    ab2_update_predictor_velocities!(predictor_velocities, Δt, χ, Gⁿ, G⁻)

    calculate_pressure_correction!(pressures.pNHS, Δt, predictor_velocities, model)

    fractional_step_velocities!(velocities, tracers, model.architecture,
                                model.grid, Δt, pressures.pNHS)

    ab2_store_source_terms!(G⁻, model.architecture, model.grid, χ, Gⁿ)

    # Compute w from recontinuity
    fill_halo_regions!(model.velocities, model.architecture,
                       boundary_condition_function_arguments(model)...)

    compute_w_from_continuity!(model)

    tick!(model.clock, Δt)

    return nothing
end

#####
##### Source term storage
#####

""" Store previous source terms before updating them. """
function ab2_store_source_terms!(G⁻, arch, grid, χ, Gⁿ)

    # Velocity fields
    @launch(device(arch), config=launch_config(grid, :xyz),
            ab2_store_velocity_source_terms!(G⁻, grid, χ, Gⁿ))

    # Tracer fields
    for i in 4:length(G⁻)
        @inbounds Gc⁻ = G⁻[i]
        @inbounds Gcⁿ = Gⁿ[i]
        @launch(device(arch), config=launch_config(grid, :xyz),
                ab2_store_tracer_source_term!(Gc⁻, grid, χ, Gcⁿ))
    end

    return nothing
end

""" Store source terms for `u`, `v`, and `w`. """
function ab2_store_velocity_source_terms!(G⁻, grid::AbstractGrid{FT}, χ, Gⁿ) where FT
    @loop_xyz i j k grid begin
        @inbounds G⁻.u[i, j, k] = Gⁿ.u[i, j, k]
        @inbounds G⁻.v[i, j, k] = Gⁿ.v[i, j, k]
        @inbounds G⁻.w[i, j, k] = Gⁿ.w[i, j, k]
    end
    return nothing
end

""" Store previous source terms for a tracer before updating them. """
function ab2_store_tracer_source_term!(Gc⁻, grid::AbstractGrid{FT}, χ, Gcⁿ) where FT
    @loop_xyz i j k grid begin
        @inbounds Gc⁻[i, j, k] = Gcⁿ[i, j, k]
    end
    return nothing
end

function ab2_time_step_tracers!(C, arch, grid, Δt, χ, Gⁿ, G⁻)
    for i in 1:length(C)
        @inbounds c = C[i]
        @inbounds Gcⁿ = Gⁿ[i+3]
        @inbounds Gc⁻ = G⁻[i+3]
        @launch device(arch) config=launch_config(grid, :xyz) ab2_time_step_tracer!(c, grid, Δt, χ, Gcⁿ, Gc⁻)
    end

    return nothing
end

#####
##### Tracer time stepping and predictor velocity updating
#####

"""
Time step tracers via

    `c^{n+1} = c^n + Δt ( (3/2 + χ) * Gc^{n} - (1/2 + χ) G^{n-1} )`

"""
function ab2_time_step_tracer!(c, grid::AbstractGrid{FT}, Δt, χ, Gcⁿ, Gc⁻) where FT
    @loop_xyz i j k grid begin
        @inbounds c[i, j, k] += Δt * ((FT(1.5) + χ) * Gcⁿ[i, j, k] - (FT(0.5) + χ) * Gc⁻[i, j, k])
    end
    return nothing
end

"""
    ab2_update_predictor_velocities!(U★, Δt, χ, Gⁿ, G⁻)

Update the predictor velocty field using the tendencies at both time step 
`n` (the current time step) and the previous time step `n-1`.
This function updates the predictor velocity in the interior and on the boundaries,
which is required for correct evaluation of the predictor velocity divergence 
across the boundary during the calculation of the pressure correction.
"""
function ab2_update_predictor_velocities!(U★, Δt, χ, Gⁿ, G⁻)
    FT = eltype(U★.u)
    @. U★.u += Δt * (FT(1.5) + χ) * Gⁿ.u - (FT(0.5) + χ) * G⁻.u
    @. U★.v += Δt * (FT(1.5) + χ) * Gⁿ.v - (FT(0.5) + χ) * G⁻.v
    @. U★.w += Δt * (FT(1.5) + χ) * Gⁿ.w - (FT(0.5) + χ) * G⁻.w
    return nothing
end
