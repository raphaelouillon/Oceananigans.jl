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
    velocities, tracers, pressures, diffusivities, Gⁿ, G⁻ =
        datatuples(model.velocities, model.tracers, model.pressures, model.diffusivities,
                   model.timestepper.Gⁿ, model.timestepper.G⁻)

    calculate_explicit_substep!(Gⁿ, velocities, tracers, pressures, diffusivities, model)

    ab2_time_step_tracers!(tracers, model.architecture, model.grid, Δt, χ, Gⁿ, G⁻)

    ab2_update_source_terms!(Gⁿ, model.architecture, model.grid, χ, G⁻)

    calculate_pressure_correction!(pressures.pNHS, Δt, Gⁿ, velocities, model)

    #complete_pressure_correction_step!(velocities, Δt, tracers, pressures, Gⁿ, model)

    update_solution!(velocities, tracers, model.architecture,
                     model.grid, Δt, Gⁿ, pressures.pNHS)

    fill_halo_regions!(model.velocities, model.architecture,
                       boundary_condition_function_arguments(model)...)

    compute_w_from_continuity!(model)

    ab2_store_source_terms!(G⁻, model.architecture, model.grid, χ, Gⁿ)

    tick!(model.clock, Δt)

    return nothing
end

#####
##### Adams-Bashforth-specific kernels
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
        @inbounds G⁻.u[i, j, k] = ((FT(0.5) + χ) * G⁻.u[i, j, k] + Gⁿ.u[i, j, k]) / (FT(1.5) + χ)
        @inbounds G⁻.v[i, j, k] = ((FT(0.5) + χ) * G⁻.v[i, j, k] + Gⁿ.v[i, j, k]) / (FT(1.5) + χ)
        @inbounds G⁻.w[i, j, k] = ((FT(0.5) + χ) * G⁻.w[i, j, k] + Gⁿ.w[i, j, k]) / (FT(1.5) + χ)
    end
    return nothing
end

""" Store previous source terms for a tracer before updating them. """
function ab2_store_tracer_source_term!(Gc⁻, grid::AbstractGrid{FT}, χ, Gcⁿ) where FT
    @loop_xyz i j k grid begin
        @inbounds Gc⁻[i, j, k] = ((FT(0.5) + χ) * Gc⁻[i, j, k] + Gcⁿ[i, j, k]) / (FT(1.5) + χ)
    end
    return nothing
end

"""
Evaluate the right-hand-side terms for velocity fields and tracer fields
at time step n+½ using a weighted 2nd-order Adams-Bashforth method.
"""
function ab2_update_source_terms!(Gⁿ, arch, grid, χ, G⁻)
    # Velocity fields
    @launch(device(arch), config=launch_config(grid, :xyz),
            ab2_update_velocity_source_terms!(Gⁿ, grid, χ, G⁻))

    # Tracer fields
    for i in 4:length(Gⁿ)
        @inbounds Gcⁿ = Gⁿ[i]
        @inbounds Gc⁻ = G⁻[i]
        @launch(device(arch), config=launch_config(grid, :xyz),
                ab2_update_tracer_source_term!(Gcⁿ, grid, χ, Gc⁻))
    end

    return nothing
end

"""
Evaluate the right-hand-side terms at time step n+½ using a weighted 2nd-order
Adams-Bashforth method

    `G^{n+½} = (3/2 + χ)G^{n} - (1/2 + χ)G^{n-1}`
"""
function ab2_update_velocity_source_terms!(Gⁿ, grid::AbstractGrid{FT}, χ, G⁻) where FT
    @loop_xyz i j k grid begin
        @inbounds Gⁿ.u[i, j, k] = (FT(1.5) + χ) * Gⁿ.u[i, j, k] - (FT(0.5) + χ) * G⁻.u[i, j, k]
        @inbounds Gⁿ.v[i, j, k] = (FT(1.5) + χ) * Gⁿ.v[i, j, k] - (FT(0.5) + χ) * G⁻.v[i, j, k]
        @inbounds Gⁿ.w[i, j, k] = (FT(1.5) + χ) * Gⁿ.w[i, j, k] - (FT(0.5) + χ) * G⁻.w[i, j, k]
    end

    return nothing
end

"""
Evaluate the right-hand-side terms at time step n+½ using a weighted 2nd-order
Adams-Bashforth method

    `G^{n+½} = (3/2 + χ)G^{n} - (1/2 + χ)G^{n-1}`
"""
function ab2_update_tracer_source_term!(Gcⁿ, grid::AbstractGrid{FT}, χ, Gc⁻) where FT
    @loop_xyz i j k grid begin
        @inbounds Gcⁿ[i, j, k] = (FT(1.5) + χ) * Gcⁿ[i, j, k] - (FT(0.5) + χ) * Gc⁻[i, j, k]
    end
    return nothing
end
