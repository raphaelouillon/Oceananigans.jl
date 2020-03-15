"""
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

Perform precomputations necessary for an explicit timestep or substep.
"""
function time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

    fill_halo_regions!(merge(model.velocities, model.tracers), model.architecture,
                       boundary_condition_function_arguments(model)...)

    calculate_diffusivities!(diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, velocities, tracers)

    fill_halo_regions!(model.diffusivities, model.architecture)

    @launch(device(model.architecture), config=launch_config(model.grid, :xy),
            update_hydrostatic_pressure!(pressures.pHY′, model.grid, model.buoyancy, tracers))

    fill_halo_regions!(model.pressures.pHY′, model.architecture)

    return nothing
end
