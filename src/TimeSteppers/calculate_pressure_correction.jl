"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(nonhydrostatic_pressure, Δt, predictor_velocities, model)

    solve_for_pressure!(nonhydrostatic_pressure, model.pressure_solver,
                        model.architecture, model.grid, Δt, predictor_velocities)

    fill_halo_regions!(model.pressures.pNHS, model.architecture)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the horizontal velocities u and v via

    `u^{n+1} = u^n + (Gu^{n+½} - δₓp_{NH} / Δx) Δt`

Note that the vertical velocity is not explicitly time stepped.
"""
function _fractional_step_velocities!(U, grid, Δt, pNHS)
    @loop_xyz i j k grid begin
        @inbounds U.u[i, j, k] -= ∂xᶠᵃᵃ(i, j, k, grid, pNHS) * Δt
        @inbounds U.v[i, j, k] -= ∂yᵃᶠᵃ(i, j, k, grid, pNHS) * Δt
    end
    return nothing
end

"Update the solution variables (velocities and tracers)."
function fractional_step_velocities!(U, C, arch, grid, Δt, pNHS)
    @launch device(arch) config=launch_config(grid, :xyz) _fractional_step_velocities!(U, grid, Δt, pNHS)
    return nothing
end
