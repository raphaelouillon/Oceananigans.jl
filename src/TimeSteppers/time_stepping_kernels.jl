#####
##### Vertical integrals
#####

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbation` downwards:

    `pHY′ = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`
"""
function update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    @loop_xy i j grid begin
        @inbounds pHY′[i, j, grid.Nz] = - ℑzᵃᵃᶠ(i, j, grid.Nz+1, grid, buoyancy_perturbation, buoyancy, C) * ΔzF(i, j, grid.Nz+1, grid)
        @unroll for k in grid.Nz-1 : -1 : 1
            @inbounds pHY′[i, j, k] =
                pHY′[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k+1, grid, buoyancy_perturbation, buoyancy, C) * ΔzF(i, j, k+1, grid)
        end
    end
    return nothing
end

"""
Compute the vertical velocity w by integrating the continuity equation from the bottom upwards

    `w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz`
"""
function compute_w_from_continuity!(model)
    @launch(device(model.architecture), config=launch_config(model.grid, :xy),
            _compute_w_from_continuity!(datatuple(model.velocities), model.grid))
    return nothing
end

function _compute_w_from_continuity!(U, grid)
    @loop_xy i j grid begin
        # U.w[i, j, 1] = 0 is enforced via halo regions.
        @unroll for k in 2:grid.Nz
            @inbounds U.w[i, j, k] = U.w[i, j, k-1] - ΔzC(i, j, k, grid) * hdivᶜᶜᵃ(i, j, k-1, grid, U.u, U.v)
        end
    end
    return nothing
end
