"""
    calculate_tendencies!(diffusivities, pressures, velocities, tracers, model)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)

    # Note:
    #
    # "tendencies" is a NamedTuple of OffsetArrays corresponding to the tendency data for use
    # in GPU computations.
    #
    # "model.timestepper.Gⁿ" is a NamedTuple of Fields, whose data also corresponds to 
    # tendency data.
    
    # Arguments needed to calculate tendencies for momentum and tracers
    tendency_calculation_args = (tendencies, model.architecture, model.grid, model.coriolis, model.buoyancy,
                                 model.surface_waves, model.closure, velocities, tracers, pressures.pHY′,
                                 diffusivities, model.forcing, model.parameters, model.clock.time)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    calculate_interior_tendency_contributions!(tendency_calculation_args...)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the 
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(
        model.timestepper.Gⁿ, model.architecture, model.velocities,
        model.tracers, boundary_condition_function_arguments(model)...)

    # Calculate momentum tendencies on boundaries in `Bounded` directions.
    calculate_velocity_tendencies_on_boundaries!(tendency_calculation_args...)

    return nothing
end

#####
##### Navier-Stokes and tracer advection equations
#####

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(G, arch, grid, coriolis, buoyancy, surface_waves, closure, 
                                                    U, C, pHY′, K, F, parameters, time)

    # Manually choose thread-block layout here as it's ~20% faster.
    # See: https://github.com/climate-machine/Oceananigans.jl/pull/308
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    if Nx == 1
        Tx, Ty = 1, min(256, Ny)
        Bx, By, Bz = Tx, floor(Int, Ny/Ty), Nz
    elseif Ny == 1
        Tx, Ty = min(256, Nx), 1
        Bx, By, Bz = floor(Int, Nx/Tx), Ty, Nz
    else
        Tx, Ty = 16, 16
        Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz
    end

    @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
            calculate_Gu!(G.u, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, parameters, time))

    @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
            calculate_Gv!(G.v, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, parameters, time))

    @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
            calculate_Gw!(G.w, grid, coriolis, surface_waves, closure, U, C, K, F, parameters, time))

    for tracer_index in 1:length(C)
        @inbounds Gc = G[tracer_index+3]
        @inbounds Fc = F[tracer_index+3]
        @inbounds  c = C[tracer_index]

        @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
                calculate_Gc!(Gc, grid, c, Val(tracer_index), closure, buoyancy, U, C, K, Fc, parameters, time))
    end

    return nothing
end

"""
    calculate_velocity_tendencies_on_boundaries!(tendency_calculation_args...)

Calculate the velocity tendencies *on* east, north, and top boundaries, when the
x-, y-, or z- directions have a `Bounded` topology.
"""
function calculate_velocity_tendencies_on_boundaries!(tendency_calculation_args...)

    calculate_east_boundary_Gu!(tendency_calculation_args...)
    calculate_north_boundary_Gv!(tendency_calculation_args...)
    calculate_top_boundary_Gw!(tendency_calculation_args...)

    return nothing
end

# Fallbacks for non-bounded topologies in x-, y-, and z-directions
@inline calculate_east_boundary_Gu!(args...) = nothing
@inline calculate_north_boundary_Gv!(args...) = nothing
@inline calculate_top_boundary_Gw!(args...) = nothing

"""
    calculate_east_boundary_Gu!(G, arch, grid::AbstractGrid{FT, <:Bounded},
                                coriolis, buoyancy, surface_waves, closure,
                                U, C, pHY′, K, F, parameters, time) where FT

Calculate `Gu` on east boundaries when the x-direction has `Bounded` topology.
"""
function calculate_east_boundary_Gu!(G, arch, grid::AbstractGrid{FT, <:Bounded},
                                     coriolis, buoyancy, surface_waves, closure,
                                     U, C, pHY′, K, F, parameters, time) where FT

    @launch(device(arch), config=launch_config(grid, :yz), 
            _calculate_east_boundary_Gu!(G.u, grid, coriolis, surface_waves, 
                                         closure, U, C, K, F, pHY′, parameters, time))

    return nothing
end

"""
    calculate_north_boundary_Gu!(G, arch, grid::AbstractGrid{FT, <:Bounded},
                                 coriolis, buoyancy, surface_waves, closure,
                                 U, C, pHY′, K, F, parameters, time) where FT

Calculate `Gv` on north boundaries when the y-direction has `Bounded` topology.
"""
function calculate_north_boundary_Gv!(G, arch, grid::AbstractGrid{FT, TX, <:Bounded},
                                      coriolis, buoyancy, surface_waves, closure,
                                      U, C, pHY′, K, F, parameters, time) where {FT, TX}

    @launch(device(arch), config=launch_config(grid, :xz), 
            _calculate_north_boundary_Gv!(G.v, grid, coriolis, surface_waves, 
                                          closure, U, C, K, F, pHY′, parameters, time))

    return nothing
end

"""
    calculate_top_boundary_Gw!(G, arch, grid::AbstractGrid{FT, <:Bounded},
                               coriolis, buoyancy, surface_waves, closure,
                               U, C, pHY′, K, F, parameters, time) where FT

Calculate `Gw` on top boundaries when the z-direction has `Bounded` topology.
"""
function calculate_top_boundary_Gw!(G, arch, grid::AbstractGrid{FT, TX, TY, <:Bounded},
                                    coriolis, buoyancy, surface_waves, closure,
                                    U, C, pHY′, K, F, parameters, time) where {FT, TX, TY}

    @launch(device(arch), config=launch_config(grid, :xy), 
            _calculate_top_boundary_Gw!(G.w, grid, coriolis, surface_waves, 
                                        closure, U, C, K, F, parameters, time))

    return nothing
end

#####
##### Tendency calculators for u-velocity, aka x-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
function calculate_Gu!(Gu, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, parameters, time)
    @loop_xyz i j k grid begin
        @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, pHY′, parameters, time)
    end
    return nothing
end

""" Calculate the right-hand-side of the u-velocity equation on the east boundary. """
function _calculate_east_boundary_Gu!(Gu, grid, coriolis, surface_waves,
                                      closure, U, C, K, F, pHY′, parameters, time)
    i = grid.Nx + 1
    @loop_yz j k grid begin
        @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, pHY′, parameters, time)
    end
    return nothing
end

#####
##### Tendency calculators for v-velocity
#####

""" Calculate the right-hand-side of the v-velocity equation. """
function calculate_Gv!(Gv, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, parameters, time)
    @loop_xyz i j k grid begin
        @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, pHY′, parameters, time)
    end
    return nothing
end

""" Calculate the right-hand-side of the v-velocity equation on the north boundary. """
function _calculate_north_boundary_Gv!(Gv, grid, coriolis, surface_waves,
                                       closure, U, C, K, F, pHY′, parameters, time)
    j = grid.Ny + 1
    @loop_xz i k grid begin
        @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, pHY′, parameters, time)
    end
    return nothing
end

#####
##### Tendency calculators for w-velocity
#####

""" Calculate the right-hand-side of the w-velocity equation. """
function calculate_Gw!(Gw, grid, coriolis, surface_waves, closure, U, C, K, F, parameters, time)
    @loop_xyz i j k grid begin
        @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, parameters, time)
    end
    return nothing
end

""" Calculate the right-hand-side of the w-velocity equation. """
function _calculate_top_boundary_Gw!(Gw, grid, coriolis, surface_waves, closure, U, C, K, F, parameters, time)
    k = grid.Nz + 1
    @loop_xy i j grid begin
        @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, parameters, time)
    end
    return nothing
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
function calculate_Gc!(Gc, grid, c, tracer_index, closure, buoyancy, U, C, K, Fc, parameters, time)
    @loop_xyz i j k grid begin
        @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, c, tracer_index,
                                                closure, buoyancy, U, C, K, Fc, parameters, time)
    end
    return nothing
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, U, C, args...)

    # Velocity fields
    for i in 1:3
        apply_z_bcs!(Gⁿ[i], U[i], arch, args...)
        apply_y_bcs!(Gⁿ[i], U[i], arch, args...)
    end

    # Tracer fields
    for i in 4:length(Gⁿ)
        apply_z_bcs!(Gⁿ[i], C[i-3], arch, args...)
        apply_y_bcs!(Gⁿ[i], C[i-3], arch, args...)
    end

    return nothing
end
