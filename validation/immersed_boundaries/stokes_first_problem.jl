using Printf
using GLMakie
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!
using Oceananigans: fields
using Oceananigans.Architectures: device
using KernelAbstractions: MultiEvent

using Oceananigans.ImmersedBoundaries: solid_node
using Oceananigans.Operators: Δzᵃᵃᶜ

Nz = 64 # Resolution
ν = 1e-2 # Viscosity
U = 1

grid = RegularRectilinearGrid(size = Nz,
                              z = (0, 1),
                              halo = 1,
                              topology = (Flat, Flat, Bounded))

flat_bottom(x, y) = 0
immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(flat_bottom))

#####
##### Two ways to specify a boundary condition: "intrinsically", and with a forcing function
#####

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))

@inline function τˣᶻ_no_slip(i, j, k, grid, clock, fields, ν)
    FT = eltype(grid)
    cell_has_solid_bottom = solid_node(Face(), Center(), Face(), i, j, k, grid)
    viscous_flux = @inbounds - 2 * ν * fields.u[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)
    viscous_flux_divergence = viscous_flux / Δzᵃᵃᶜ(i, j, k, grid)
    return ifelse(cell_has_solid_bottom, viscous_flux_divergence, zero(FT))
end

u_immersed_viscous_flux = Forcing(τˣᶻ_no_slip, discrete_form=true, parameters=ν)

kwargs = (architecture = CPU(),
          closure = IsotropicDiffusivity(ν=ν),
          advection = nothing,
          tracers = nothing,
          coriolis = nothing,
          buoyancy = nothing)

not_immersed_model = NonhydrostaticModel(grid = grid; boundary_conditions = (; u=u_bcs), kwargs...)
immersed_model = NonhydrostaticModel(grid = immersed_grid; forcing = (; u = u_immersed_viscous_flux), kwargs...)

function progress(sim)

    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, min(u): %.2f",
                    100 * sim.model.clock.time / sim.stop_time,
                    sim.model.clock.iteration,
                    sim.model.clock.time,
                    minimum(sim.model.velocities.u))

    return nothing
end
                            
for model in (immersed_model, not_immersed_model)
    # Linear stratification
    set!(model, u = U)

    Δt = 1e-1 * grid.Δz^2 / ν

    simulation = Simulation(model,
                            Δt = Δt,
                            stop_time = 0.1,
                            progress = progress,
                            iteration_interval = 100)

    if model.grid isa ImmersedBoundaryGrid
        prefix = "immersed_stokes_first_problem"
    else
        prefix = "not_immersed_stokes_first_problem"
    end

    simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u=model.velocities.u),
                                                          schedule = TimeInterval(0.01),
                                                          prefix = prefix,
                                                          field_slicer = nothing,
                                                          force = true)

    run!(simulation)

    @info """
        Simulation complete.
        Runtime: $(prettytime(simulation.run_time))
    """
end

immersed_filepath = "immersed_stokes_first_problem.jld2"
not_immersed_filepath = "not_immersed_stokes_first_problem.jld2"

z = znodes(Center, grid)

uti = FieldTimeSeries(immersed_filepath, "u", grid=grid)
utn = FieldTimeSeries(not_immersed_filepath, "u", grid=grid)

times = uti.times
Nt = length(times)
n = Node(1)
uii(n) = interior(uti[n])[1, 1, :]
uin(n) = interior(utn[n])[1, 1, :]
upi = @lift uii($n)
upn = @lift uin($n)

fig = Figure(resolution=(400, 600))

ax = Axis(fig[1, 1], xlabel="u(z)", ylabel="z")
lines!(ax, upi, z, label="immersed", linewidth=2, linestyle="--")
lines!(ax, upn, z, label="not immersed")

title_gen(n) = @sprintf("Stokes first problem at t = %.2f", times[n])
title_str = @lift title_gen($n)
ax_t = fig[0, :] = Label(fig, title_str)

record(fig, prefix * ".mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end

display(fig)
