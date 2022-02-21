using Printf
using GLMakie
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, mask_immersed_field!
using Oceananigans: fields
using Oceananigans.Architectures: device
using KernelAbstractions: MultiEvent

Nz = 128  # Resolution
Bu = 0.1  # Slope Burger number
f = 1     # Coriolis parameter
κ = 0.1   # Diffusivity and viscosity (Prandtl = 1)

sin²θ = 0.5 # Slope of 45 deg

# Buoyancy frequency
@show N² = Bu * f^2 / sin²θ

# Time-scale for diffusion over 1 unit
τ = 1 / κ

underlying_grid = RegularRectilinearGrid(size = (2Nz, Nz),
                                         x = (-6, 6),
                                         z = (-1, 5),
                                         topology = (Bounded, Flat, Bounded))

@inline slope(x, y) = min(x, 1)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(slope))

#=
μ = 0.1 # restoring rate
@inline b_forcing_func(x, y, z, b, p) = p.μ * (p.N² * z - b)
b_forcing = Forcing(b_forcing_func, field_dependencies=:b, parameters=(; N², μ))

model = NonhydrostaticModel(architecture = GPU(),
                            grid = grid,
                            advection = CenteredSecondOrder(),
                            closure = IsotropicDiffusivity(ν=κ, κ=κ),
                            tracers = :b,
                            coriolis = FPlane(f=1),
                            # forcing = (; b=b_forcing),
                            buoyancy = BuoyancyTracer())

# Linear stratification
set!(model, b = (x, y, z) -> N² * z)

start_time = [time_ns()]

progress(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                             100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                             s.model.clock.time, maximum(abs, model.velocities.w))

Δt = 0.1 * grid.Δx^2 / model.closure.κ.b

simulation = Simulation(model, Δt = Δt, stop_time = 100/f, progress = progress, iteration_interval = 100)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(0.1),
                                                      prefix = "diffusion_driven_flow",
                                                      field_slicer = nothing,
                                                      force = true)
                        
run!(simulation)

@info """
    Simulation complete.
    Runtime: $(prettytime(simulation.run_time))
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""
=#

#####
##### Visualization
#####

filepath = "diffusion_driven_flow.jld2"

ut = FieldTimeSeries(filepath, "u", grid=grid)
vt = FieldTimeSeries(filepath, "v", grid=grid)
wt = FieldTimeSeries(filepath, "w", grid=grid)
bt = FieldTimeSeries(filepath, "b", grid=grid)

times = ut.times
Nt = length(times)

ut = [ut[n] for n = 1:Nt]
vt = [vt[n] for n = 1:Nt]
wt = [wt[n] for n = 1:Nt]
bt = [bt[n] for n = 1:Nt]

# Preprocess
events = []
for n = 1:Nt
    for f in (ut[n], vt[n], wt[n], bt[n])
        push!(events, mask_immersed_field!(f, NaN))
    end
end

wait(device(CPU()), MultiEvent(Tuple(events)))

# Plot

n = Node(1)

ui(n) = interior(ut[n])[:, 1, :]
vi(n) = interior(vt[n])[:, 1, :]
wi(n) = interior(wt[n])[:, 1, :]
bi(n) = interior(bt[n])[:, 1, :]

fluid_u(n) = filter(isfinite, ui(n)[:])
fluid_v(n) = filter(isfinite, vi(n)[:])
fluid_w(n) = filter(isfinite, wi(n)[:])
fluid_b(n) = filter(isfinite, bi(n)[:])

up = @lift ui($n)
vp = @lift vi($n)
wp = @lift wi($n)
bp = @lift bi($n)

@show max_u = + maximum(abs, fluid_u(Nt))
@show max_v = + maximum(abs, fluid_v(Nt))
@show max_w = + maximum(abs, fluid_w(Nt))

min_u = - max_u
min_v = - max_v
min_w = - max_w

max_b = @lift maximum(fluid_b($n))
min_b = @lift minimum(fluid_b($n))

fig = Figure(resolution=(1800, 1200))

ax_b = Axis(fig[1, 1], title="Buoyancy")
hm_b = heatmap!(ax_b, bp, colorrange=(min_b, max_b), colormap=:thermal)
cb_b = Colorbar(fig[1, 2], hm_b)

ax_u = Axis(fig[1, 3], title="x-velocity")
hm_u = heatmap!(ax_u, up, colorrange=(min_u, max_u), colormap=:balance)
cb_u = Colorbar(fig[1, 4], hm_u)

ax_v = Axis(fig[2, 1], title="y-velocity")
hm_v = heatmap!(ax_v, vp, colorrange=(min_v, max_v), colormap=:balance)
cb_v = Colorbar(fig[2, 2], hm_v)

ax_w = Axis(fig[2, 3], title="z-velocity")
hm_w = heatmap!(ax_w, wp, colorrange=(min_w, max_w), colormap=:balance)
cb_w = Colorbar(fig[2, 4], hm_w)

title_str = @lift "Diffusion driven flow at t = $(times[$n])"
ax_t = fig[0, :] = Label(fig, title_str)

record(fig, "diffusion_driven_flow.mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end

display(fig)

