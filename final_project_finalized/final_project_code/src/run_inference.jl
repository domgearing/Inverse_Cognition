using Pkg
Pkg.activate("..")

include("../src/GridWorld.jl")
include("../src/Agent.jl")
include("../src/DataGen.jl")
include("../src/Inference.jl")

import .GridWorld
using .Agent
using .Inference
using CSV, DataFrames

states  = DataGen.parse_states(row.states)
actions = DataGen.parse_actions(row.actions)

w = GridWorld.World(
    5, 5,
    walls = [GridWorld.State(3,3)],
    goals = Dict(
        :A => GridWorld.State(5,5),
        :B => GridWorld.State(1,5)
    )
)

#precompute Q tables
Qdict = Dict{Symbol, Dict{Tuple{Int,Int}, Dict{GridWorld.Action, Float64}}}()
for gname in keys(w.goals)
    Qdict[gname] = Agent.compute_Q_for_goal(w, gname)
end

df = CSV.read("data/episodes.csv", DataFrame)

row = df[1, :]
states  = DataGen.parse_states(row.states)
actions = DataGen.parse_actions(row.actions)

post = Inference.posterior_over_goal_and_beta(w, Qdict, states, actions;
                                             goals=[:A,:B],
                                             betas=[0.5,1.0,3.0,8.0])

println("True goal = ", row.goal_name, ", true β = ", row.beta)
println("Posterior over (goal,β):")
for ((g,β), p) in post
    println("  (", g, ", ", β, ") => ", round(p, digits=3))
end