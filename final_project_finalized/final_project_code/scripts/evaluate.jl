using Pkg

#activate proj root 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

#include source files 
include(joinpath(@__DIR__, "..", "src", "GridWorld.jl"))
include(joinpath(@__DIR__, "..", "src", "Agent.jl"))
include(joinpath(@__DIR__, "..", "src", "Inference.jl"))

using .GridWorld
using .Inference
using CSV

#build world
w = World(
    5, 5,
    [State(3,3)],
    Dict(:A => State(5,5), :B => State(1,5))
)

results, goal_conf, beta_conf = Inference.evaluate_all(
    joinpath(@__DIR__, "..", "data", "episodes.csv"),
    w;
    goals=[:A,:B],
    betas=[0.5, 1.0, 3.0, 8.0],
    save_plots_dir=joinpath(@__DIR__, "..", "plots")
)

CSV.write(joinpath(@__DIR__, "..", "data", "eval_results.csv"), results)

println("Wrote data/eval_results.csv and plots/confusion_*.png")