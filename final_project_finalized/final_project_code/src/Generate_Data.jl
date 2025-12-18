using Pkg, DataFrames
Pkg.activate("..")  #proj root

include("../src/GridWorld.jl")
include("../src/Agent.jl")
include("../src/DataGen.jl")

using .GridWorld
using .DataGen

#def world
w = World(
    5, 5,
    walls = [State(3,3)],  #one wall in the middle
    goals = Dict(
        :A => State(5,5),
        :B => State(1,5)
    )
)

df = generate_dataset(w;
                      betas=[0.5, 1.0, 3.0, 8.0],
                      episodes_per_combo=50,
                      max_steps=30,
                      outpath="data/episodes.csv")

println("Generated ", nrow(df), " episodes.")