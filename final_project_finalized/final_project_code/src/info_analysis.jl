using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "GridWorld.jl"))
include(joinpath(@__DIR__, "..", "src", "Agent.jl"))
include(joinpath(@__DIR__, "..", "src", "Inference.jl"))
include(joinpath(@__DIR__, "..", "src", "InfoAnalysis.jl"))

using .GridWorld
using .InfoAnalysis
using CSV, DataFrames
using Statistics: mean
using Plots

#world
w = GridWorld.World(
    5, 5,
    [GridWorld.State(3,3)],
    Dict(:A => GridWorld.State(5,5), :B => GridWorld.State(1,5))
)

csv_path = joinpath(@__DIR__, "..", "data", "episodes.csv")

stats_df, info = InfoAnalysis.analyze_entropy_and_mi(csv_path, w; goals=[:A,:B], betas=[0.5,1.0,3.0,8.0], use_empirical_prior=true)

println("H(beta) prior (bits) = ", round(info.Hβ, digits=3))
println("Estimated I(beta; trajectory) (bits) = ", round(info.I, digits=3))

mkpath(joinpath(@__DIR__, "..", "figures"))

# scatter posterior entropy vs episode length 
p1 = scatter(stats_df.T, stats_df.Hbeta_post;
             xlabel="Episode length T",
             ylabel="Posterior entropy H(β | traj) [bits]",
             title="Uncertainty about β vs. Trajectory length",
             markersize=3,
             alpha=0.6,
             legend=false)

savefig(p1, joinpath(@__DIR__, "..", "figures", "Hbeta_vs_T_scatter.png"))

# binned mean curve 
function binned_mean(x::Vector{Int}, y::Vector{Float64}; nbins::Int=12)
    xmin, xmax = minimum(x), maximum(x)
    edges = round.(Int, range(xmin, xmax; length=nbins+1))
    centers = Int[]
    means = Float64[]
    for i in 1:nbins
        lo, hi = edges[i], edges[i+1]
        idx = findall(t -> (t >= lo) && (t < hi), x)
        if !isempty(idx)
            push!(centers, round(Int, (lo+hi)/2))
            push!(means, mean(y[idx]))
        end
    end
    return centers, means
end

centers, meansH = binned_mean(stats_df.T, stats_df.Hbeta_post; nbins=12)

p2 = plot(centers, meansH;
          xlabel="Episode length T (binned)",
          ylabel="Mean H(β | traj) [bits]",
          title="β posterior entropy decreases then saturates",
          legend=false,
          linewidth=3)

h_asym = meansH[end]
hline!(p2, [h_asym], linestyle=:dash)

savefig(p2, joinpath(@__DIR__, "..", "figures", "Hbeta_vs_T_binned.png"))

#convert entropy to info gained about β 
#info gained / episode ≈ H(β) - H(β|traj)
info_gain = info.Hβ .- stats_df.Hbeta_post

p3 = scatter(stats_df.T, info_gain;
             xlabel="Episode length T",
             ylabel="Information gained about β [bits]",
             title="Information gain about β vs. Trajectory length",
             markersize=3,
             alpha=0.6,
             legend=false)

savefig(p3, joinpath(@__DIR__, "..", "figures", "InfoGain_vs_T_scatter.png"))

cent2, meansIG = binned_mean(stats_df.T, info_gain; nbins=12)
p4 = plot(cent2, meansIG;
          xlabel="Episode length T (binned)",
          ylabel="Mean info gain [bits]",
          title="Information gain increases then saturates",
          legend=false,
          linewidth=3)
savefig(p4, joinpath(@__DIR__, "..", "figures", "InfoGain_vs_T_binned.png"))

#save per-episode stats
CSV.write(joinpath(@__DIR__, "..", "data", "beta_entropy_stats.csv"), stats_df)

println("Wrote figures/*.png and data/beta_entropy_stats.csv")