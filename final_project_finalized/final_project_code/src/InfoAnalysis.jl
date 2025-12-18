module InfoAnalysis

using ..Inference
using ..GridWorld
using ..Agent

using CSV, DataFrames
using Statistics: mean
using LinearAlgebra: logsumexp

export entropy, beta_marginal_from_posterior,
       analyze_entropy_and_mi

#helper

#entropy in bits
function entropy(p::AbstractVector{<:Real}; eps=1e-12)
    s = 0.0
    for pi in p
        if pi > eps
            s -= pi * log2(pi)
        end
    end
    return s
end

#marginalize posterior Dict((g,β)=>p) to p(β)
function beta_marginal_from_posterior(post::Dict{Tuple{Symbol,Float64},Float64},
                                      goals::Vector{Symbol},
                                      betas::Vector{Float64})
    pβ = zeros(Float64, length(betas))
    for (j,β) in enumerate(betas)
        pβ[j] = sum(post[(g,β)] for g in goals)
    end
    pβ ./= sum(pβ)
    return pβ
end

#empirical prior over β from dataset labels
function empirical_beta_prior(df::DataFrame, betas::Vector{Float64})
    counts = zeros(Float64, length(betas))
    for r in eachrow(df)
        β = Float64(r.beta)
        j = findfirst(==(β), betas)
        counts[j] += 1
    end
    return counts ./ sum(counts)
end

"""
analysis
load episodes.csv
compute posterior per episode using Bayesian observer
compute H(β|trajectory) and plot vs T
estimate I(β; a_{1:T}) ≈ H(β) - E[H(β|a_{1:T})]
return DataFrame with per-episode stats + MI estimate
"""
function analyze_entropy_and_mi(csv_path::String, w::World;
                                goals::Vector{Symbol}=collect(keys(w.goals)),
                                betas::Vector{Float64}=[0.5,1.0,3.0,8.0],
                                use_empirical_prior::Bool=true)

    df = CSV.read(csv_path, DataFrame)

    #precompute Qdict for likelihood
    Qdict = Dict{Symbol, Dict{Tuple{Int,Int}, Dict{Action,Float64}}}()
    for g in goals
        Qdict[g] = Agent.compute_Q_for_goal(w, g)
    end

    #prior over beta for MI
    pβ_prior = use_empirical_prior ? empirical_beta_prior(df, betas) :
                                     fill(1/length(betas), length(betas))
    Hβ = entropy(pβ_prior)

    #per episode stats
    T_list = Int[]
    β_true = Float64[]
    Hβ_post = Float64[]
    β_map = Float64[]

    for r in eachrow(df)
        states = Inference.parse_states(String(r.states))
        acts   = Inference.parse_actions(String(r.actions))
        T = length(acts)

        post = Inference.posterior_over_goal_and_beta(w, Qdict, states, acts;
                                              goals=goals, betas=betas)

        pβ = beta_marginal_from_posterior(post, goals, betas)
        push!(T_list, T)
        push!(β_true, Float64(r.beta))
        push!(Hβ_post, entropy(pβ))
        #MAP beta from marginal
        push!(β_map, betas[argmax(pβ)])
    end

    out = DataFrame(T=T_list, beta_true=β_true, Hbeta_post=Hβ_post, beta_map=β_map)

    #MI estimate
    I = Hβ - mean(out.Hbeta_post)

    return out, (Hβ=Hβ, I=I, pβ_prior=pβ_prior, betas=betas)
end

end 