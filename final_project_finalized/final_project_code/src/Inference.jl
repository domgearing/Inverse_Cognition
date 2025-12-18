module Inference

import ..GridWorld
using ..GridWorld: World, State, step!, actions, Action, manhattan_distance
using ..Agent: compute_Q_for_goal, softmax_probs

using Statistics: mean
using DataFrames
using CSV
using StatsPlots

export parse_states, parse_actions,
       loglik_episode, posterior_over_goal_and_beta,
       map_goal_beta, evaluate_all,
       confusion_matrix, plot_confusion

#numeric helper
function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

#parsing helpers
function parse_states(str::AbstractString)
    parts = split(str, ";"; keepempty=false)
    out = State[]
    for p in parts
        s = strip(p)
        s = replace(s, "(" => "", ")" => "")
        xy = split(s, ",")
        x = parse(Int, strip(xy[1]))
        y = parse(Int, strip(xy[2]))
        push!(out, State(x, y))
    end
    return out
end

const ACTION_MAP = Dict(
    "UP"    => GridWorld.UP,
    "DOWN"  => GridWorld.DOWN,
    "LEFT"  => GridWorld.LEFT,
    "RIGHT" => GridWorld.RIGHT
)

function parse_actions(str::AbstractString)
    parts = split(str, ";"; keepempty=false)
    out = Action[]
    for p in parts
        tok = strip(p)
        haskey(ACTION_MAP, tok) || error("Bad action token: $tok")
        push!(out, ACTION_MAP[tok])
    end
    return out
end

#likelihood + posteriors 

"""
log-lik of observed actions given (goal, β)
softmax_probs logic from Agent.jl (so policies match)
assumes states length = actions+1
ϵ - epsilon mixture with uniform over actions
λ - goal-progress shaping term
"""
function loglik_episode(
    Qdict::Dict{Symbol, Dict{Tuple{Int,Int}, Dict{Action,Float64}}},
    states::Vector{State},
    actions_obs::Vector{Action},
    goal_name::Symbol,
    β::Float64;
    ϵ::Float64 = 0.02,
    λ::Float64 = 0.0,
    w::Union{Nothing,World} = nothing
)
    T = length(actions_obs)
    length(states) == T + 1 || error("states must have length T+1")

    Qg = Qdict[goal_name]
    logp = 0.0

    for t in 1:T
        s = states[t]
        a_obs = actions_obs[t]

        if w !== nothing
            s_next = step!(w, s, a_obs)
            if !(s_next.x == states[t+1].x && s_next.y == states[t+1].y)
                return -Inf
            end
            if λ != 0.0
                d_prev = manhattan_distance(s, w.goals[goal_name])
                d_next = manhattan_distance(s_next, w.goals[goal_name])
                logp += λ * (d_prev - d_next)
            end
        end

        probs = softmax_probs(Qg, s, β)           #aligned with GridWorld.actions ordering
        probs = (1-ϵ) .* probs .+ ϵ .* fill(1/length(actions), length(actions))

        idx = findfirst(==(a_obs), actions)
        idx === nothing && error("Observed action $a_obs not in actions=$actions")

        logp += log(probs[idx] + 1e-12)           #tiny epsilon - safety
    end

    return logp
end

"""
posterior over discrete grid of goals x betas for single episode
return Dict((goal, β) => posterior_prob)
supports priors + (optional) w, λ and ϵ
"""
function posterior_over_goal_and_beta(
    w::World,
    Qdict::Dict{Symbol, Dict{Tuple{Int,Int}, Dict{Action,Float64}}},
    states::Vector{State},
    actions_obs::Vector{Action};
    goals::Vector{Symbol},
    betas::Vector{Float64},
    prior_goal::Dict{Symbol,Float64}=Dict(g=>1/length(goals) for g in goals),
    prior_beta::Dict{Float64,Float64}=Dict(b=>1/length(betas) for b in betas),
    ϵ::Float64 = 0.02,
    λ::Float64 = 0.0
)
    lps = Dict{Tuple{Symbol,Float64},Float64}()

    for g in goals, β in betas
        ll = loglik_episode(Qdict, states, actions_obs, g, β; ϵ=ϵ, λ=λ, w=w)
        lp = ll + log(prior_goal[g]) + log(prior_beta[β])
        lps[(g,β)] = lp
    end

    normconst = logsumexp(collect(values(lps)))
    post = Dict{Tuple{Symbol,Float64},Float64}()
    for (k, lp) in lps
        post[k] = exp(lp - normconst)
    end
    return post
end

"""
MAP est. (goal_hat, beta_hat) from posterior dict
"""
function map_goal_beta(post::Dict{Tuple{Symbol,Float64},Float64})
    best = first(keys(post))
    bestp = -Inf
    for (k, p) in post
        if p > bestp
            bestp = p
            best = k
        end
    end
    return best[1], best[2]
end

#eval + plot

"""
get confusion matrix counts
rows=true labels, cols=pred labels
return (labels, matrix)
"""
function confusion_matrix(trues::Vector, preds::Vector)
    labels = sort(collect(unique(vcat(trues, preds))))
    idx = Dict(l => i for (i,l) in enumerate(labels))
    M = zeros(Int, length(labels), length(labels))
    for (t,p) in zip(trues, preds)
        M[idx[t], idx[p]] += 1
    end
    return labels, M
end

"""
plot confusion matrix as heatmap (StatsPlots)
"""
function plot_confusion(labels, M; title::String="Confusion Matrix", clims=nothing)
    heatmap(string.(labels), string.(labels), M;
            xlabel="Predicted", ylabel="True",
            title=title, aspect_ratio=1,
            cbar=true, annot=M, clims=clims)
end

"""
Eval over episodes.csv
build Qdict from world
parse episodes
compute posterior + MAP per episode
return result DataFrame and confusion matrices
saves plots to save_plots_dir (png + pdf + svg)
"""
function evaluate_all(
    csv_path::String,
    w::World;
    goals::Vector{Symbol}=collect(keys(w.goals)),
    betas::Vector{Float64}=[0.5,1.0,3.0,8.0],
    ϵ::Float64=0.02,
    λ::Float64=0.0,
    save_plots_dir::Union{Nothing,String}="figures"
)
    # Precompute Q
    Qdict = Dict{Symbol, Dict{Tuple{Int,Int}, Dict{Action,Float64}}}()
    for g in goals
        Qdict[g] = compute_Q_for_goal(w, g)
    end

    df = CSV.read(csv_path, DataFrame)

    true_goals = Symbol[]
    true_betas = Float64[]
    pred_goals = Symbol[]
    pred_betas = Float64[]

    for r in eachrow(df)
        states = parse_states(String(r.states))
        acts   = parse_actions(String(r.actions))

        gtrue = Symbol(r.goal_name)
        βtrue = Float64(r.beta)

        post = posterior_over_goal_and_beta(w, Qdict, states, acts; goals=goals, betas=betas, ϵ=ϵ, λ=λ)
        ghat, βhat = map_goal_beta(post)

        push!(true_goals, gtrue); push!(true_betas, βtrue)
        push!(pred_goals, ghat);  push!(pred_betas, βhat)
    end

    goal_acc = mean(true_goals .== pred_goals)
    beta_acc = mean(true_betas .== pred_betas)

    println("Goal accuracy: ", round(goal_acc, digits=3))
    println("Beta accuracy: ", round(beta_acc, digits=3))

    results = DataFrame(
        episode_id = df.episode_id,
        true_goal  = true_goals,
        pred_goal  = pred_goals,
        true_beta  = true_betas,
        pred_beta  = pred_betas
    )

    g_labels, gM = confusion_matrix(true_goals, pred_goals)
    b_labels, bM = confusion_matrix(true_betas, pred_betas)

    if save_plots_dir !== nothing
        mkpath(save_plots_dir)

        pg = plot_confusion(g_labels, gM; title="Goal Confusion (ϵ=$(ϵ), λ=$(λ))")
        savefig(pg, joinpath(save_plots_dir, "confusion_goal_eps=$(ϵ)_lam=$(λ).png"))
        savefig(pg, joinpath(save_plots_dir, "confusion_goal_eps=$(ϵ)_lam=$(λ).pdf"))
        savefig(pg, joinpath(save_plots_dir, "confusion_goal_eps=$(ϵ)_lam=$(λ).svg"))

        pb = plot_confusion(b_labels, bM; title="Beta Confusion (ϵ=$(ϵ), λ=$(λ))")
        savefig(pb, joinpath(save_plots_dir, "confusion_beta_eps=$(ϵ)_lam=$(λ).png"))
        savefig(pb, joinpath(save_plots_dir, "confusion_beta_eps=$(ϵ)_lam=$(λ).pdf"))
        savefig(pb, joinpath(save_plots_dir, "confusion_beta_eps=$(ϵ)_lam=$(λ).svg"))
    end

    return results, (g_labels, gM), (b_labels, bM), Qdict
end

end 