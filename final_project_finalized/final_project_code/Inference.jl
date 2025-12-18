module Inference

using Gen: @gen, categorical
using ..GridWorld: World, State, step!, actions, Action

export parse_states, parse_actions, loglik_episode, posterior_over_goal_and_beta

function parse_states(str::AbstractString)
    parts = split(str, ";"; keepempty=false)
    states = State[]
    for p in parts
        # p - (x,y)
        s = strip(p, ['(',')',' '])
        xy = split(s, ",")
        x = parse(Int, xy[1])
        y = parse(Int, xy[2])
        push!(states, State(x,y))
    end
    return states
end

function parse_actions(str::AbstractString)
    parts = split(str, ";"; keepempty=false)
    acts = Action[]
    for p in parts
        push!(acts, Action(Symbol(p)))  #action(:UP)
    end
    return acts
end

"""
compute log-lik of (states, actions) under (goal_name, β)
"""
function loglik_episode(w::World,
                        Qdict::Dict{Symbol, Dict{Tuple{Int,Int}, Dict{Action,Float64}}},
                        states::Vector{State},
                        actions::Vector{Action},
                        goal_name::Symbol,
                        β::Float64)

    Qg = Qdict[goal_name]
    logp = 0.0
    for (s, a) in zip(states[1:end-1], actions)  #last state has no outgoing action
        stuple = (s.x,s.y)
        qa = Qg[stuple]
        vals = [qa[a2] for a2 in actions(GRIDWORLD.actions)]  #already have 'actions'
        #reconstruct probs same as softmax_policy, only need P(a)
        #same logic as softmax_policy, but not resample
        vals_all = [qa[a2] for a2 in GridWorld.actions]
        exps = exp.(β .* vals_all)
        probs = exps ./ sum(exps)
        #find index of selected action
        idx = findfirst(==(a), GridWorld.actions)
        logp += log(probs[idx])
    end
    return logp
end

"""
compute posterior P(G,β | episode) on discrete grid of goals x betas
return Dict{Tuple{Symbol,Float64},Float64}
"""
function posterior_over_goal_and_beta(w::World,
                                      Qdict::Dict{Symbol, Dict{Tuple{Int,Int}, Dict{Action,Float64}}},
                                      states::Vector{State},
                                      actions::Vector{Action};
                                      goals::Vector{Symbol}=collect(keys(w.goals)),
                                      betas::Vector{Float64}=[0.5,1.0,3.0,8.0],
                                      prior_goal::Dict{Symbol,Float64}=Dict(g=>1/length(goals) for g in goals),
                                      prior_beta::Dict{Float64,Float64}=Dict(β=>1/length(betas) for β in betas))

    lps = Dict{Tuple{Symbol,Float64},Float64}()

    for g in goals
        for β in betas
            lp_lik = loglik_episode(w, Qdict, states, actions, g, β)
            lp_prior = log(prior_goal[g]) + log(prior_beta[β])
            lps[(g,β)] = lp_lik + lp_prior
        end
    end

    #normalize in log-space
    logvals = collect(values(lps))
    normconst = logsumexp(logvals)
    post = Dict{Tuple{Symbol,Float64},Float64}()
    for (k, lp) in lps
        post[k] = exp(lp - normconst)
    end
    return post
end

end