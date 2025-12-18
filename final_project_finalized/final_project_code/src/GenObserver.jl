module GenObserver

using Gen
using ..GridWorld
using ..Agent

export observer_model, infer_posterior_mh

@gen function observer_model(Qdict, goals::Vector{Symbol}, betas::Vector{Float64},
                             states::Vector{State}, T::Int)

    g_idx ~ categorical(fill(1/length(goals), length(goals)))
    b_idx ~ categorical(fill(1/length(betas), length(betas)))
    goal = goals[g_idx]
    β = betas[b_idx]

    Qg = Qdict[goal]

    for t in 1:T
        s = states[t]
        probs = softmax_probs(Qg, s, β)
        a_idx ~ categorical(probs)  #address (:a, t)
    end
end

"""
run MH over (g_idx, b_idx) with fixed observed actions
return est. posterior counts over (goal, beta)
"""
function infer_posterior_mh(Qdict, goals, betas, states, actions_obs;
                            n_iters::Int=5000)

    T = length(actions_obs)
    #build observation choicemap for actions
    obs = choicemap()
    for t in 1:T
        a = actions_obs[t]
        idx = findfirst(==(a), GridWorld.actions)
        obs[(:a, t)] = idx
    end

    #init trace
    (trace, _) = generate(observer_model, (Qdict, goals, betas, states, T), obs)

    #simple proposals, rand flip goal or beta index
    function propose_goal(trace)
        new_idx = rand(1:length(goals))
        return choicemap((:g_idx,)=>new_idx)
    end
    function propose_beta(trace)
        new_idx = rand(1:length(betas))
        return choicemap((:b_idx,)=>new_idx)
    end

    counts = Dict{Tuple{Symbol,Float64},Int}()

    for i in 1:n_iters
        if rand() < 0.5
            (trace, _) = mh(trace, propose_goal, ())
        else
            (trace, _) = mh(trace, propose_beta, ())
        end
        g = goals[trace[:g_idx]]
        β = betas[trace[:b_idx]]
        counts[(g,β)] = get(counts, (g,β), 0) + 1
    end

    return counts
end

end 