module Agent

using ..GridWorld
using StatsBase

export compute_Q_for_goal, softmax_probs, softmax_sample_action, simulate_episode

"""
get simple Q(s,a) = - distance(next_state, goal)
for each state, action (fast + sufficient)
"""
function compute_Q_for_goal(w::World, goal_name::Symbol)
    goal = w.goals[goal_name]
    q_goal = Dict{Tuple{Int,Int}, Dict{Action,Float64}}()
    for x in 1:w.width, y in 1:w.height
        s = State(x,y)
        if GridWorld.is_wall(w, s)
            continue
        end
        stuple = (x,y)
        q_goal[stuple] = Dict{Action,Float64}()
        for a in GridWorld.actions
            s2 = step!(w, s, a)
            q_goal[stuple][a] = -manhattan_distance(s2, goal)
        end
    end
    return q_goal
end

"""
return action probs π(a|s, goal, β) over GridWorld.actions
as Vector{Float64} 
"""
function softmax_probs(Qg::Dict{Tuple{Int,Int}, Dict{Action,Float64}},
                       s::State,
                       β::Float64)
    stuple = (s.x, s.y)
    qa = Qg[stuple]
    vals = [qa[a] for a in GridWorld.actions]
    exps = exp.(β .* vals)
    probs = exps ./ sum(exps)
    return probs
end

"""
sample action from softmax_probs, returns (action, probs)
"""
function softmax_sample_action(Qg::Dict{Tuple{Int,Int}, Dict{Action,Float64}},
                               s::State,
                               β::Float64)
    probs = softmax_probs(Qg, s, β)
    idx = sample(1:length(GridWorld.actions), Weights(probs))
    return GridWorld.actions[idx], probs
end

"""
Simulate one episode given goal and β
returns (states, actions)
states length = actions length + 1
"""
function simulate_episode(w::World,
                          Qg::Dict{Tuple{Int,Int}, Dict{Action,Float64}},
                          goal::State;
                          β::Float64=1.0,
                          max_steps::Int=50,
                          start_state::State=State(1,1))
    s = start_state
    states = State[]
    acts = Action[]
    push!(states, s)

    for t in 1:max_steps
        if is_terminal(w, s, goal)
            break
        end
        a, _ = softmax_sample_action(Qg, s, β)
        push!(acts, a)
        s = step!(w, s, a)
        push!(states, s)
        if is_terminal(w, s, goal)
            break
        end
    end

    return states, acts
end

end # module