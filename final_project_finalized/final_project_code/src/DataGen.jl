module DataGen

using ..GridWorld
using ..Agent
using DataFrames
using CSV

export generate_dataset, parse_states, parse_actions

function parse_states(s::AbstractString)
    parts = split(s, ';')
    out = GridWorld.State[]
    for p in parts
        m = match(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", p)
        m === nothing && error("Bad state token: $p")
        push!(out, GridWorld.State(parse(Int, m.captures[1]),
                                  parse(Int, m.captures[2])))
    end
    return out
end

const ACTION_MAP = Dict(
    "UP" => GridWorld.UP,
    "DOWN" => GridWorld.DOWN,
    "LEFT" => GridWorld.LEFT,
    "RIGHT" => GridWorld.RIGHT
)

function parse_actions(s::AbstractString)
    out = GridWorld.Action[]
    for tok in split(s, ';')
        tok = strip(tok)
        haskey(ACTION_MAP, tok) || error("Bad action token: $tok")
        push!(out, ACTION_MAP[tok])
    end
    return out
end

"""
Generate dataset of episodes for all (goal, β) combos
"""
function generate_dataset(w::World;
                          betas::Vector{Float64} = [0.5, 1.0, 3.0, 8.0],
                          episodes_per_combo::Int = 50,
                          max_steps::Int = 50,
                          outpath::String = "data/episodes.csv")

    #precomp Q / goal
    Qdict = Dict{Symbol, Dict{Tuple{Int,Int}, Dict{GridWorld.Action,Float64}}}()
    for gname in keys(w.goals)
        Qdict[gname] = compute_Q_for_goal(w, gname)
    end

    rows = Vector{Dict{Symbol,Any}}()

    episode_id = 1
    for (gname, goal_state) in w.goals
        Qg = Qdict[gname]
        for β in betas
            for _ in 1:episodes_per_combo
                states, acts = simulate_episode(w, Qg, goal_state; β=β, max_steps=max_steps)

                #encode states, actions as strings 
                state_str = join(["($(s.x),$(s.y))" for s in states], ";")
                act_str   = join([string(a) for a in acts], ";")

                push!(rows, Dict(
                    :episode_id => episode_id,
                    :goal_name  => String(gname),
                    :beta       => β,
                    :states     => state_str,
                    :actions    => act_str,
                    :num_steps  => length(acts)
                ))
                episode_id += 1
            end
        end
    end

    df = DataFrame(rows)
    mkpath(dirname(outpath))
    CSV.write(outpath, df)
    return df
end

end 