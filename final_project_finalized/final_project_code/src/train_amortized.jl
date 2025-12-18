using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CSV, DataFrames, StatsBase
using Flux
using Random

include("GridWorld.jl")
include("Agent.jl")
include("Inference.jl")

using .GridWorld
using .Agent
using .Inference

using Pkg; Pkg.activate("."); Pkg.add("Flux")

df = CSV.read("data/episodes.csv", DataFrame)

#def same world used to gen. episodes
w = GridWorld.World(
    5, 5,
    walls = [GridWorld.State(3,3)],
    goals = Dict(
        :A => GridWorld.State(5,5),
        :B => GridWorld.State(1,5)
    )
)
#precompute Q tables for each goal
goals = sort(unique(Symbol.(df.goal_name)))
Qdict = Dict{Symbol, Dict{Tuple{Int,Int}, Dict{GridWorld.Action,Float64}}}()
for g in [:A, :B]
    Qdict[g] = Agent.compute_Q_for_goal(w, g)
end

#label sets
goals = sort(unique(Symbol.(df.goal_name)))
betas = sort(unique(Float64.(df.beta)))
goal_to_i = Dict(g=>i for (i,g) in enumerate(goals))
beta_to_i = Dict(b=>i for (i,b) in enumerate(betas))

#featurize action counts + length
function featurize_under_goal(states_str::String, actions_str::String, goal_sym::Symbol, Qdict)
    states = Inference.parse_states(states_str)
    toks = split(actions_str, ";"; keepempty=false)
    T = length(toks)

    Qg = Qdict[goal_sym]
    q_obs = Float32[]
    regret = Float32[]

    for t in 1:T
        s = states[t]
        a = Inference.ACTION_MAP[strip(toks[t])]
        qa = Qg[(s.x, s.y)]
        vals = Float32[qa[aa] for aa in GridWorld.actions]
        q_a = Float32(qa[a])
        push!(q_obs, q_a)
        push!(regret, maximum(vals) - q_a)
    end

    return Float32[
        mean(q_obs), std(q_obs),
        mean(regret), std(regret),
        minimum(regret), maximum(regret),
        Float32(T)
    ] #7 dims
end

function featurize_concat(states_str::String, actions_str::String, goals::Vector{Symbol}, Qdict)
    feats = Float32[]
    for g in goals
        append!(feats, featurize_under_goal(states_str, actions_str, g, Qdict))
    end
    return feats  # 7*|goals| dims
end

goals = sort(unique(Symbol.(df.goal_name)))

X = hcat([
    featurize_concat(String(r.states), String(r.actions), goals, Qdict)
    for r in eachrow(df)
]...)  # (7*G) x N
y_goal = Flux.onehotbatch([goal_to_i[Symbol(r.goal_name)] for r in eachrow(df)], 1:length(goals))
y_beta = Flux.onehotbatch([beta_to_i[Float64(r.beta)] for r in eachrow(df)], 1:length(betas))

N = size(X, 2)
perm = randperm(N)
ntrain = round(Int, 0.8N)

tr = perm[1:ntrain]
te = perm[ntrain+1:end]

Xtr = X[:, tr]
Xte = X[:, te]

y_goal_tr = y_goal[:, tr]
y_goal_te = y_goal[:, te]

y_beta_tr = y_beta[:, tr]
y_beta_te = y_beta[:, te]

#two-head network (shared trunk)
trunk = Chain(Dense(14, 32, relu), Dense(32, 32, relu))
head_goal = Dense(32, length(goals))
head_beta = Dense(32, length(betas))

function model(x)
    h = trunk(x)
    return head_goal(h), head_beta(h)
end

loss(x, yg, yb) = begin
    pg, pb = model(x)
    Flux.logitcrossentropy(pg, yg) + Flux.logitcrossentropy(pb, yb)
end


using Functors
Functors.@functor TwoHead  #tells Flux/Optimisers how to traverse params

struct TwoHead
    trunk
    head_goal
    head_beta
end

#make callable
function (m::TwoHead)(x)
    h = m.trunk(x)
    return m.head_goal(h), m.head_beta(h)
end

m = TwoHead(trunk, head_goal, head_beta)

opt = ADAM(1e-3)
opt_state = Flux.setup(opt, m)

for epoch in 1:nepochs
    perm = randperm(size(Xtr, 2))
    for i in 1:batchsize:length(perm)
        idx = perm[i:min(i+batchsize-1, end)]
        xb  = Xtr[:, idx]
        ygb = y_goal_tr[:, idx]
        ybb = y_beta_tr[:, idx]

        function loss_for(m)
            pg, pb = m(xb)
            Flux.logitcrossentropy(pg, ygb) +
            Flux.logitcrossentropy(pb, ybb)
        end

        grads = Flux.gradient(loss_for, m)[1]
        Flux.update!(opt_state, m, grads)
    end

    # eval on train
    pg_tr, pb_tr = m(Xtr)
    tr_goal_acc = mean(Flux.onecold(pg_tr, 1:length(goals)) .==
                       Flux.onecold(y_goal_tr, 1:length(goals)))
    tr_beta_acc = mean(Flux.onecold(pb_tr, 1:length(betas)) .==
                       Flux.onecold(y_beta_tr, 1:length(betas)))

    #eval on test
    pg_te, pb_te = m(Xte)
    te_goal_acc = mean(Flux.onecold(pg_te, 1:length(goals)) .==
                       Flux.onecold(y_goal_te, 1:length(goals)))
    te_beta_acc = mean(Flux.onecold(pb_te, 1:length(betas)) .==
                       Flux.onecold(y_beta_te, 1:length(betas)))

    println(
        "Epoch $epoch | ",
        "train(goal=", round(tr_goal_acc, digits=3),
        ", β=", round(tr_beta_acc, digits=3), ") | ",
        "test(goal=", round(te_goal_acc, digits=3),
        ", β=", round(te_beta_acc, digits=3), ")"
    )
end

pg, pb = m(Xte)

test_goal_acc = mean(
    Flux.onecold(pg, 1:length(goals)) .==
    Flux.onecold(y_goal_te, 1:length(goals))
)

test_beta_acc = mean(
    Flux.onecold(pb, 1:length(betas)) .==
    Flux.onecold(y_beta_te, 1:length(betas))
)

println("\nFinal TEST performance:")
println("  Goal accuracy: ", round(test_goal_acc, digits=3))
println("  Beta accuracy: ", round(test_beta_acc, digits=3))
