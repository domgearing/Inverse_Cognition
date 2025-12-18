using CSV, DataFrames

df = CSV.read("data/episodes.csv", DataFrame)

include("GridWorld.jl")
include("Agent.jl")
include("Inference.jl")

using .GridWorld

w = GridWorld.World(
    5, 5,
    walls = [GridWorld.State(3,3)],
    goals = Dict(
        :A => GridWorld.State(5,5),
        :B => GridWorld.State(1,5)
    )
)

#load data
df = CSV.read("data/episodes.csv", DataFrame)

#define world
w = GridWorld.World(
    5, 5,
    walls = [GridWorld.State(3,3)],
    goals = Dict(
        :A => GridWorld.State(5,5),
        :B => GridWorld.State(1,5)
    )
)

Qdict = Dict{Symbol, Dict{Tuple{Int,Int}, Dict{GridWorld.Action,Float64}}}()
for g in [:A, :B]
    Qdict[g] = Agent.compute_Q_for_goal(w, g)
end


#helpers
goal_sym(x) = x isa Symbol ? x : Symbol(String(x))

function map_argmax(post::Dict{Tuple{Symbol,Float64},Float64})
    best_k = first(keys(post))
    best_v = -Inf
    for (k,v) in post
        if v > best_v
            best_v = v
            best_k = k
        end
    end
    return best_k, best_v
end

function idxmap(vals)
    Dict(v => i for (i,v) in enumerate(vals))
end

#eval
function evaluate_all(df::DataFrame, w, Qdict; goals=[:A,:B], betas=[0.5,1.0,3.0,8.0], ϵ=0.02, λ=0.2)

    goals = collect(goals)
    betas = collect(betas)

    g2i = idxmap(goals)
    b2i = idxmap(betas)

    #confusion matrix (rows=true, cols=pred)
    goal_conf = zeros(Int, length(goals), length(goals))
    beta_conf = zeros(Int, length(betas), length(betas))

    #joint confusion - (goal,beta)
    joint_labels = [(g,β) for g in goals for β in betas]
    j2i = idxmap(joint_labels)
    joint_conf = zeros(Int, length(joint_labels), length(joint_labels))

    n = nrow(df)
    n_goal_correct = 0
    n_beta_correct = 0
    n_joint_correct = 0

    for i in 1:n
        row = df[i, :]

        true_g = goal_sym(row.goal_name)
        true_β = Float64(row.beta)

        states  = Inference.parse_states(String(row.states))
        actions = Inference.parse_actions(String(row.actions))

        post = Inference.posterior_over_goal_and_beta(
    w, Qdict, states, actions;
    goals=goals, betas=betas, ϵ=ϵ, λ=λ
)

        (pred_g, pred_β), _ = map_argmax(post)

        #update goal confusion
        goal_conf[g2i[true_g], g2i[pred_g]] += 1
        if pred_g == true_g
            n_goal_correct += 1
        end

        #update beta confusion
        beta_conf[b2i[true_β], b2i[pred_β]] += 1
        if pred_β == true_β
            n_beta_correct += 1
        end

        #update joint confusion
        joint_conf[j2i[(true_g,true_β)], j2i[(pred_g,pred_β)]] += 1
        if pred_g == true_g && pred_β == true_β
            n_joint_correct += 1
        end
    end

    goal_acc  = n_goal_correct / n
    beta_acc  = n_beta_correct / n
    joint_acc = n_joint_correct / n

    return (
        goal_conf=goal_conf,
        beta_conf=beta_conf,
        joint_conf=joint_conf,
        goals=goals,
        betas=betas,
        joint_labels=joint_labels,
        goal_acc=goal_acc,
        beta_acc=beta_acc,
        joint_acc=joint_acc
    )
end


lambdas = [0.0, 0.05, 0.1, 0.2, 0.3]
epsilons = [0.0, 0.01, 0.02, 0.05]

println("\nGrid search over (λ, ϵ)")
println("λ\tϵ\tGoalAcc\tBetaAcc\tJointAcc")

results_table = []

for λ in lambdas, ϵ in epsilons
    res = evaluate_all(
        df, w, Qdict;
        goals=[:A,:B],
        betas=[0.5,1.0,3.0,8.0],
        λ=λ,
        ϵ=ϵ
    )

    push!(results_table, (λ=λ, ϵ=ϵ,
                           goal=res.goal_acc,
                           beta=res.beta_acc,
                           joint=res.joint_acc))

    println(
        round(λ, digits=2), "\t",
        round(ϵ, digits=2), "\t",
        round(res.goal_acc,  digits=3), "\t",
        round(res.beta_acc,  digits=3), "\t",
        round(res.joint_acc, digits=3)
    )
end

#req. goal accuracy >= 0.98, maximize beta accuracy
filtered = filter(r -> r.goal ≥ 0.98, results_table)

best = argmax(r -> r.beta, filtered)

println("\nBest hyperparameters (goal ≥ 0.98):")
println("  λ = ", best.λ, ", ϵ = ", best.ϵ)
println("  Goal accuracy  = ", round(best.goal,  digits=3))
println("  Beta accuracy  = ", round(best.beta,  digits=3))
println("  Joint accuracy = ", round(best.joint, digits=3))

#run
df = CSV.read("data/episodes.csv", DataFrame)

results = evaluate_all(df, w, Qdict; goals=[:A,:B], betas=[0.5,1.0,3.0,8.0], ϵ=0.02)

println("\nAccuracy:")
println("  Goal accuracy:  ", round(results.goal_acc,  digits=3))
println("  Beta accuracy:  ", round(results.beta_acc,  digits=3))
println("  Joint accuracy: ", round(results.joint_acc, digits=3))

println("\nGoal confusion (rows=true, cols=pred). Order = ", results.goals)
println(results.goal_conf)

println("\nBeta confusion (rows=true, cols=pred). Order = ", results.betas)
println(results.beta_conf)

using StatsPlots
using FilePathsBase

#setup
figdir = "figures"
mkpath(figdir)

#goal confusion
goal_labels = string.(results.goals)

p_goal = heatmap(
    goal_labels,
    goal_labels,
    results.goal_conf;
    xlabel = "Predicted goal",
    ylabel = "True goal",
    title  = "Goal Confusion Matrix",
    cbar   = true,
    aspect_ratio = 1,
    annot = results.goal_conf
)

savefig(p_goal, joinpath(figdir, "goal_confusion.png"))

#beta confusion
beta_labels = string.(results.betas)

p_beta = heatmap(
    beta_labels,
    beta_labels,
    results.beta_conf;
    xlabel = "Predicted β",
    ylabel = "True β",
    title  = "Beta Confusion Matrix",
    cbar   = true,
    aspect_ratio = 1,
    annot = results.beta_conf
)

savefig(p_beta, joinpath(figdir, "beta_confusion.png"))

#row normalized beta confusion
normalize_rows(M) = M ./ sum(M, dims=2)

beta_norm = normalize_rows(results.beta_conf)

p_beta_norm = heatmap(
    beta_labels,
    beta_labels,
    beta_norm;
    xlabel = "Predicted β",
    ylabel = "True β",
    title  = "Beta Confusion Matrix (Row-normalized)",
    cbar   = true,
    aspect_ratio = 1,
    annot = round.(beta_norm, digits=2)
)

savefig(p_beta_norm, joinpath(figdir, "beta_confusion_normalized.png"))