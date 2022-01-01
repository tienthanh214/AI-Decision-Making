include("properties.jl")


# Algorithm 24.6
mutable struct JointCorrelatedPolicy
    p # dictionary mapping from joint actions to probabilities
    JointCorrelatedPolicy(p::Base.Generator) = new(Dict(p))
end

(π::JointCorrelatedPolicy)(a) = get(π.p, a, 0.0)

function (π::JointCorrelatedPolicy)()
    D = SetCategorical(collect(keys(π.p)), collect(values(π.p)))
    return rand(D)
end


# Algorithm 24.7
struct CorrelatedEquilibrium end

function solve(M::CorrelatedEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    model = Model(Ipopt.Optimizer)
    @variable(model, π[joint(𝒜)] ≥ 0)
    @objective(model, Max, sum(π[a] * R(a) for a in joint(𝒜)))
    @constraint(model, [i = ℐ, ai = 𝒜[i], ai′ = 𝒜[i]],
        sum(R(a)[i] * π[a] for a in joint(𝒜) if a[i] == ai)
        ≥
        sum(R(joint(a, ai′, i))[i] * π[a] for a in joint(𝒜) if a[i] == ai))
    @constraint(model, sum(π) == 1)
    optimize!(model)
    return JointCorrelatedPolicy(a => value(π[a]) for a in joint(𝒜))
end

π = solve(CorrelatedEquilibrium(), travelersDilemma)