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


# Algorithm 24.7 (Utilitarian) [Fixed bug by me]
struct CorrelatedEquilibrium end

joint(a, ai′, i) = Tuple(k == i ? ai′ : v for (k, v) in enumerate(a))

function solve(M::CorrelatedEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    model = Model(Ipopt.Optimizer)
    @variable(model, π[joint(𝒜)] ≥ 0)
    @objective(model, Max, sum(sum(π[a] * R(a)[i] for a in joint(𝒜)) for i in ℐ))
    @constraint(model, [i = ℐ, ai = 𝒜[i], ai′ = 𝒜[i]],
        sum(R(a)[i] * π[a] for a in joint(𝒜) if a[i] == ai)
        ≥
        sum(R(joint(a, ai′, i))[i] * π[a] for a in joint(𝒜) if a[i] == ai))
    @constraint(model, sum(π) == 1)
    optimize!(model)
    return JointCorrelatedPolicy(a => value(π[a]) for a in joint(𝒜))
end


π = solve(CorrelatedEquilibrium(), travelersDilemma)

π¹ = Dict(a => 0.0 for a in travelersDilemma.𝒜[1])
π² = Dict(a => 0.0 for a in travelersDilemma.𝒜[2])

for (k, v) in π.p
    π¹[k[1]] += v
    π²[k[2]] += v
end

for i in travelersDilemma.𝒜[1]
    println(i => (π¹[i], π²[i]))
end
