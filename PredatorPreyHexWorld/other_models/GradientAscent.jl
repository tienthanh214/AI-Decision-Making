# -------------- IMPORT PACKAGE ------------------
# import Pkg

# import_packages = ["Plots", "Ipopt", "JuMP", "Distributions", "LinearAlgebra", "JLD", "Plots"]
# for pkg in import_packages
#     if !haskey(Pkg.installed(), pkg)
#         Pkg.add(pkg)
#     end
# end

# if !haskey(Pkg.installed(), "DecisionMakingProblems")
#     Pkg.add(url = "https://github.com/algorithmsbooks/DecisionMakingProblems.jl")
# end

import Base.Iterators: product
using JuMP, Ipopt, Distributions, LinearAlgebra, JLD, Plots
using DecisionMakingProblems

struct SetCategorical{S}
    elements::Vector{S} # Set elements (could be repeated)
    distr::Categorical # Categorical distribution over set elements
    
    function SetCategorical(elements::AbstractVector{S}) where S
        weights = ones(length(elements))
        return new{S}(elements, Categorical(normalize(weights, 1)))
    end

    function SetCategorical(elements::AbstractVector{S}, weights::AbstractVector{Float64}) where S
        ℓ₁ = norm(weights,1)
        if ℓ₁ < 1e-6 || isinf(ℓ₁)
            return SetCategorical(elements)
        end
        distr = Categorical(normalize(weights, 1))
        return new{S}(elements, distr)
    end
end

Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]
Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]

    function Distributions.pdf(D::SetCategorical, x)
    sum(e == x ? w : 0.0 for (e,w) in zip(D.elements, D.distr.p))
end

# -------------- POLICY ------------------
struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k,v) in zip(keys(p), vs)))
    end
    
    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)

function (πi::SimpleGamePolicy)()
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)
end

joint(X) = vec(collect(product(X...)))
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    return sum(R(a)[i]*p(a) for a in joint(𝒜))
end

struct MGPolicy
    p # dictionary mapping states to simple game policies
    MGPolicy(p::Base.Generator) = new(Dict(p))
end

(πi::MGPolicy)(s, ai) = πi.p[s](ai)
(πi::SimpleGamePolicy)(s, ai) = πi(ai)

probability(𝒫::MG, s, π, a) = prod(πj(s, aj) for (πj, aj) in zip(π, a))
reward(𝒫::MG, s, π, i) =
    sum(𝒫.R(s,a)[i]*probability(𝒫,s,π,a) for a in joint(𝒫.𝒜))
transition(𝒫::MG, s, π, s′) =
    sum(𝒫.T(s,a,s′)*probability(𝒫,s,π,a) for a in joint(𝒫.𝒜))

function policy_evaluation(𝒫::MG, π, i)
    𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
    p(s,a) = prod(πj(s, aj) for (πj, aj) in zip(π, a))
    R′ = [sum(R(s,a)[i]*p(s,a) for a in joint(𝒜)) for s in 𝒮]
    T′ = [sum(T(s,a,s′)*p(s,a) for a in joint(𝒜)) for s in 𝒮, s′ in 𝒮]
    return (I - γ*T′)\R′
end

# -------------- SIMULATE FOR LEARNING ------------------
function randstep(𝒫::MG, s, a)
    s′ = rand(SetCategorical(𝒫.𝒮, [𝒫.T(s, a, s′) for s′ in 𝒫.𝒮]))
    r = 𝒫.R(s,a)
    return s′, r
end

function simulate(𝒫::MG, π, start_state, k_max; k_reset = typemax(Int64))
    print("Start state: ", start_state, '\n')
    s = start_state
    for k = 1:k_max
        if k % 100 == 0
            print(k, '/', k_max, '\n')
        end
        a = Tuple(πi(s)() for πi in π)
        s′, r = randstep(𝒫, s, a)
        for (i, πi) in enumerate(π)
            update!(πi, s, a, s′)
            π[i] = πi
        end
        s = s′
        if k % k_reset == 0
            s = start_state
        end
    end
    return π
end

# -------------- GRADIENT ASCENT ------------------
function project_to_simplex(y)
    u = sort(copy(y), rev=true)
    i = maximum([j for j in eachindex(u) if u[j] + (1 - sum(u[1:j])) / j > 0.0])
    δ = (1 - sum(u[j] for j = 1:i)) / i
    return [max(y[j] + δ, 0.0) for j in eachindex(u)]
end

mutable struct MGGradientAscent
    𝒫 # Markov game
    i # agent index
    t # time step
    Qi # state-action value estimates
    πi # current policy
end

function MGGradientAscent(𝒫::MG, i)
    ℐ, 𝒮, 𝒜 = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜
    Qi = Dict((s, a) => 0.0 for s in 𝒮, a in joint(𝒜))
    uniform() = Dict(s => SimpleGamePolicy(ai => 1.0 for ai in 𝒫.𝒜[i])
                    for s in 𝒮)
    return MGGradientAscent(𝒫, i, 1, Qi, uniform())
end

function (πi::MGGradientAscent)(s)
    𝒜i, t = πi.𝒫.𝒜[πi.i], πi.t
    ϵ = 1 / sqrt(t)
    πi′(ai) = ϵ/length(𝒜i) + (1-ϵ)*πi.πi[s](ai)
    return SimpleGamePolicy(ai => πi′(ai) for ai in 𝒜i)
end

function update!(πi::MGGradientAscent, s, a, s′)
    𝒫, i, t, Qi = πi.𝒫, πi.i, πi.t, πi.Qi
    ℐ, 𝒮, 𝒜i, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜[πi.i], 𝒫.R, 𝒫.γ
    jointπ(ai) = Tuple(j == i ? ai : a[j] for j in ℐ)
    α = 1 / sqrt(t)
    Qmax = maximum(Qi[s′, jointπ(ai)] for ai in 𝒜i)
    πi.Qi[s, a] += α * (R(s, a)[i] + γ * Qmax - Qi[s, a])
    u = [Qi[s, jointπ(ai)] for ai in 𝒜i]
    π′ = [πi.πi[s](ai) for ai in 𝒜i]
    π = project_to_simplex(π′ + u / sqrt(t))
    πi.t = t + 1
    πi.πi[s] = SimpleGamePolicy(ai => p for (ai, p) in zip(𝒜i, π))
end

# -------------- GRADIENT ASCENT SIMULATING ------------------
function GAtoMGPolicy(𝒫::MG, πi::MGGradientAscent)
    return MGPolicy(s => πi(s) for s in 𝒫.𝒮)
end

function gradient_ascent(PPHW::DecisionMakingProblems.PredatorPreyHexWorldMG, k_max)
    𝒫 = MG(PPHW)
    π = [MGGradientAscent(𝒫, i) for i in 𝒫.ℐ]
    cnt = 0
    for s in 𝒫.𝒮
        cnt += 1
        if s[1] == s[2]
            continue
        end
        if s != (3, 10)
            continue
        end
        print(cnt, '/', length(𝒫.𝒮), '\n')
        π = simulate(𝒫, π, s, k_max, k_reset=20)
    end
    π = [GAtoMGPolicy(𝒫, π[i]) for i in 𝒫.ℐ]
    return π
end

pphw = PredatorPreyHexWorld()
𝒫 = MG(pphw)
π = gradient_ascent(pphw, 1000000)
save("trained_GA.jld", "trained_pi", π)