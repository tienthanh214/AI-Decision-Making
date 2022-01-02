# main algorithm for Predator-Prey Hex World problem

# -------------- IMPORT PACKAGE AND INCLUDE ------------------
import Pkg

import_packages = ["Ipopt", "JuMP", "Distributions", "LinearAlgebra", "JLD", "Plots"]
for pkg in import_packages
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

if !haskey(Pkg.installed(), "DecisionMakingProblems")
    Pkg.add(url = "https://github.com/algorithmsbooks/DecisionMakingProblems.jl")
end

using JLD
using DecisionMakingProblems
include("simple_game.jl")
include("markov_game.jl")

# -------------- SIMULATE FOR LEARNING ------------------
function randstep(𝒫::MG, s, a)
    # Create a randomized step base on action 
    s′ = rand(SetCategorical(𝒫.𝒮, [𝒫.T(s, a, s′) for s′ in 𝒫.𝒮]))
    r = 𝒫.R(s,a)
    return s′, r
end

function simulate!(𝒫::MG, π, start_state, k_max; k_reset = 0)
    # Simulate for learning-based algorithms
    s = start_state
    for k = 1:k_max
        a = Tuple(πi(s)() for πi in π)
        s′, r = randstep(𝒫, s, a)
        for i in collect(1:length(π))
            update!(π[i], s, a, s′)
        end
        s = s′
        if (k_reset != 0 && k % k_reset == 0)
            s = start_state
        end
    end
    return π
end

# -------------- FICTITIOUS PLAY ------------------
mutable struct MGFictitiousPlay
    # Definition for Fictitious Play
    𝒫 # Markov game
    i # agent index
    Qi # state-action value estimates
    Ni # state-action counts
end

function MGFictitiousPlay(𝒫::MG, i)
    # Construct FP for agent i
    ℐ, 𝒮, 𝒜, R = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R
    Qi = Dict((s, a) => R(s, a)[i] for s in 𝒮 for a in joint(𝒜))
    Ni = Dict((j, s, aj) => 1.0 for j in ℐ for s in 𝒮 for aj in 𝒜[j])
    return MGFictitiousPlay(𝒫, i, Qi, Ni)
end

function (πi::MGFictitiousPlay)(s)
    # Return a SimpleGamePolicy for state s, based on uitility
    𝒫, i, Qi = πi.𝒫, πi.i, πi.Qi
    ℐ, 𝒮, 𝒜, T, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ
    πi′(i,s) = SimpleGamePolicy(ai => πi.Ni[i,s,ai] for ai in 𝒜[i])
    πi′(i) = MGPolicy(s => πi′(i,s) for s in 𝒮)
    π = [πi′(i) for i in ℐ]
    U(s,π) = sum(πi.Qi[s,a]*probability(𝒫,s,π,a) for a in joint(𝒜))
    Q(s,π) = reward(𝒫,s,π,i) + γ*sum(transition(𝒫,s,π,s′)*U(s′,π) for s′ in 𝒮)
    Q(ai) = Q(s, joint(π, SimpleGamePolicy(ai), i))
    ai = argmax(Q, 𝒫.𝒜[πi.i])
    return SimpleGamePolicy(ai)
end

function update!(πi::MGFictitiousPlay, s, a, s′)
    # Update Fictitious Play based on simulated actions
    𝒫, i, Qi = πi.𝒫, πi.i, πi.Qi
    ℐ, 𝒮, 𝒜, T, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ
    for (j,aj) in enumerate(a)
        πi.Ni[j,s,aj] += 1
    end
    πi′(i,s) = SimpleGamePolicy(ai => πi.Ni[i,s,ai] for ai in 𝒜[i])
    πi′(i) = MGPolicy(s => πi′(i,s) for s in 𝒮)
    π = [πi′(i) for i in ℐ]
    U(π,s) = sum(πi.Qi[s,a]*probability(𝒫,s,π,a) for a in joint(𝒜))
    Q(s,a) = R(s,a)[i] + γ*sum(T(s,a,s′)*U(π,s′) for s′ in 𝒮)
    for a in joint(𝒜)
        πi.Qi[s,a] = Q(s,a)
    end
end

# -------------- FICTITIOUS PLAY SIMULATING ------------------
function MGFPtoMGPolicy(𝒫::MG, πi::MGFictitiousPlay)
    # Translate from MGFictitiousPlay to MGPolicy
    return MGPolicy(s => πi(s) for s in 𝒫.𝒮)
end

function fictitious_play(pphw::DecisionMakingProblems.PredatorPreyHexWorldMG, k_max)
    # Concurrent simluating for Fititiious Play algorithm
    𝒫 = MG(pphw)
    π = [MGFictitiousPlay(𝒫, i) for i in 𝒫.ℐ]
    for i in collect(1:k_max)
        print("Iter: ", i, '/', k_max, '\n')
        for (i, s) in enumerate(𝒫.𝒮)
            if s[1] == s[2]
                continue
            end
            print(i, '/', length(𝒫.𝒮), '\n')
            π = simulate!(𝒫, π, s, 10)
        end
    end    
    π = [MGFPtoMGPolicy(𝒫, πi) for πi in π]
    return π
end

# -------------- LEARNING ---------------------
pphw = PredatorPreyHexWorld()
𝒫 = MG(pphw)
π = fictitious_play(pphw, 30)
save("trained_policy/trained_FP.jld", "trained_pi", π)