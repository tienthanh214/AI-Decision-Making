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
function randstep(š«::MG, s, a)
    # Create a randomized step base on action 
    sā² = rand(SetCategorical(š«.š®, [š«.T(s, a, sā²) for sā² in š«.š®]))
    r = š«.R(s,a)
    return sā², r
end

function simulate!(š«::MG, Ļ, start_state, k_max; k_reset = 0)
    # Simulate for learning-based algorithms
    s = start_state
    for k = 1:k_max
        a = Tuple(Ļi(s)() for Ļi in Ļ)
        sā², r = randstep(š«, s, a)
        for i in collect(1:length(Ļ))
            update!(Ļ[i], s, a, sā²)
        end
        s = sā²
        if (k_reset != 0 && k % k_reset == 0)
            s = start_state
        end
    end
    return Ļ
end

# -------------- FICTITIOUS PLAY ------------------
mutable struct MGFictitiousPlay
    # Definition for Fictitious Play
    š« # Markov game
    i # agent index
    Qi # state-action value estimates
    Ni # state-action counts
end

function MGFictitiousPlay(š«::MG, i)
    # Construct FP for agent i
    ā, š®, š, R = š«.ā, š«.š®, š«.š, š«.R
    Qi = Dict((s, a) => R(s, a)[i] for s in š® for a in joint(š))
    Ni = Dict((j, s, aj) => 1.0 for j in ā for s in š® for aj in š[j])
    return MGFictitiousPlay(š«, i, Qi, Ni)
end

function (Ļi::MGFictitiousPlay)(s)
    # Return a SimpleGamePolicy for state s, based on uitility
    š«, i, Qi = Ļi.š«, Ļi.i, Ļi.Qi
    ā, š®, š, T, R, Ī³ = š«.ā, š«.š®, š«.š, š«.T, š«.R, š«.Ī³
    Ļiā²(i,s) = SimpleGamePolicy(ai => Ļi.Ni[i,s,ai] for ai in š[i])
    Ļiā²(i) = MGPolicy(s => Ļiā²(i,s) for s in š®)
    Ļ = [Ļiā²(i) for i in ā]
    U(s,Ļ) = sum(Ļi.Qi[s,a]*probability(š«,s,Ļ,a) for a in joint(š))
    Q(s,Ļ) = reward(š«,s,Ļ,i) + Ī³*sum(transition(š«,s,Ļ,sā²)*U(sā²,Ļ) for sā² in š®)
    Q(ai) = Q(s, joint(Ļ, SimpleGamePolicy(ai), i))
    ai = argmax(Q, š«.š[Ļi.i])
    return SimpleGamePolicy(ai)
end

function update!(Ļi::MGFictitiousPlay, s, a, sā²)
    # Update Fictitious Play based on simulated actions
    š«, i, Qi = Ļi.š«, Ļi.i, Ļi.Qi
    ā, š®, š, T, R, Ī³ = š«.ā, š«.š®, š«.š, š«.T, š«.R, š«.Ī³
    for (j,aj) in enumerate(a)
        Ļi.Ni[j,s,aj] += 1
    end
    Ļiā²(i,s) = SimpleGamePolicy(ai => Ļi.Ni[i,s,ai] for ai in š[i])
    Ļiā²(i) = MGPolicy(s => Ļiā²(i,s) for s in š®)
    Ļ = [Ļiā²(i) for i in ā]
    U(Ļ,s) = sum(Ļi.Qi[s,a]*probability(š«,s,Ļ,a) for a in joint(š))
    Q(s,a) = R(s,a)[i] + Ī³*sum(T(s,a,sā²)*U(Ļ,sā²) for sā² in š®)
    for a in joint(š)
        Ļi.Qi[s,a] = Q(s,a)
    end
end

# -------------- FICTITIOUS PLAY SIMULATING ------------------
function MGFPtoMGPolicy(š«::MG, Ļi::MGFictitiousPlay)
    # Translate from MGFictitiousPlay to MGPolicy
    return MGPolicy(s => Ļi(s) for s in š«.š®)
end

function fictitious_play(pphw::DecisionMakingProblems.PredatorPreyHexWorldMG, k_max)
    # Concurrent simluating for Fititiious Play algorithm
    š« = MG(pphw)
    Ļ = [MGFictitiousPlay(š«, i) for i in š«.ā]
    for i in collect(1:k_max)
        print("Iter: ", i, '/', k_max, '\n')
        for (i, s) in enumerate(š«.š®)
            if s[1] == s[2]
                continue
            end
            print(i, '/', length(š«.š®), '\n')
            Ļ = simulate!(š«, Ļ, s, 10)
        end
    end    
    Ļ = [MGFPtoMGPolicy(š«, Ļi) for Ļi in Ļ]
    return Ļ
end

# -------------- LEARNING ---------------------
pphw = PredatorPreyHexWorld()
š« = MG(pphw)
Ļ = fictitious_play(pphw, 30)
save("trained_policy/trained_FP.jld", "trained_pi", Ļ)