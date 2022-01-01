# Algorithm 24.1. Data structure for a simple game.
struct SimpleGame
  γ # discount factor
  ℐ # agents
  𝒜 # joint action space
  R # joint reward function
end


# Algorithm 24.2
struct SimpleGamePolicy
  p # dictionary mapping actions to probabilities

  function SimpleGamePolicy(p::Base.Generator)
      return SimpleGamePolicy(Dict(p))
  end

  function SimpleGamePolicy(p::Dict)
      vs = collect(values(p))
      vs ./= sum(vs)
      return new(Dict(k => v for (k, v) in zip(keys(p), vs)))
  end

  SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)

function (πi::SimpleGamePolicy)()
  D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
  return rand(D)
end

joint(X) = vec(collect(Iterators.product(X...)))

joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

function utility(𝒫::SimpleGame, π, i)
  𝒜, R = 𝒫.𝒜, 𝒫.R
  p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
  return sum(R(a)[i] * p(a) for a in joint(𝒜))
end

import Pkg
if !haskey(Pkg.installed(), "Ipopt")
    Pkg.add("Ipopt")
end
if !haskey(Pkg.installed(), "JuMP")
    Pkg.add("JuMP")
end

using JuMP, Ipopt

const N_AGENTS = 2
const ACTIONS = vec(collect(2:10))

function joint_reward(a::Tuple{Int64, Int64})
    ai, aj = a
    if ai == aj
        return (ai, ai)
    elseif ai < aj
        return (ai + 2, ai - 2)
    end
    return (aj - 2, aj + 2)
end

travelersDilemma = SimpleGame(
    1.0,
    vec(collect(1:N_AGENTS)),
    [ACTIONS for _ in 1:N_AGENTS],
    joint_reward)