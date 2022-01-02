include("simple_game.jl")


const N_AGENTS = 2
const ACTIONS = vec(collect(2:20))

function joint_reward(a::Tuple{Int64,Int64})
    ai, aj = a
    return ai == aj ? (ai, aj) : (ai < aj ? (ai + 2, ai - 2) : (aj - 2, aj + 2))
end

travelersDilemma = SimpleGame(
    1.0,
    vec(collect(1:N_AGENTS)),
    [ACTIONS for _ in 1:N_AGENTS],
    joint_reward)