include("simple_game.jl")


const N_AGENTS = 2
const ACTIONS = vec(collect(2:100))

function joint_reward(a::Tuple{Int64, Int64})
    ai, aj = a
    ai = Float64(ai)
    aj = Float64(aj)
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