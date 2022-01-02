include("simple_game.jl")


# Khai báo số agent và action của mỗi agent
const N_AGENTS = 2
ACTIONS = vec(collect(2:100))

# Hàm joint reward
function joint_reward(a::Tuple{Int64,Int64})
    ai, aj = a
    return ai == aj ? (ai, aj) : (ai < aj ? (ai + 2, ai - 2) : (aj - 2, aj + 2))
end

# Biến trò chơi của bài toán
travelersDilemma = SimpleGame(
    1.0,
    vec(collect(1:N_AGENTS)),
    [ACTIONS for _ in 1:N_AGENTS],
    joint_reward)