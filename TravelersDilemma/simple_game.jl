include("resource.jl")


# Algorithm 24.1 (Algorithms for Decision Making)
struct SimpleGame
    γ # discount factor
    ℐ # agents
    𝒜 # joint action space
    R # joint reward function
end


# Algorithm 24.2 (Algorithms for Decision Making)
struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities

    # Trả về policy ngẫu nhiên
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    # Trả về policy từ dict
    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k, v) in zip(keys(p), vs)))
    end

    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

# Trả về policy của ai
(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)

# Trả về hành động ngẫu nhiên trong policy πi
function (πi::SimpleGamePolicy)()
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)
end

# Trả về tích các vector
joint(X) = vec(collect(Iterators.product(X...)))

joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)] # helper of best_response

# Hàm lợi ích kỳ vọng từ 𝒫 và policy π của agent i
function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    return sum(R(a)[i] * p(a) for a in joint(𝒜))
end


# Algorithm 24.3 (Algorithms for Decision Making)
# Trả về deterministic policy tốt nhất
function best_response(𝒫::SimpleGame, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    ai = argmax(U, 𝒫.𝒜[i])
    return SimpleGamePolicy(ai)
end
