include("properties.jl")


# Algorithm 24.11 (Algorithms for Decision Making)
# Mô hình Fictitious Play
mutable struct FictitiousPlay
    𝒫 # simple game
    i # agent index
    N # array of action count dictionaries
    πi # current policy
end

# Constructor của mô hình
function FictitiousPlay(𝒫::SimpleGame, i)
    N = [Dict(aj => 1 for aj in 𝒫.𝒜[j]) for j in 𝒫.ℐ]
    πi = SimpleGamePolicy(ai => 1.0 for ai in 𝒫.𝒜[i])
    return FictitiousPlay(𝒫, i, N, πi)
end

# Trả về action ngẫu nhiên từ policy πi
(πi::FictitiousPlay)() = πi.πi()

# Trả về xác suất của action ai
(πi::FictitiousPlay)(ai) = πi.πi(ai)

# Cập nhật mô hình Fictitious Play
function update!(πi::FictitiousPlay, a)
    N, 𝒫, ℐ, i = πi.N, πi.𝒫, πi.𝒫.ℐ, πi.i
    # Tăng số lần các action xuất hiện
    for (j, aj) in enumerate(a)
        N[j][aj] += 1
    end
    # Cập nhật policy mới
    p(j) = SimpleGamePolicy(aj => u / sum(values(N[j])) for (aj, u) in N[j])
    π = [p(j) for j in ℐ]
    πi.πi = best_response(𝒫, π, i)
end


# Algorithm 24.10 (Algorithms for Decision Making)
# Lặp để cập nhật mô hình
function simulate(𝒫::SimpleGame, π, k_max)
    for k = 1:k_max
        a = [πi() for πi in π]
        for πi in π
            update!(πi, a)
        end
    end
    return π
end

# Thử nghiệm các tham số k_max và ghi kết quả
for k_max in [100, 1000, 10000, 100000]
    π = simulate(
        travelersDilemma,
        [FictitiousPlay(travelersDilemma, i) for i in travelersDilemma.ℐ],
        k_max)

    println("After ", k_max, " iterations, the (deterministic) policy:")
    
    π¹ = π[1].πi
    π² = π[2].πi
    
    println("π¹ = ", π¹)
    println("π² = ", π²)
    println()
end