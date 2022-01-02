include("properties.jl")


# Algorithm 24.4 (Algorithms for Decision Making)
# Hàm trả về policy của mô hình softmax
function softmax_response(𝒫::SimpleGame, π, i, λ)
    𝒜i = 𝒫.𝒜[i]
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    return SimpleGamePolicy(ai => exp(λ * U(ai)) for ai in 𝒜i)
end


# Algorithm 24.9 (Algorithms for Decision Making)
# Cấu trúc mô tả thông số của mô hình Hierarchical Softmax
struct HierarchicalSoftmax
    λ # precision parameter
    k # level
    π # initial policy
end

# Constructor của HierarchicalSoftmax
function HierarchicalSoftmax(𝒫::SimpleGame, λ, k)
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
    return HierarchicalSoftmax(λ, k, π)
end

# Hàm lặp để điều chỉnh mô hình softmax
function solve(M::HierarchicalSoftmax, 𝒫)
    π = M.π
    for k in 1:M.k
        π = [softmax_response(𝒫, π, i, M.λ) for i in 𝒫.ℐ]
    end
    return π
end

# Giải
π = solve(HierarchicalSoftmax(travelersDilemma, 0.3, 4), travelersDilemma)

π¹ = π[1].p
π² = π[2].p

# Ghi kết quả
for a in ACTIONS
    println(a => (π¹[a], π²[a]))
end