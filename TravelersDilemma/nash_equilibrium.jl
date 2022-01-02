include("properties.jl")


# Algorithm 24.5 (Algorithms for Decision Making)
struct NashEquilibrium end

# Trả về dạng tensor của 𝒫
function tensorform(𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    ℐ′ = eachindex(ℐ)
    𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
    R′ = [R(a) for a in joint(𝒜)]
    return ℐ′, 𝒜′, R′
end

# Tìm Nash Equilibrium
function solve(M::NashEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = tensorform(𝒫)
    model = Model(Ipopt.Optimizer)
    # Khai báo
    @variable(model, U[ℐ])
    # Ràng buộc 3
    @variable(model, π[i = ℐ, 𝒜[i]] ≥ 0)
    # Hàm mục tiêu
    @NLobjective(model, Min,
        sum(U[i] - sum(prod(π[j, a[j]] for j in ℐ) * R[y][i]
                       for (y, a) in enumerate(joint(𝒜))) for i in ℐ))
    # Ràng buộc 1
    @NLconstraint(model, [i = ℐ, ai = 𝒜[i]],
        U[i] ≥ sum(
            prod(j == i ? (a[j] == ai ? 1.0 : 0.0) : π[j, a[j]] for j in ℐ)
            *
            R[y][i] for (y, a) in enumerate(joint(𝒜))))
    # Ràng buộc 2
    @constraint(model, [i = ℐ], sum(π[i, ai] for ai in 𝒜[i]) == 1)
    # Tối ưu mô hình
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai] => value(π[i, ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end


# Giải
π = solve(NashEquilibrium(), travelersDilemma)

π¹ = π[1].p
π² = π[2].p

# Ghi kết quả
for a in ACTIONS
    println(a => (π¹[a], π²[a]))
end