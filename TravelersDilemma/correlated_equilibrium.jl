include("properties.jl")


# Algorithm 24.6 (Algorithms for Decision Making)
# Policy của Correlated Equilibrium
mutable struct JointCorrelatedPolicy
    p # dictionary mapping from joint actions to probabilities
    JointCorrelatedPolicy(p::Base.Generator) = new(Dict(p))
end

# Trả về policy của joint action a
(π::JointCorrelatedPolicy)(a) = get(π.p, a, 0.0)

# Trả về action ngẫu nhiên trong policy π
function (π::JointCorrelatedPolicy)()
    D = SetCategorical(collect(keys(π.p)), collect(values(π.p)))
    return rand(D)
end


# Algorithm 24.7 (Algorithms for Decision Making) Utilitarian | Fixed bug | Modified
struct CorrelatedEquilibrium end

# Hàm đổi action trong công thức
joint(a, ai′, i) = Tuple(k == i ? ai′ : v for (k, v) in enumerate(a))

function solve(M::CorrelatedEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    model = Model(Ipopt.Optimizer)
    # Khai báo cũng như ràng buộc 3
    @variable(model, π[joint(𝒜)] ≥ 0)
    # Hàm mục tiêu
    @objective(model, Max, sum(sum(π[a] * R(a)[i] for a in joint(𝒜)) for i in ℐ))
    # Ràng buộc 1
    @constraint(model, [i = ℐ, ai = 𝒜[i], ai′ = 𝒜[i]],
        sum(R(a)[i] * π[a] for a in joint(𝒜) if a[i] == ai)
        ≥
        sum(R(joint(a, ai′, i))[i] * π[a] for a in joint(𝒜) if a[i] == ai))
    # Ràng buộc 2
    @constraint(model, sum(π) == 1)
    # Tối ưu mô hình
    optimize!(model)
    return JointCorrelatedPolicy(a => value(π[a]) for a in joint(𝒜))
end


# Giải
π = solve(CorrelatedEquilibrium(), travelersDilemma)

# Ghi kết quả
for x in π.p
    println(x)
end