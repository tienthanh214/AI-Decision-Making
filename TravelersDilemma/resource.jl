import Pkg

function addPackage(pkg::String)
    if !haskey(Pkg.installed(), pkg)
        Pkg.add(pkg)
    end
end

addPackage("Distributions")
addPackage("LinearAlgebra")
addPackage("JuMP")
addPackage("Ipopt")

using Distributions, LinearAlgebra, JuMP, Ipopt, Random


# Appendices (Algorithms for Decision Making)
# G.5 Convenience Functions
struct SetCategorical{S}
    elements::Vector{S} # Set elements (could be repeated)
    distr::Categorical # Categorical distribution over set elements

    # Trả về SetCategorical với phân phối đều
    function SetCategorical(elements::AbstractVector{S}) where {S}
        weights = ones(length(elements))
        return new{S}(elements, Categorical(normalize(weights, 1)))
    end

    # Trả về SetCategorical với phân phối weights
    function SetCategorical(elements::AbstractVector{S}, weights::AbstractVector{Float64}) where {S}
        ℓ₁ = norm(weights, 1)
        if ℓ₁ < 1e-6 || isinf(ℓ₁)
            return SetCategorical(elements)
        end
        distr = Categorical(normalize(weights, 1))
        return new{S}(elements, distr)
    end
end

# Trả về phần tử ngẫu nhiên trong SetCategorical D
Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]
Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]

function Distributions.pdf(D::SetCategorical, x)
    sum(e == x ? w : 0.0 for (e, w) in zip(D.elements, D.distr.p))
end
