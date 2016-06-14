using Distributions

##############
# Simulation #
##############

function simulate(dist::Sampleable, n::Int, f::Function)
    samp = map(f, rand(dist, n))
    m = mean(samp)
    dist = Normal()
    gap = quantile(dist, 0.925) * std(samp)/sqrt(n)
    return m, (m - gap, m + gap)    
end

simulate(Gamma(1,2), 1000, x->(1.0/(1.0 + x^2)))
simulate(TDist(5.0), 1000, x->(1.0/(1.0 + x^2)))
simulate(Frechet(1.0, 2.0), 1000, x->(1.0/(1.0 + x^2)))

##########################
# Operators for MvNormal #
##########################

import Base: *, +
*(A::Matrix{Float64}, dist::MvNormal) = MvNormal(A*mean(dist), A'cov(dist)*A)

+(dist::MvNormal, a::Vector{Float64}) = MvNormal(mean(dist) + a, cov(dist))
+(a::Vector{Float64}, dist::MvNormal) = dist + a

dist = MvNormal(eye(3))
A = rand(3,3)
a = rand(3)
a + A*dist
