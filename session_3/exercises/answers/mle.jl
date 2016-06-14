using Optim
using StatsFuns

function skewnormpdf(x::Real, ξ::Real, ω::Real, α::Real)
    z = (x-ξ)/ω
    2.0*(normpdf(z)/ω)*normcdf(α*z)
end

function skewnormpdf2(x::Real, ξ::Real, logω::Real, α::Real)
    skewnormpdf(x,ξ, exp(logω), α)
end

function loglikelihood(x::Vector{Float64}, params::Vector{Float64})
    ξ, logω, α = params
    sum([log(skewnormpdf2(xi, ξ, logω, α)) for xi in x])
end

data = vec(readdlm("../data.txt"))
# Recall that optimize minimizes rather than maximizes so we must specify the negative log likelihood
results = optimize(params-> -loglikelihood(data, params), [0.0, 0.0, 0.0])
mle = Optim.minimizer(results)
ξ, ω, α = mle[1], exp(mle[2]), mle[3]

