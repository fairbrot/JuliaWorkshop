using RCall, StatsBase

###############
# Load script #
###############

R"source(\"../functions.R\")"

###############################################
# Comparison of random matrix multiplications #
###############################################

function ran_psd_julia(n::Int)
    A = rand(n,n)
    return (A'A)/sqrt(n)
end

function ran_psd_R(n::Int)
    x = R"ran_psd($(n))"
    return rcopy(x)
end

# Warm-up runs
ran_psd_julia(10);
ran_psd_R(10);

# Time comparison
@time ran_psd_julia(1000);
@time ran_psd_R(1000);

#####################################
# Comparison of Metropolis-Hastings #
#####################################

function target(x::Real)
    return 1.0/(1+x^4)
end

function mcmc_julia(n::Int)
    x = Array(Float64, n)
    x[1] = 0.0
    for i in 2:n
        prop = randn() + x[i-1]
        α = target(prop)/density(x[i-1])
        if (α >= 1 || rand() < α)
            x[i] = prop
        else
            x[i] = x[i-1]
        end
    end
    return x
end

function mcmc_R(n::Int)
    x = R"mcmc($(n))"
    return rcopy(x)
end

# Warm-up runs
mcmc_julia(10);
mcmc_R(10);

# Time comparison
@time mcmc_julia(10000);
@time mcmc_R(10000);
