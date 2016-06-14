# Constructs a random PSD matrix
ran_psd <- function(n) {
    A <- matrix(runif(n*n), nrow=n)
    return(crossprod(A)/sqrt(n))
}

# Target density used in MCMC below
target <- function(x) {
    return(1.0/(1+x^4))
}

# Takes sample size n from distribution
# whose pdf is proportional to 1/(1+x^4)
# using Metropolis-Hastings with standard
# Normal proposal
mcmc <- function(n) {
    x = vector(mode="numeric", length=n)
    x[1] = 0.0 # Starting point
    for (i in 2:n) {
        prop=rnorm(1, mean=x[i-1])
        acc_rat = target(prop)/density(x[i-1])
        if (acc_rat >= 1 || runif(1) < acc_rat)
            x[i] = prop
        else {
            x[i] = x[i-1]
        }
    }
    return (x)
}
