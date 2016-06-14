using Gadfly, Changepoints

##########################################
# Sampling time series with changepoints #
##########################################

n = 100 # Length of time series
λ = 10  # Frequency of changepoints

norm_ts, norm_cps =  @changepoint_sampler n λ Normal(Uniform(-2, 2), Uniform(0.0, 10.0))
plot(norm_ts, norm_cps)


pois_ts, pois_cps = @changepoint_sampler n λ Poisson(Uniform(1.0, 20.0))
plot(pois_ts, pois_cps)

########################
# Finding changepoints #
########################

β = 1.0 # Penalty

norm_seg_cost = NormalMeanVarSegment(norm_ts)
PELT(norm_seg_cost, n; pen=β)
@PELT norm_ts Normal(?, ?) β

pois_seg_cost = PoissonSegment(pois_ts)
PELT(pois_seg_cost, n; pen=β)
@PELT pois_ts Poisson(?) β
