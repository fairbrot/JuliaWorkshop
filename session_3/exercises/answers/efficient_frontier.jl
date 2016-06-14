using JuMP

################################################
# Randomly generate data for Markowitz problem #
################################################

n = 10
A = rand(n,n)
Σ = A'A/sqrt(n)
μ = -1.0 * rand(n) + 2
τ = mean(μ)

#########################################
# Formulate and solve Markowitz problem #
#########################################

m = Model()
@variable(m, 0.0 <= x[i=1:n] <= 1.0)
@constraint(m, sum{x[i], i=1:n}==1.0)
@constraint(m, dot(μ, x) >= τ)
@objective(m, Min, dot(x,Σ*x))
print(m)

solve(m)

########################
# Reformulated problem #
########################

m = Model()
@variable(m, 0.0 <= x[i=1:n] <= 1.0)
@variable(m, τ == 0.0) # need '==' operator to ensure variable is fixed
@constraint(m, sum{x[i], i=1:n}==1.0)
@constraint(m, dot(μ, x) >= τ)
@objective(m, Min, dot(x,Σ*x))


##########################################
# Find efficient frontier with new model #
##########################################

exp_ret = Float64[]
var_ret = Float64[]
for t in linspace(0.95*minimum(μ), 0.95*maximum(μ), 20)
    setvalue(τ, t)
    solve(m);
    push!(exp_ret, dot(getvalue(x), μ));
    push!(var_ret, getobjectivevalue(m));
end
    
plot(y=exp_ret, x=var_ret, Geom.line,
     Guide.title("Efficient Frontier"),
     Guide.xlabel("Expected Return"),
     Guide.ylabel("Variance"))
