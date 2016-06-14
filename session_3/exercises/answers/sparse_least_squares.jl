using JuMP
using MultivariateStats
using Gurobi


######################
# Problem generation #
######################

function generate_problem(p::Int, n::Int, m::Int)
    raw_β = rand(n)
    sort_β = sort(raw_β)
    ord_p = sort_β[n-p]
    β = map(x-> (x>ord_p? x:0.0), raw_β)
    X = rand(m,n)
    y = X*β + 0.5*rand(m)
    return β, X, y
end

p, n, m = 30, 60, 300
β, X, y = generate_problem(p,n,m)

"""
# Description
Solves sparse least-squares problem.

min_β || y - Xβ ||^2
s.t.
||β||_0 ⫹ p

# Arguments
`X::Matrix{Float64}`: m x n design matrix
`y::Vector{Float64}`: output observations
`p::Int`: Maximum number of non-negative coeffients

# Returns
`(β::Vector{Float64}, res::Vector{Float64})`: model coefficients and residuals vectors
"""
function solve_sparse_least_squares(X::Matrix{Float64}, y::Vector{Float64}, p::Int)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    m, n = size(X)
    @variable(model, β[i=1:n])
    @variable(model, z[i=1:n], Bin)
    @objective(model, Min, dot(X*β, X*β) - 2 * dot(y, X*β))
    @constraint(model, use_var_1[i=1:n], -1000*z[i] <= β[i])
    @constraint(model, use_var_2[i=1:n], 1000*z[i] >= β[i])
    @constraint(model, sum(z) <= p)
    status = solve(model)
    if status != :Optimal
        error("Failed to find optimal solution")
    end

    β_val = getvalue(β)
    res = y - X*β_val
    return (β_val, res)
end



solve_sparse_least_squares(X, y, p);

###################
# With warm start #
###################

"""
Takes a solution from the least squares problem
and rounds down least significant coefficients
to create a feasible solution for the sparse least
squares problem.
"""
function round_solution(β::Vector{Float64}, p::Int)
    sort_β = sort(abs(β))
    ord_p = sort_β[n-p]
    new_β = map(x-> (abs(x)>ord_p? x:0.0), β)
    new_z = Float64[x!=0.0? 1.0: 0.0 for x in new_β]
    return new_β, new_z
end

function solve_sparse_least_squares_with_warm_start(X::Matrix{Float64}, y::Vector{Float64}, p::Int)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    m, n = size(X)
    @variable(model, β[i=1:n])
    @variable(model, z[i=1:n], Bin)
    @objective(model, Min, dot(X*β, X*β) - 2 * dot(y, X*β))
    @constraint(model, use_var_1[i=1:n], -1000*z[i] <= β[i])
    @constraint(model, use_var_2[i=1:n], 1000*z[i] >= β[i])
    @constraint(model, sum(z) <= p)

    relax_β = llsq(X, y; bias=false)
    init_β, init_z = round_solution(relax_β, p)

    for i in 1:n
        setvalue(β[i], init_β[i])
        setvalue(z[i], init_z[i])
    end
    
    status = solve(model)
    
    if status != :Optimal
        error("Failed to find optimal solution")
    end

    β_val = getvalue(β)
    res = y - X*β_val
    return (β_val, res)
end

@time solve_sparse_least_squares(X, y, p);
@time solve_sparse_least_squares_with_warm_start(X, y, p);
