using JuMP

function rand_weight_mat(n::Int)
    x = rand(n,n)
    for i in 1:n, j in 1:i
        x[j,i] = x[i,j]
    end
    return x
end

function solve_kpp(W::Matrix{Float64}, k::Int)
    n = size(W,1)
    model = Model()
    @variable(model, x[1:n, 1:k], Bin)
    @variable(model, y[i=1:n, j=1:i-1], Bin)
    @constraint(model, assign[i=1:n], sum{x[i,c], c=1:k} == 1.0)
    @constraint(model, tri1[i=1:n, j=1:i-1, c=1:k], y[i,j] >= x[i,c] + x[j,c] - 1.0)
    @constraint(model, tri2[i=1:n, j=1:i-1, c=1:k], x[i,c] >= y[i,j] + x[j,c] - 1.0)
    @constraint(model, tri3[i=1:n, j=1:i-1, c=1:k], x[j,c] >= x[i,c] + y[i,j] - 1.0)
    @objective(model, Min, sum{W[i,j]*y[i,j], i=1:n, j=1:i-1})
    result = solve(model)
    return getvalue(x), getobjectivevalue(model)
end

n = 10
W = rand_weight_mat(n)
k = 3
solve_kpp(W,k)

function solve_kpp_cts_relax(W::Matrix{Float64}, k::Int)
    n = size(W,1)
    model = Model()
    @variable(model, 0.0 <= x[1:n, 1:k] <= 1.0)
    @variable(model,0.0 <= y[i=1:n, j=1:i-1] <= 1.0)
    @constraint(model, assign[i=1:n], sum{x[i,c], c=1:k} == 1.0)
    @constraint(model, tri1[i=1:n, j=1:i-1, c=1:k], y[i,j] >= x[i,c] + x[j,c] - 1.0)
    @constraint(model, tri2[i=1:n, j=1:i-1, c=1:k], x[i,c] >= y[i,j] + x[j,c] - 1.0)
    @constraint(model, tri3[i=1:n, j=1:i-1, c=1:k], x[j,c] >= x[i,c] + y[i,j] - 1.0)
    @objective(model, Min, sum{W[i,j]*y[i,j], i=1:n, j=1:i-1})
    result = solve(model)
    return result, getvalue(x), getobjectivevalue(model)
end

# Note that relaxation always gives the trivial lower bound when we use positive weights
solve_kpp_cts_relax(W, 3)

function solve_kpp_sdp_relax(W::Matrix{Float64}, k::Int)
    n = size(W,1)
    model = Model()
    @variable(model, 0.0 <= Y[1:n,1:n] <= 1.0, Symmetric)
    @SDconstraint(model, k*Y - eye(n,n) >= 0)
    @SDconstraint(model, Y >= 0)
    @constraint(model, diags[i=1:n], Y[i,i] == 1.0)
    @objective(model, Min, sum{W[i,j]*Y[i,j], i=1:n, j=1:i-1})
    result = solve(model)
    return result, getobjectivevalue(model)
end

solve_kpp_sdp_relax(W, k)
