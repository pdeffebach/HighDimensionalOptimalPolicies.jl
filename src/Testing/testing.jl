function test_objfun(x)
    x1 = x[1]
    x2 = x[2]

    #res = (1 - x1)^2 + 500 * (x2 - x1^2)^2
    res = (x1 - 1)^2 + (x2 - 1)^2
    -res
end

function plot_test_objfun(objfun, β)
    xs = range(-2, 2; length = 200)
    ys = range(-1, 3; length = 200)
    Makie.heatmap(xs, ys, (x, y) -> exp(β * objfun((x, y))))
end

function test_initfun(rng)
    σ = 0.1
    [rand(Normal(1, 1)) for i in 1:2]
end

function test_nextfun(rng, x)
    σ = 0.1
    [rand(Normal(xi, σ)) for xi in x]
end

function test_MCMC()
    get_best_policy(
        MCMCSolver();
        initfun = test_initfun,
        objfun = test_objfun,
        nextfun = test_nextfun,
        β = 20.0)
end

function test_Pigeons()
    get_best_policy(
        PigeonsSolver();
        initfun = test_initfun,
        objfun = test_objfun,
        nextfun = test_nextfun,
        β = 20.0)
end

