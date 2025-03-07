"""
    MetropolitanHastingsProblem{Fobj, Fnext}


"""
struct MetropolitanHastingsProblem{Fobj, Fnext}
    obj_fun::Fobj
    next_policy::Fnext
    β::Float64
end

function accept_candidate(W_curr, W_cand, β)
    β * (W_cand - W_curr) > log(rand())
end

function solve(x_init, mhp::MetropolitanHastingsProblem; max_itr = 1000)
    itr = 1
    (;obj_fun, next_policy, β) = mhp
    W_curr = obj_fun(x_init)
    x_curr = x_init
    while true
        x_cand = next_policy(x_curr)
        W_cand = obj_fun(x_cand)
        @show accept = accept_candidate(W_curr, W_cand, β)
        if accept
            x_curr = x_cand
            W_curr = W_cand
            @show x_cand
        end

        if itr == max_itr
            break
        end
        itr = itr + 1
    end

    return x_curr
end

function mcmc(initfun, objfun, nextfun, β)
    rng = Random.default_rng()
    x0  = initfun(rng)
    y0 = objfun(x0)
    max_itr = 10000
    itr = 1
    while true
        x1 = nextfun(rng, x0)
        y1 = objfun(x1)
        α = exp(β * (y1 - y0))
        if α > 1
            x0 = x1
            y0 = y1
        elseif rand(rng) < α
            x0 = x1
            y0 = y1
        else
            # Do nothing
        end

        if itr > max_itr
            break
        end
        itr = itr + 1
    end
    return x0
end

function mcmc_outer(initfun, objfun, nextfun, β)
    #βs = range(0, 1, length = 10) .* β
    tasks = map(1:5) do _
        Threads.@spawn mcmc(initfun, objfun, nextfun, β)
    end
    cands = fetch.(tasks)
    (obj, ind) = findmax(objfun, cands)
    cands[ind]
end
