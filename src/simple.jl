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