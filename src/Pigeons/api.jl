struct PigeonsSolver <: AbstractPolicySolver end

function get_best_policy(::PigeonsSolver; initfun, objfun, nextfun, β, kwargs...)
    H = HighDimensionalProblem(initfun, objfun, nextfun)

    pt = pigeons(
        target = OuterHighDimensionalProblem(β, H),
        reference = OuterHighDimensionalProblem(0.0, H),
        record = [traces],
        n_chains = 10,
        n_rounds = 10,
        explorer = H,
        multithreaded = false,
        show_report = false)

    out_arr = sample_array(pt)
    out_arr[end, 1:(end-1), 1]
end
