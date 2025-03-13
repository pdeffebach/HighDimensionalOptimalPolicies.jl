struct PigeonsSolver <: AbstractPolicySolver end

struct PigeonsSolverOutput{I, PT} <: AbstractPolicyOutput
    input::I
    pt::PT
end

using Infiltrator

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

    PigeonsSolverOutput(OuterHighDimensionalProblem(β, H), pt)
end

function get_parameters_vec(out::PigeonsSolverOutput)

    init_guess = out.input.problem.initfun(Random.default_rng())
    init_type = typeof(init_guess)

    chn = Chains(out.pt)
    nt_params = get(chn, section = :parameters)
    num_itrs = length(first(nt_params))
    num_cols = length(nt_params)
    map(1:num_itrs) do i
        init_type([nt_params[j][i] for j in 1:num_cols])
    end
end

function get_average_policy(out::PigeonsSolverOutput)
    #chn = Chains(out.pt)
    #describe(chn)[1][:, :mean]
    mean(get_parameters_vec(out))
end

function get_last_policy(out::PigeonsSolverOutput)
    last(get_parameters_vec(out))
end

function get_objective_vec(out::PigeonsSolverOutput)
    chn = Chains(out.pt)
    nt_params = get(chn, section = :internals)
    num_itrs = length(first(nt_params))
    log_densities = map(1:num_itrs) do i
        nt_params[:log_density][i]
    end
    log_densities ./ out.input.ξ
end

function test_mixing(out::PigeonsSolverOutput, log_n̄)
    ξ = out.input.ξ
    obj_vec = get_objective_vec(out)
    K = length(obj_vec)
    obj_max = maximum(obj_vec)
    obj_mean = mean(obj_vec)
    obj_std = std(obj_vec)

    T̂ = obj_max - obj_mean - (log_n̄ / ξ)
    z = T̂ / obj_std
    p = cdf(Normal(0, 1), z)
    @infiltrate
end