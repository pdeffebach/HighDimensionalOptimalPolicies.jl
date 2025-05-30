######################################################################
# Shared type definitions ############################################
######################################################################

abstract type AbstractPolicySolver end
# TODO: Rename AbstractMultiPolicyOutput to AbstractPolicyOutput
abstract type AbstractPolicyOutput end
abstract type AbstractMultiPolicyOutput end

"""
    GenericSolverInput{Finit, Fnext, Fobj, V<:AbstractVector{<:Real}}

Stores basic information to conduct the Parallel Tempering algorithm.

Does not store all implementation details, such as the number of
iterations, swaps, etc.

$FIELDS
"""
struct GenericSolverInput{Finit, Fnext, Fobj, V<:AbstractVector{<:Real}}
    """
    `initfun(rng)` must return a random initial guess from the
    state space.
    """
    initfun::Finit
    """
    `nextfun(rng, state)` must return a new guess conditional on
    the current state.
    """
    nextfun::Fnext
    """
    `objfun(state)` must return the welfare value of the current
    state. Higher values indicate higher welfare.
    """
    objfun::Fobj
    """
    The vector representing the inverse temperatures used in the
    algorithm.
    """
    invtemps::V
end

######################################################################
# Shared function definitions ########################################
######################################################################

"""
    $(SIGNATURES)

Returns the last half of an array.
"""
function last_half(x::AbstractVector)
    middle_ind = floor(Int, length(x) / 2)
    x[middle_ind:end]
end

"""
    $SIGNATURES

Return a vector of length `length` of inverse temperatures with the
maximum inverse temperature in the first position and 0 in the last
position. `invtemps_curvature` controls how the inverse temperature
declines towards 0 according to.


```math
\\text{Inverse temperature}_{i} &= \\frac{i}{N}^{invtemps_curvature}
```

With \$N\$ representing the length of the array and \$i\$ ranging from
\$1\$ to \$0\$ in equally spaced increments.

* A value close to `0` means inverse temperatures are clustered close to
  the maximum inverse temperature.
* A value of `1` means equally spaced temperatures.
* A value greater than `1` means inverse temperatures are clustered
  close to `0`.

I recommend a value greater than `1` to ensure a high degree of
mixing.
"""
function make_invtemps(max_invtemp::Real; length::Integer, invtemps_curvature::Real)
    @assert invtemps_curvature > 0
    invtemps = (range(1, 0, length = length) .^ invtemps_curvature) .* max_invtemp
end


function test_mixing(obj_vec::Vector{<:Real}, invtemp::Real, log_n::Real; K::Union{Integer, Nothing} = nothing)
    if !isnothing(K)
        obj_vec = StatsBase.sample(obj_vec, K)
    end
    obj_max = maximum(obj_vec)
    obj_mean = mean(obj_vec)
    obj_std = std(obj_vec)

    T̂ = obj_max - obj_mean - (log_n / invtemp)
    z = T̂ / obj_std
#    p = cdf(Normal(0, 1), z)
end

"""
    test_mixing(out::AbstractPolicyOutput, log_n; K = nothing)

Test for the level of mixing in a collection of policies.

## Arguments

* `out`: An output
* `log_n`: The log of the size of the state space.
* `K`: Optionally analyze a subset of the samples.
"""
function test_mixing(out::AbstractPolicyOutput, log_n::Real; K::Union{Integer, Nothing} = nothing)
    invtemp = out.invtemp
    obj_vec = get_objective_vec(out; only_last_half = true)
    test_mixing(obj_vec, invtemp, log_n; K)
end

"""
    $SIGNATURES

Return the vector of policies for inverse temperature at index `ind`,
sorted highest inverse temperature first.

If `only_last_half` is set to `true`, then the first half of the
draws, i.e. the burn-in period, is ignored.
"""
function get_policy_vec(out::AbstractMultiPolicyOutput; ind = 1, only_last_half = true)
    get_policy_vec(out.v[ind]; only_last_half)
end

"""
    $SIGNATURES

Return the average policy for inverse temperature at index `ind`,
sorted highest inverse temperature first.

If `only_last_half` is set to `true`, then the first half of the
draws, i.e. the burn-in period, is ignored.
"""
function get_average_policy(out::AbstractMultiPolicyOutput; ind = 1, only_last_half = true)
    get_average_policy(out.v[ind]; only_last_half)
end

"""
    $SIGNATURES

Return the last policy for inverse temperature at index `ind`, sorted
highest inverse temperature first.
"""
function get_last_policy(out::AbstractMultiPolicyOutput; ind = 1)
    get_last_policy(out.v[ind])
end

"""
    $SIGNATURES

Return the vector of objectives at index `ind`, sorted highest inverse
temperature first.
"""
function get_objective_vec(out::AbstractMultiPolicyOutput; ind = 1, only_last_half = true)
    get_objective_vec(out.v[ind]; only_last_half)
end

"""
    test_mixing(out::AbstractMultiPolicyOutput, log_n; K = nothing, ind = 1)

Test the mixing for the inverse temperature at index `ind`, sorted
highest inverse temperature first.
"""
function test_mixing(out::AbstractMultiPolicyOutput, log_n; K = nothing, ind = 1)
    test_mixing(out.v[ind], log_n; K)
end

function get_num_invtemps(out::AbstractMultiPolicyOutput)
    length(get_invtemps(out))
end