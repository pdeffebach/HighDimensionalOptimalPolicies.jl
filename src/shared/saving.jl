######################################################################
# Tables.jl API ######################################################
######################################################################
import Tables.DictColumnTable
"""
    Tables.dictcolumntable(out::AbstractPolicyOutput; only_last_half = false)

Create a `Tables.DictColumnTable` from `out`. Convenient for saving
results to a `DataFrame` for further analysis, or for saving as a CSV.
See [`save_policy_output_csv`](@ref) for automated saving of policy outputs.

```julia
d = Tables.dictcolumntable(out)
DataFrame(d)
CSV.write(d, fname)
```

The resulting table has columns

* `iter`: A row identifier. When `only_last_half = true`, it will not
  represent
* `param1`, `param2`, etc: a column for each parameter in the policy vector
* `obj`: The objective value associated with that policy
"""
function Tables.dictcolumntable(out::AbstractPolicyOutput; only_last_half = true)
    policies = get_policy_vec(out; only_last_half)
    objs = get_objective_vec(out; only_last_half)
    N = length(policies)
    nms = vcat(
        :iter,
        [Symbol("param", i) for i in 1:length(first(policies))],
        :obj)

    policies_cols = [[p[i] for p in policies] for i in eachindex(first(policies))]

    schema = Tables.Schema(nms, nothing)

    # Any here is a hack to prevent auto-promotion in vcat
    values = vcat(Any[1:N], policies_cols, [objs])
    dict = OrderedCollections.OrderedDict(nms .=> values)

    Tables.DictColumnTable(schema, dict)
end

"""
    Tables.dictcolumntable(out::AbstractMultiPolicyOutput; only_last_half = true, only_max_invtemp = false)

Create a `Tables.DictColumnTable` from `out`. Convenient for saving
results to a `DataFrame` for further analysis, or for saving as a CSV.
See [`save_policy_output_csv`](@ref) for automated saving of policy outputs.

When `only_max_invtemp` is set to `true`, only saves the
highest inverse temperature, to avoid saving too much output.
When saving a `PigeonsSolverOutput`, `only_max_invtemp` is set
to true automatically, because inverse temperatures are not
deterministic across runs.

```julia
d = Tables.dictcolumntable(out)
DataFrame(d)
CSV.write(d, fname)
```

The resulting table has columns

* `invtemp`: The inverse temperature of that run
* `iter`: A row identifier. When `only_last_half = true`, it will not
  represent
* `param1`, `param2`, etc: a column for each parameter in the policy vector
* `obj`: The objective value associated with that policy
"""
function Tables.dictcolumntable(out::AbstractMultiPolicyOutput; only_last_half = true, only_max_invtemp = false)
    num_invtemps = get_num_invtemps(out)
    invtemps = get_invtemps(out)

    policies = get_policy_vec(out; ind = 1)
    objs = get_objective_vec(out; ind = 1)
    N = length(policies)
    invtemps_col = fill(invtemps[1], N)
    nms = vcat(
        :invtemp,
        :iter,
        [Symbol("param", i) for i in 1:length(first(policies))],
        :obj)

    policies_cols = [[p[i] for p in policies] for i in eachindex(first(policies))]

    itrs = collect(1:N)

    if only_max_invtemp == false
        for ind in 2:num_invtemps
            policies_t = get_policy_vec(out; ind)
            objs_t = get_objective_vec(out; ind)
            N_t = length(policies_t)
            invtemps_col_t = fill(invtemps[ind], N_t)
            itrs_t = collect(1:N_t)

            policies_cols_t = [[p[i] for p in policies_t] for i in eachindex(first(policies_t))]

            policies_cols = map(vcat, policies_cols, policies_cols_t)
            objs = vcat(objs, objs_t)
            itrs = vcat(itrs, itrs_t)
            invtemps_col = vcat(invtemps_col, invtemps_col_t)
        end
    end

    schema = Tables.Schema(nms, nothing)

    # Any here is a hack to prevent auto-promotion in vcat
    values = vcat(Any[invtemps_col], [itrs], policies_cols, [objs])
    dict = OrderedCollections.OrderedDict(nms .=> values)

    Tables.DictColumnTable(schema, dict)
end

######################################################################
# Saving results #####################################################
######################################################################

"""
    function save_policy_output_csv(
        out::AbstractMultiPolicyOutput;
        identifier = nothing,
        outdir::Union{Nothing, AbstractString} = nothing,
        only_max_invtemp::Bool = false,
        make_dir = false)

Saves the result of `out` to a `.csv` file with a randomly
generated string to ensure no collisions between filenames.

See [`MultiCSVPolicyOutput`] for reading the outputs of `save_policy_output_csv`
into a new policy output.

!!! warning
    `save_policy_output_csv` does not validate inputs on writing, and
    `MultiCSVPolicyOutput` does not validate inputs on reading. It is
    up to the user to ensure that all inputs to the solver are *the
    exact same* for all `.csv` files saved.

## Keyword arguments

* `identifer`: A string which is appended to the saved file name to help
  the user keep track of different files.
* `outdir`: The output directory
* `only_max_invtemp`: Whether to only save the highest inverse
  termperature.
* `make_dir`: Whether to create `outdir` if it does not exist.
"""
function save_policy_output_csv(
    out::AbstractMultiPolicyOutput;
    identifier = nothing,
    outdir::Union{Nothing, AbstractString} = nothing,
    only_max_invtemp::Bool = false,
    make_dir = false)

    if isnothing(outdir)
        err = "No outdir provided. Please supply keyword argument outdir"
        throw(ArgumentError(err))
    end
    if make_dir == true && isdir(outdir) == false
        mkdir(outdir)
    end
    if out isa PigeonsSolverOutput
        @warn "Only saving highest inverse temperature because PigeonsSolverOutput does not have deterministic temperatures."
        only_max_invtemp = true
    end

    type_name = string(nameof(typeof(out)))

    d = Tables.dictcolumntable(out; only_max_invtemp)

    fname = if isnothing(identifier)
        type_name * "_" * randstring(12) * ".csv"
    else
        identifier * "_" * type_name * "_" * randstring(12) * ".csv"
    end

    outpath = joinpath(outdir, fname)

    CSV.write(outpath, d)
end

######################################################################
# Reading results ####################################################
######################################################################
struct CSVPolicyOutput{VP, VO, I} <: AbstractPolicyOutput
    params::VP
    objs::VO
    invtemp::I
end

function Base.show(io::IO, ::MIME"text/plain", t::CSVPolicyOutput)
    s = """
    CSVPolicyOutput
        Inverse Temperature: $(t.invtemp)
        Final objective value: $(last(t.objs))
        Sample length: $(length(t.objs))
    """
    print(io, s)
end

function Base.show(io::IO, t::CSVPolicyOutput)
    s = "CSVPolicyOutput (invtemp = $(t.invtemp))"
    print(io, s)
end

struct MultiCSVPolicyOutput{V, I} <: AbstractMultiPolicyOutput
    v::V
    invtemps::I
end

function Base.show(io::IO, t::MultiCSVPolicyOutput)
    max_invtemp = first(t.invtemps)
    num_temps = length(t.v)
    s = """
    MultiCSVPolicyOutput with $num_temps temperatures and maximum inverse temperature $max_invtemp
    """
    print(io, s)
end

"""
    MultiCSVPolicyOutput(outdir::AbstractString)

Construct a simplified output type by reading and concatenating all
`.csv` files in a directory.

!!! warning
    `save_policy_output_csv` does not validate inputs on writing, and
    `MultiCSVPolicyOutput` does not validate inputs on reading. It is
    up to the user to ensure that all inputs to the solver are *the
    exact same* for all `.csv` files saved.

The resulting ouput, a `MultiCSVPolicyOutput` type, follows the
`AbstractPolicyOutput` API, meaning the functions

* `get_policy_vec`
* `get_objective_vec`
* `get_average_policy`
* `get_last_policy`
* `get_invtemps`

However you will *not* have access to the underlying functions, i.e.
the `initfun`, `objfun`, `nextfun` functions used as an input to
the solver.

When reading `.csv` files, assumes that all `.csv` files are generated
by `save_policy_output_csv` from policy outputs using the *same* model input.
However does not validate these inputs.
"""
function MultiCSVPolicyOutput(outdir::AbstractString)
    files = readdir(outdir, join = true)
    if any(file -> endswith(file, ".csv") == false, files)
         error("Non-csv file encountered in directory $outdir")
    end

    all_row_iterator = CSV.File(files)
    g = SplitApplyCombine.group(row -> row.invtemp, all_row_iterator)
    row_keys = keys(first(first(g)))
    param_names = row_keys[3:(end-1)]
    invtemps = collect(keys(g))
    v = map(collect(keys(g))) do k
        data = g[k]
        objs = map(row -> row.obj, data)
        params = map(data) do row
            collect(row[p] for p in param_names)
        end
        CSVPolicyOutput(params, objs, k)
    end

   MultiCSVPolicyOutput(v, invtemps)
end

function get_policy_vec(out::CSVPolicyOutput; only_last_half = false)
    if only_last_half == true
        last_half(out.params)
    else
        out.params
    end
end

function get_average_policy(out::CSVPolicyOutput; only_last_half = false)
    mean(get_policy_vec(out; only_last_half))
end

function get_last_policy(out::CSVPolicyOutput)
    last(get_policy_vec(out; last_half = false))
end

function get_objective_vec(out::CSVPolicyOutput; only_last_half = false)
    if only_last_half == true
        last_half(out.objs)
    else
        out.objs
    end
end

function get_invtemps(t::MultiCSVPolicyOutput)
    t.invtemps
end
