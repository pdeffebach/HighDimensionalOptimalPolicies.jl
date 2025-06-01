using Documenter, DocumenterCitations

using HighDimensionalOptimalPolicies

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

######################################################################
# Set the draft keyword argument #####################################
######################################################################
# Set this to false while editing. To actually see all output,
# set to true.
draft = false

makedocs(;
    sitename="High Dimensional Optimal Policies",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = Any[
        "Introduction" => "index.md",
        "Tutorial" => "tutorial.md",
        "Mathematical Appendix" => "math.md",
        "Optimal Transport Example" => "optimal_transport.md",
        "References" => "references.md",
    ],
    plugins = [bib],
    draft = draft
)

deploydocs(
    repo = "github.com/pdeffebach/HighDimensionalOptimalPolicies.jl.git",
    target = "build",
    deps = nothing,
    make = nothing)