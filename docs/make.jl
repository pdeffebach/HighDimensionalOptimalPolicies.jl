using Documenter, DocumenterCitations

using HighDimensionalOptimalPolicies

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

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
    plugins = [bib]
)

deploydocs(
    repo = "github.com/pdeffebach/HighDimensionalOptimalPolicies.jl.git",
    target = "build",
    deps = nothing,
    make = nothing)