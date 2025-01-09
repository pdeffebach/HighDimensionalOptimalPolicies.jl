using Documenter, HighDimensionalOptimalPolicies

makedocs(;
    sitename="High Dimensional Optimal Policies",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = Any[
        "Introduction" => "index.md",
        "Mathematical Appendix" => "math.md",
    ]
)