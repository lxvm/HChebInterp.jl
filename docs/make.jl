push!(LOAD_PATH, "../src")
using Documenter, HChebInterp

Documenter.HTML(
    mathengine = MathJax3(Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"],
        ),
    )),
)

makedocs(
    sitename="HChebInterp.jl",
    modules=[HChebInterp],
    pages = [
        "Home" => "index.md",
        "Manual" => "methods.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/HChebInterp.jl.git",
)