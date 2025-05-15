using Markdown

function include_md(fname, blockname; write_file = false)
    s = String(read(fname))
    docs = Markdown.parse(s)
    setup_regex = Regex("^@setup $blockname")
    example_regex = Regex("^@example $blockname")
    to_eval = Any[]
    for c in docs.content
        if c isa Markdown.Code
            if occursin(example_regex, c.language) || occursin(setup_regex, c.language)
                push!(to_eval, c.code)
            end
        end
    end
    code_str = join(to_eval, "\n\n")
    if write_file == false
        include_string(Main, code_str)
    else
        write("test.jl", code_str)
    end
end