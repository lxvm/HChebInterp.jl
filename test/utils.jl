using HChebInterp

function hchebinterp_count(f, args...; kws...)
    numevals::Int = 0
    g = if f isa BatchFunction
        BatchFunction(f.x) do x
            n += length(x)
            return f.f(x)
        end
    else
        x -> (numevals += 1; f(x))
    end
    fun = hchebinterp(g, args...; kws...)
    return fun, numevals
end

function hchebinterp_count_batch(f::BatchFunction, args...; kws...)
    ncall::Int = 0
    g = BatchFunction(f.x, f.max_batch) do x
        ncall += 1
        return f.f(x)
    end
    fun = hchebinterp(g, args...; kws...)
    return fun, ncall
end

function leafboxes(p::HChebInterp.TreePoly)
    tree = zip(p.searchtree, p.funtree)
    boxes = [(fun.lb, fun.ub) for (children, fun) in tree if isempty(children)]
    return boxes
end
