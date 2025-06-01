function quickstart()
    n_edges = 100
    n_edges_to_upgrade = 50
    network_values = sort(rand(LogNormal(1.0), n_edges), rev = true)
    network_values = network_values ./ sum(network_values)

    initfun = let n_edges = n_edges, n_edges_to_upgrade = n_edges_to_upgrade
        rng -> begin
            fill(false, n_edges_to_upgrade)
            inds = sample(rng, 1:n_edges, n_edges_to_upgrade; replace = false)
            p = fill(false, n_edges)
            p[inds] .= true
            p
        end
    end

    nextfun =  let n_edges = n_edges, n_edges_to_upgrade = n_edges_to_upgrade
        (rng, state) -> begin
            upgraded_edges = findall(state)
            not_upgraded_edges = findall(==(false), state)

            edge_to_drop = sample(rng, upgraded_edges)
            edge_to_add = sample(rng, not_upgraded_edges)

            new_edges_to_upgrade = copy(state)
            new_edges_to_upgrade[edge_to_drop] = false
            new_edges_to_upgrade[edge_to_add] = true

            new_edges_to_upgrade
        end
    end

    objfun = let network_values = network_values
        state -> begin
            sum(state .* network_values)
        end
    end

    (; initfun, nextfun, objfun)
end