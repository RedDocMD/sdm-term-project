module Altmin

include("models.jl")

import Flux
using Statistics: mean

struct Flatten
end

function linearize_tensor(tensor)
    dims = size(tensor)
    compressed_len = reduce(*, dims[1:ndims(tensor)-1]; init=1)
    reshape(tensor, (compressed_len, size(tensor, ndims(tensor))))
end

@Flux.functor Flatten
(m::Flatten)(x) = reshape(x, :, size(x, ndims(x)))

function compute_code_loss(codes, nmod, lin, loss_fn, codes_target, mu, lambda_c)
    output = lin(nmod(codes))
    loss = (1 / mu) * loss_fn(output) + Flux.mse(codes_target, output)
    if lambda_c > 0.0
        loss += (lambda_c / mu) * mean(abs.(codes))
    end
    return loss
end

function mark_code_mods(model)
    module_types = [Models.ConvMod, Models.DenseMod, Models.BatchNormMod, Models.LinMod]
    for mod in Iterators.flatten((model.features_vec, model.classifiers_vec))
        if any(t -> isa(mod, t), module_types)
            mod.has_codes = true
        end
    end
end

struct AltMinModel{C}
    model::C
    n_inputs::Union{Int32, Nothing}
end

function get_mods(model, optimizer, scheduler, data_parallel=False)
    mark_code_mods(model)

    model_mods::Vector{Any} = []
    nmod::Vector{Any} = []
    lmod::Vector{Any} = []

    for m in Iterators.flatten((model.features_vec, [Models.Flatten()], model.classifier_vec))
        if m.has_codes
            nmod = insert_mod(model_mods, nmod, false)
            push!(lmod, m)
        else
            lmod = insert_mod(model_mods, lmod, true)
            push!(nmod, m)
        end
    end

    insert_mod(model_mods, nmod, false)
    insert_mod(model_mods, lmod, true)

    id_codes = [i for (i, m) in Iterators.enumerate(model_mods) if m.has_codes]

    model_tmp = model_mods[1:id_codes[end - 1]]
    push!(model_tmp, model_mods[id_codes[end - 1]+1:end])
    model_tmp[end].has_codes = false
    model_mods = model_tmp

    for m in model_mods
        if m.has_codes
            m.optimizer = optimizer
            # Scheduler?
        end
        # Last layer?
    end

    # Data parallel?
    return AltMinModel(Flux.Chain(model_mods...), )
end

function insert_mod(model_mods, mods, has_codes)
    if length(mod) == 1
        push!(model_mods, mods[0])
        model_mods[end].has_codes = has_codes
    elseif length(mod) > 1
        append!(model_mods, mods)
        model_mods[end].has_codes = has_codes
    end
    return []
end

end