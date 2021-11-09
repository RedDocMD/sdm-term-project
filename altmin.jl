
include("models.jl")

module Altmin

import Flux
using Statistics: mean
using Flux: gpu
import Main.Models

function compute_code_loss(codes, nmod, lin, loss_fn, codes_target, mu, lambda_c)
    output = lin(nmod(codes))
    loss = (1 / mu) * loss_fn(output) + Flux.mse(codes_target, output)
    if lambda_c > 0.0
        loss += (lambda_c / mu) * mean(abs.(codes))
    end
    return loss
end

function mark_code_mods!(model)
    module_types = [Models.ConvMod, Models.DenseMod, Models.BatchNormMod, Models.LinMod]
    for mod in Iterators.flatten((model.features_vec, [Models.Flatten()], model.classifier_vec))
        if any(t -> mod isa t, module_types)
            mod.has_codes = true
        end
    end
end

struct AltMinModel{C}
    model_mods::Vector{Any}
    model::C
    n_inputs::Union{Int32, Nothing}
end

Flux.@functor AltMinModel
(m::AltMinModel)(x) = m.model(x)

function get_mods(model, optimizer, scheduler)
    mark_code_mods!(model)

    model_mods::Vector{Any} = []
    nmod::Vector{Any} = []
    lmod::Vector{Any} = []

    for m in Iterators.flatten((model.features_vec, [Models.Flatten()], model.classifier_vec))
        if m.has_codes !== nothing && m.has_codes
            nmod = insert_mod!(model_mods, nmod, false)
            push!(lmod, m |> gpu)
        else
            lmod = insert_mod!(model_mods, lmod, true)
            push!(nmod, m |> gpu)
        end
    end

    insert_mod!(model_mods, nmod, false)
    insert_mod!(model_mods, lmod, true)

    id_codes = [i for (i, m) in Iterators.enumerate(model_mods) 
                  if m.has_codes !== nothing && m.has_codes]

    model_tmp = model_mods[1:id_codes[end - 1]]
    push!(model_tmp, Models.Chain(model_mods[id_codes[end - 1]+1:end]))
    model_tmp[end].has_codes = false
    model_mods = model_tmp

    for m in model_mods
        if m.has_codes !== nothing && m.has_codes
            m.optimizer = optimizer
            m.scheduler = scheduler
        end
    end
    model_mods[end].optimizer = optimizer
    model_mods[end].scheduler = scheduler

    # Data parallel?

    if :n_inputs ∈ fieldnames(typeof(model))
        n_inputs = (model.n_inputs)::Union{Int32, Nothing}
    else
        n_inputs = nothing::Union{Int32, Nothing}
    end
    return AltMinModel(model_mods, Flux.Chain(model_mods...) |> gpu, n_inputs)
end

function insert_mod!(model_mods, mods, has_codes)
    if length(mods) == 1
        push!(model_mods, mods[1] |> gpu)
        model_mods[end].has_codes = has_codes
    elseif length(mods) > 1
        push!(model_mods, Models.Chain(mods) |> gpu)
        model_mods[end].has_codes = has_codes
    end
    return []
end

function get_codes(model, inputs)
    if model.n_inputs !== nothing
        x = reshape(inputs, Int(model.n_inputs), :)
    else
        x = inputs
    end

    codes = []
    for m in model.model_mods
        x = m(x)
        if m.has_codes !== nothing && m.has_codes
            push!(codes, copy(x))
        end
    end
    return x, codes
end

end