module Models

import Flux
using Flux: cpu

function LinMod(n_inputs, n_outputs; bias=false, batchnorm=false)
    layers::Vector{Any} = [Flux.Dense(n_inputs, n_outputs, bias=bias)]
    if batchnorm
        push!(layers, Flux.BatchNorm(Int(n_outputs)))
    end
    return Flux.Chain(layers...)
end

function FFNet(n_inputs, n_hiddens; n_hidden_layers=Int32(2), n_outputs=Int32(10), nlin=Flux.relu, bias=false, batchnorm=false)
    nlin_arr(x) = nlin.(x)
    layers = [LinMod(n_inputs, n_hiddens, bias=bias, batchnorm=batchnorm), nlin_arr]
    for i = 1:n_hidden_layers
        push!(layers, LinMod(n_hiddens, n_hiddens, bias=bias, batchnorm=batchnorm), nlin_arr)
    end
    push!(layers, Flux.Dense(n_hiddens, n_outputs))
    return Flux.Chain(layers...)
end

function test(model, data_loader, criterion=Flux.Losses.logitcrossentropy, label="")
    test_loss, correct = 0.0, 0
    no_mini_batches = 0
    no_datapoints = 0
    for (data, targ) in data_loader
        output = model(data)
        # Check if output is a tuple or not?
        test_loss += criterion(output, targ)
        correct += count(((x, y),) -> x == y, zip(argmax(targ |> cpu, dims=1), argmax(output |> cpu, dims=1)))
        no_mini_batches += 1
        no_datapoints += size(data, ndims(data))
    end
    avg_test_loss = test_loss / no_mini_batches
    accuracy = Float32(correct) / no_datapoints
    return avg_test_loss, accuracy
end

end