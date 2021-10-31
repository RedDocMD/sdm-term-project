module Models

import Flux
using Flux: cpu, gpu
using Printf

struct LinMod{C}
    classifier::C
end

function LinMod(n_inputs, n_outputs; bias=false, batchnorm=false)
    layers::Vector{Any} = [Flux.Dense(n_inputs, n_outputs, bias=bias)]
    if batchnorm
        push!(layers, Flux.BatchNorm(Int(n_outputs)))
    end
    return LinMod(Flux.Chain(layers...) |> gpu)
end

Flux.@functor LinMod
(m::LinMod)(x) = m.classifier(x)

struct FFNet{C}
    features_vec::Vector{Any}
    classifier_vec::Vector{Any}
    classifier::C
end

function FFNet(;n_inputs, n_hiddens, n_hidden_layers=Int32(2), n_outputs=Int32(10), nlin=Flux.relu, bias=false, batchnorm=false)
    nlin_arr(x) = nlin.(x)
    layers::Vector{Any} = [LinMod(n_inputs, n_hiddens, bias=bias, batchnorm=batchnorm), nlin_arr]
    for i = 1:n_hidden_layers
        push!(layers, LinMod(n_hiddens, n_hiddens, bias=bias, batchnorm=batchnorm), nlin_arr)
    end
    push!(layers, Flux.Dense(n_hiddens, n_outputs))
    return FFNet([], layers, Flux.Chain(layers...) |> gpu)
end

Flux.@functor FFNet
(m::FFNet)(x) = m.classifier(x)

struct LeNet{F, C}
    features_vec::Vector{Any}
    classifier_vec::Vector{Any}
    features::F
    classifier::C
end

function LeNet(;num_input_channels=3, num_classes=10, window_size=32, bias=true)
    relu(x) = Flux.relu.(x)

    features_vec::Vector{Any} = [
        Flux.Conv((5, 5), num_input_channels => 6; bias=bias),
        relu,
        Flux.MaxPool((2, 2)),

        Flux.Conv((5, 5), 6 => 16, bias=bias),
        relu,
        Flux.MaxPool((2, 2))
    ]
    features = Flux.Chain(features_vec...) |> gpu

    inp_size = 16 * (((window_size - 4) ÷ 2 - 4) ÷ 2) ^ 2
    classifier_vec::Vector{Any} = [
        Flux.Dense(inp_size, 120, bias=bias),
        relu,
        Flux.Dense(120, 84, bias=bias),
        relu,
        Flux.Dense(84, num_classes, bias=bias)
    ]
    classifier = Flux.Chain(classifier_vec...) |> gpu

    return LeNet(features_vec, classifier_vec, features, classifier)
end

function linearize_tensor(tensor)
    dims = size(tensor)
    compressed_len = reduce(*, dims[1:ndims(tensor)-1]; init=1)
    reshape(tensor, (compressed_len, size(tensor, ndims(tensor))))
end

Flux.@functor LeNet
function (m::LeNet)(x) 
    interm = m.features(x)
    interm = linearize_tensor(interm)
    return m.classifier(interm)
end

function test(model, data_loader; criterion=Flux.Losses.logitcrossentropy, label="")
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
    if length(label) > 0
        loss_str = @sprintf "%.4f" avg_test_loss
        acc_str = @sprintf "%.2f" 100.0 * accuracy
        println("$label: Average loss: $loss_str, Accuracy: $correct/$no_datapoints ($acc_str%)")
    end
    return accuracy
end

end