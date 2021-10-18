using Flux
using ArgParse

include("utils/utils.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--model"
            help = "name of model: \"feedforward\", \"binary\" or \"LeNet\""
            arg_type = String
            default = "feedforward"
        "--n-hidden-layers"
            help = "number of hidden layers (ignored for LeNet)"
            arg_type = Int32
            default = Int32(2)
        "--n-hiddens"
            help = "number of hidden units (ignored for LeNet)"
            arg_type = Int32
            default = Int32(100)
        "--dataset"
            help = "name of dataset"
            arg_type = String
            default = "mnist"
        "--data-augmentation"
            help = "enables data augmentation"
            action = :store_true
        "--batch-size"
            help = "input batch size for training"
            arg_type = Int32
            default = Int32(200)
        "--epochs"
            help = "number of epochs to train for"
            arg_type = Int32
            default = Int32(50)
        "--lr"
            help = "learning rate"
            arg_type = Float32
            default = Float32(0.003)
        "--lr-decay"
            help = "learning rate decay factor per epoch"
            arg_type = Float32
            default = Float32(1.0)
        "--no-batchnorm"
            help = "disables batch-normalisation"
            action = :store_true
        "--seed"
            help = "random seed"
            arg_type = Int32
            default = Int32(1)
        "--log-interval"
            help = "how many batches to wait before logging training status"
            arg_type = Int32
            default = Int32(100)
        "--save-interval"
            help = "how many batches to wait before saving test performance (if set to zero, it does not save)"
            arg_type = Int32
            default = Int32(1000)
        "--log-first-epoch"
            help = "whether or not it should test and log after every mini-batch in first epoch"
            action = :store_true
        "--no-cuda"
            help = "disables CUDA training"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    device, num_gpus = Utils.get_devices(0)
    num_workers = num_gpus > 0 ? num_gpus : 1

    model_name = lowercase(args["model"])
    if model_name == "feedforward" || model_name == "binary"
        model_name = "$(model_name)_$(args["n-hidden-layers"])x$(args["n-hiddens"])"
    end
    file_name = "save_adam_baseline_$(model_name)_$(args["dataset"])_$(args["seed"]).pt"

    println("\nSGD Baseline")
    println("* Loading dataset: $(args["dataset"])")
    println("* Loading model: $(model_name)")
    println("      BatchNorm: $(!args["no-batchnorm"])")

    if lowercase(args["model"]) == "feedforward"
        @time trainloader, testloader = Utils.load_dataset(args["dataset"], args["batch-size"], false)
    end
end

main()