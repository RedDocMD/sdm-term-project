using Flux
using ArgParse

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
        "--n-iter-codes"
            help = "number of internal iterations for codes optimization"
        arg_type = Int32
            default = Int32(5)
        "--n-iter-weights"
            help = "number of internal iterations for learning weights"
            arg_type = Int32
            default = Int32(1)
        "--lr-codes"
            help = "learning rate for codes update"
            arg_type = Float32
            default = Float32(0.3)
        "--lr-out"
            help = "learning rate for last layer weights updates"
            arg_type = Float32
            default = Float32(0.008)
        "--lr-weights"
            help = "learning rate for hidden weights updates"
            arg_type = Float32
            default = Float32(0.008)
        "--lr-half-epochs"
            help = "number of epochs after which learning rate is halved"
            arg_type = Int32
            default = Int32(8)
        "--no-batchnorm"
            help = "disables batch-normalisation"
            action = :store_true
        "--lambda-c"
            help = "codes sparsity"
            arg_type = Float32
            default = Float32(0)
        "--lambda-w"
            help = "weight sparsity"
            arg_type = Float32
            default = Float32(0.0)
        "--mu"
            help = "initial mu parameter"
            arg_type = Float32
            default = Float32(0.003)
        "--d-mu"
            help = "increase in mu after every mini-batch"
            arg_type = Float32
            default = Float32(1 / 300)
        "--postprocessing-steps"
            help = "number of Carreirs-Peripinan post-processing steps after training"
            arg_type = Int32
            default = Int32(0)
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
    
parsed_args = parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg => $val")
end