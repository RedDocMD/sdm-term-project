module Utils

import CUDA

function get_devices(device_num::Int=0)
    device = CUDA.CuDevice(device_num) 
    num_gpus = length(CUDA.devices())
    return device, num_gpus
end

include("datasets.jl")

end