module Altmin

import Flux

function mark_code_mods(model, module_types=[])
    if size(module_types) == 0
        module_types = [Flux.Conv, Flux.Dense, Flux.BatchNorm]
    end

end

end