using Pkg

# Install once if needed:
# Pkg.add(url="https://github.com/meringlab/FlashWeave.jl")

using FlashWeave

base = @__DIR__
data_path = joinpath(base, "flashweave_all_microbes_HC_only.tsv")
meta_path = joinpath(base, "flashweave_metadata_HC_numeric.tsv")

# HC-only reference network. Metadata are supplied to let FlashWeave account for
# available covariates if applicable; for a pure HC-only reference these are
# mostly age/sex/BMI and technical placeholders if provided.
netw_results = learn_network(
    data_path,
    meta_path,
    sensitive=true,
    heterogeneous=false
)

save_network(joinpath(base, "flashweave_HC_reference_network.gml"), netw_results)
save_network(joinpath(base, "flashweave_HC_reference_network_detailed.edgelist"), netw_results, detailed=true)

println("FlashWeave HC reference network complete")
