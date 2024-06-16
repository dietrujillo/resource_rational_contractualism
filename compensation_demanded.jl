module CompensationDemanded

using CSV
using DataFrames
using StatsBase: mean, iqr, median, quantile, std

export COMPENSATION_DEMANDED_TABLE

include("dataloading.jl")
using .DataLoading: VALID_DAMAGE_TYPES

_combine_function(df::DataFrame, f)::DataFrameRow = combine(df, names(df) .=> f, renamecols=false)[1, :]
function _build_damage_table(compensation_demanded::DataFrame)::DataFrame
    out = DataFrame()
    push!(out, _combine_function(compensation_demanded, mean))
    push!(out, _combine_function(compensation_demanded, x -> quantile(x, 0.9)))
    push!(out, _combine_function(compensation_demanded, median))
    push!(out, _combine_function(compensation_demanded, iqr))
    push!(out, _combine_function(compensation_demanded, std))
    return out
end

const COMPENSATION_DEMANDED_FILEPATH = "data/data_wide_willing.csv"
const COMPENSATION_DEMANDED_TABLE = _build_damage_table(
    CSV.read(COMPENSATION_DEMANDED_FILEPATH, DataFrame)[:, VALID_DAMAGE_TYPES]
)

end # Module CompensationDemanded
