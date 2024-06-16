module DataLoading

export VALID_DAMAGE_TYPES, load_dataset, data_to_dict, simplex_grid

using Combinatorics
using CSV
using DataFrames
using StatsBase

const DATA_PATH = "data/within-data.csv"
const EXCLUDE_DATA_PATH = "data/exclusions.csv"
const VALID_DAMAGE_TYPES::Vector{Symbol} = [
    :bluemailbox,
    :blueoutsidedoor,
    :bluehouse,
    :cuttree,
    :breakwindows,
    :razehouse,
    :bleachlawn, 
    :blueinsidedoor,
    :erasemural,
    :smearpoop
]

const OFFER_AS_INT_DICT = Dict(
    "hundred" => 100,
    "thousand" => 1000,
    "tenthousand" => 10000,
    "hunthousand" => 100000,
    "million" => 1000000,
)

function load_dataset(data_path::String = DATA_PATH, exclusions::Bool = true)
    table = CSV.read(data_path, DataFrame)
    rename!(table, [:subjectcode, :answer, :question, :context] .=> [:responseID, :bargain_accepted, :amount_offered, :damage_type])

    if exclusions
        exclusion_table = DataFrame(CSV.File(EXCLUDE_DATA_PATH))
        delete!(exclusion_table, [1,2,3,4,nrow(exclusion_table)])
        rename!(exclusion_table, [:Column2] .=> [:responseID])
        select!(exclusion_table, [:excluded, :responseID])
        excludeIDs = [id for id in dropmissing(exclusion_table, disallowmissing=true)[:,:responseID]]
        for id in excludeIDs
            table = filter!(:responseID => !=(id), table)
        end
    end

    table[!,:responseID] = convert.(String, table[:,:responseID])
    table[!,:damage_type] = Symbol.(table[:,:damage_type])
    table[!,:amount_offered] = convert.(Float64, table[:,:amount_offered])
    table[!,:bargain_accepted] = convert.(Bool, table[:,:bargain_accepted])

    return table
end

function data_to_dict(df, key_column, value_column)
    out = Dict()
    for key in unique(df[:, key_column])
        key_df = filter(key_column => (x -> x == key), df)
        value = convert(Vector, key_df[:, value_column])
        out[key] = value
    end
    return out
end

function simplex_grid(num_outcomes, points_per_dim)
    num_points = multinomial(num_outcomes-1, points_per_dim)
    points = Array{Float64,2}(undef, num_outcomes, num_points)
    for (p,comb) in enumerate(with_replacement_combinations(1:num_outcomes, points_per_dim))
        distr = counts(comb, 1:num_outcomes) ./ points_per_dim
        points[:,p] = distr
    end
    return [points[:,i] for i in 1:size(points,2)]
end

end # Module DataLoading
