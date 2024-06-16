using ProgressBars
using Random: seed!

include("acceptance_model.jl")
include("compensation_demanded.jl")
include("dataloading.jl")
include("visualizations.jl")

using .AcceptanceModel: acceptance_model, acceptance_inference, acceptance_prediction
using .CompensationDemanded
using .DataLoading: VALID_DAMAGE_TYPES, load_dataset, data_to_dict, simplex_grid
using .Visualizations

seed!(42)

# Prepare data
const data = load_dataset()
const individuals = unique(data[:, :responseID])

const damage_means = Dict(VALID_DAMAGE_TYPES .=> [COMPENSATION_DEMANDED_TABLE[:, damage_type][1] for damage_type in VALID_DAMAGE_TYPES])
const damage_stds = Dict(VALID_DAMAGE_TYPES .=> [COMPENSATION_DEMANDED_TABLE[:, damage_type][5] for damage_type in VALID_DAMAGE_TYPES]) 

const acceptances = data_to_dict(data, :responseID, :bargain_accepted)
const amounts = data_to_dict(data, :responseID, :amount_offered)
const damages = data_to_dict(data, :responseID, :damage_type)
const offers = data_to_dict(data, :responseID, :amount_offered)

const type_priors = simplex_grid(3, 30)

"Run Bayesian inference and predict on the dataset."
function run_acceptance_model(
    type_priors::Vector{Vector{Float64}},
    acceptances::Dict,
    amounts::Dict,
    damages::Dict,
    individuals::Vector{String},
    damage_means::Dict{Symbol, Float64},
    damage_stds::Dict{Symbol, Float64}
)
    grid_search_results = Dict()
    grid_search_predictions = Dict()
    for type_prior in ProgressBar(type_priors)
        model_results = Dict()
        model_predictions = Dict()
        for individual in individuals
            model_results[individual] = acceptance_inference(
                acceptances[individual],
                amounts[individual],
                damages[individual],
                damage_means,
                damage_stds,
                type_prior
            )
            model_predictions[individual] = acceptance_prediction(
                model_results[individual],
                amounts[individual],
                damages[individual],
                damage_means,
                damage_stds,
                type_prior
            )
        end
        grid_search_results[type_prior] = model_results
        grid_search_predictions[type_prior] = model_predictions
    end
    return grid_search_results, grid_search_predictions
end

function get_model_likelihood(model_results)
    return sum([v.lml for v in values(model_results)])
end

function filter_valid_results(grid_search_results::Dict)
    # Filter individuals that could not be modeled (log likelihood of -Inf) from the results
    get_valid_results(model_results) = Dict(filter(p -> last(p).lml != -Inf, collect(model_results)))

    keys_to_delete = filter(x -> x[1] ∈ [0] || x[3] ∈ [0], type_priors)  # Filter the type priors where any rule-based and agreement-based individuals don't exist
    valid_results_without_rr = [type_prior => get_valid_results(model_results) for (type_prior, model_results) in grid_search_results if type_prior ∉ keys_to_delete]
    valid_results_with_rr = [k => v for (k, v) in valid_results_without_rr if k[2] != 0]

    return valid_results_without_rr, valid_results_with_rr
end

function get_prior_likelihoods(valid_results)
    prior_likelihoods = Dict([type_prior => get_model_likelihood(model_results) for (type_prior, model_results) in valid_results])
    prior_likelihoods = sort(collect(prior_likelihoods), by=last, rev=true)

    return prior_likelihoods
end

function main(damage_type::Symbol = :bluehouse)
    grid_search_results, grid_search_predictions = run_acceptance_model(type_priors, acceptances, amounts, damages, individuals, damage_means, damage_stds)
    valid_results_without_rr, valid_results_with_rr = filter_valid_results(grid_search_results)
    prior_likelihoods = get_prior_likelihoods(valid_results_with_rr)

    best_prior = first(prior_likelihoods[1])
    println("Best Performing Prior: $best_prior")

    best_likelihood_without_rr = last(first(filter(x -> first(x)[2] == 0, get_prior_likelihoods(valid_results_without_rr))))
    worst_likelihood_with_rr = last(last(prior_likelihoods))
    println("The best performing model without resource-rationality has a log likelihood of $best_likelihood_without_rr.")
    println("The worst performing model with it has a log likelihood of $worst_likelihood_with_rr.")

    plot_priors_heatmap(first.(prior_likelihoods), last.(prior_likelihoods))
    plot_types(damage_type, grid_search_predictions[best_prior], amounts, damages, acceptances, Dict(valid_results_with_rr)[best_prior])
    plot_thresholds(damage_type, grid_search_predictions[best_prior], amounts, damages, acceptances, Dict(valid_results_with_rr)[best_prior])
end

main()
