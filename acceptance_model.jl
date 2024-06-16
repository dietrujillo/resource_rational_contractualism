module AcceptanceModel

using Gen
using Gen.Distributions: SkewNormal, Normal
import Gen.Distributions: cdf
using StatsBase: mean
using Integrals

RULEBASED = 1
FLEXIBLE = 2
AGREEMENTBASED = 3


@gen function acceptance_model(
    amounts_offered::Vector{Float64},
    damage_types::Vector{Symbol},
    damage_means::Dict{Symbol, Float64},
    damage_stds::Dict{Symbol, Float64},
    type_prior::Vector{Float64},
    thresholds::Vector{Float64}
)

    # Sample individual type
    individual_type ~ categorical(type_prior)

    if individual_type == RULEBASED # Rule-based
        threshold_probs = [1, 0, 0, 0, 0, 0]
    elseif individual_type == FLEXIBLE # Flexible
        threshold_probs = [0, 1//4, 1//4, 1//4, 1//4, 0]
    else # Agreement-based
        threshold_probs = [0, 0, 0, 0, 0, 1]
    end

    # Sample high-stakes threshold
    high_stakes_threshold ~ categorical(threshold_probs)
    threshold_value = thresholds[high_stakes_threshold]

    # Sample acceptance for each offer
    p_accepts = []
    for (i, (amount, dmg_type)) in enumerate(zip(amounts_offered, damage_types))
        if amount <= threshold_value # Always reject if less than threshold
            p_accept = 0.0
        else # Accept if amount > dmg, where dmg ~ N(dmg_mean, dmg_std)
            p_accept = min(1 - 1e-10, max(1e-10, cdf(Normal(damage_means[dmg_type], damage_stds[dmg_type]), amount)))
        end
        accept = {(:acceptance, i)} ~ bernoulli(p_accept)
        push!(p_accepts, p_accept)
    end
    return p_accepts
end

function is_valid_combination(type::Int64, threshold_index::Int64)
    if (type == RULEBASED)
        return threshold_index == 1
    elseif (type == AGREEMENTBASED)
        return threshold_index == 6
    else
    return threshold_index > 1 && threshold_index < 6
    end
end

"Runs exact Bayesian inference over an individual's type and threshold."
function acceptance_inference(
    acceptances::Vector{Bool},
    amounts_offered::Vector{Float64},
    damage_types::Vector{Symbol},
    damage_means::Dict{Symbol, Float64},
    damage_stds::Dict{Symbol, Float64},
    type_prior::Vector{Float64} = ones(3) ./ 3,
    thresholds::Vector{Float64} = [Inf, 1e2, 1e3, 1e4, 1e5, -Inf]
)
    @assert length(amounts_offered) == length(acceptances)
    @assert length(damage_types) == length(amounts_offered)
    N = length(amounts_offered)
    # Construct observation choicemap
    observations = choicemap(((:acceptance, i) => acceptances[i] for i in 1:N)...)
    individual_types = [RULEBASED, FLEXIBLE, AGREEMENTBASED]
    # Construct argument tuple to generative model
    model_args = (amounts_offered, damage_types, damage_means, damage_stds,
                  type_prior, thresholds)
    # Generate traces for each (type, threshold) combination
    traces = Trace[]
    weights = Float64[]
    for type in individual_types
        for (threshold_idx, threshold) in enumerate(thresholds)
            if (is_valid_combination(type, threshold_idx))
                constraints = choicemap()
                constraints[:individual_type] = type
                constraints[:high_stakes_threshold] = threshold_idx
                constraints = merge(constraints, observations)
                tr, w = Gen.generate(acceptance_model, model_args, constraints)
                push!(traces, tr)
                push!(weights, w)
            end
        end
    end
    # Compute log marginal likelihood
    lml = logsumexp(weights)
    # Compute posterior probability of each (type, threshold) combination
    joint_probs = exp.(weights .- lml)
    # Compute posterior probability over types
    type_probs = zeros(length(type_prior))
    for (i, type) in enumerate(individual_types)
        for (tr, p) in zip(traces, joint_probs)
            if tr[:individual_type] == type
                type_probs[i] += p
            end
        end
    end
    # Compute posterior probability over thresholds
    threshold_probs = zeros(length(thresholds))
    for (i, threshold) in enumerate(thresholds)
        for (tr, p) in zip(traces, joint_probs)
            if tr[:high_stakes_threshold] == i
                threshold_probs[i] += p
            end
        end
    end
    # Return inference results
    return (
        weights = weights,
        joint_probs = joint_probs,
        type_probs = type_probs,
        threshold_probs = threshold_probs,
        lml = lml,
        traces = traces
    )
end

"Generates samples for acceptance conditioned on individual type and threshold"
function acceptance_prediction(
    model_results,
    amounts_offered::Vector{Float64},
    damage_types::Vector{Symbol},
    damage_means::Dict{Symbol, Float64},
    damage_stds::Dict{Symbol, Float64},
    type_prior::Vector{Float64} = ones(3) ./ 3,
    thresholds::Vector{Float64} = [Inf, 1e2, 1e3, 1e4, 1e5, -Inf]
)
    # Constrain individual type and threshold to the maximum likelihood estimates
    constraints = Gen.choicemap()
    constraints[:individual_type] = argmax(model_results.type_probs)
    constraints[:high_stakes_threshold] = argmax(model_results.threshold_probs)

    (new_trace, _) = Gen.generate(acceptance_model, (amounts_offered, damage_types, damage_means, damage_stds, type_prior, thresholds), constraints)
    predictions = Gen.get_retval(new_trace)
    
    return predictions
end

end # Module AcceptanceModel