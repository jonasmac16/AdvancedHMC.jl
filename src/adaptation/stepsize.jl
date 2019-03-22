######################
### Mutable states ###
######################

mutable struct DAState{TI<:Integer,TF<:Real}
    m     :: TI
    ϵ     :: TF
    μ     :: TF
    x_bar :: TF
    H_bar :: TF
end

function DAState(ϵ::Real)
    μ = computeμ(ϵ)
    return DAState(0, ϵ, μ, 0.0, 0.0)
end

function computeμ(ϵ::Real)
    return log(10 * ϵ) # see NUTS paper sec 3.2.1
end

function reset!(dastate::DAState{TI,TF}) where {TI<:Integer,TF<:Real}
    dastate.μ = computeμ(da.state.ϵ)
    dastate.m = zero(TI)
    dastate.x_bar = zero(TF)
    dastate.H_bar = zero(TF)
end

mutable struct MSSState{T<:Real}
    ϵ :: T
end

################
### Adapters ###
################

abstract type StepSizeAdapter <: AbstractAdapter end

struct FixedStepSize{T<:Real} <: StepSizeAdapter
    ϵ :: T
end

function getss(fss::FixedStepSize)
    return fss.ϵ
end

struct DualAveraging{TI<:Integer,TF<:Real} <: StepSizeAdapter
  γ     :: TF
  t_0   :: TF
  κ     :: TF
  δ     :: TF
  state :: DAState{TI,TF}
end

function DualAveraging(γ::AbstractFloat, t_0::AbstractFloat, κ::AbstractFloat, δ::AbstractFloat, ϵ::AbstractFloat)
    return DualAveraging(γ, t_0, κ, δ, DAState(ϵ))
end

function DualAveraging(δ::AbstractFloat, ϵ::AbstractFloat)
    return DualAveraging(0.05, 10.0, 0.75, δ, ϵ)
end

function getss(da::DualAveraging)
    return da.state.ϵ
end

struct ManualSSAdapter{T<:Real} <:StepSizeAdapter
    state :: MSSState{T}
end

function getss(mssa::ManualSSAdapter)
    return mssa.state.ϵ
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp
function adapt_stepsize!(da::DualAveraging, α::AbstractFloat)
    DEBUG && @debug "Adapting step size..." α
    da.state.m += 1
    m = da.state.m

    # Clip average MH acceptance probability
    α = α > 1 ? 1 : α

    γ = da.γ; t_0 = da.t_0; κ = da.κ; δ = da.δ
    μ = da.state.μ; x_bar = da.state.x_bar; H_bar = da.state.H_bar

    η_H = 1.0 / (m + t_0)
    H_bar = (1.0 - η_H) * H_bar + η_H * (δ - α)

    x = μ - H_bar * sqrt(m) / γ     # x ≡ logϵ
    η_x = m^(-κ)
    x_bar = (1.0 - η_x) * x_bar + η_x * x

    ϵ = exp(x)
    DEBUG && @debug "Adapting step size..." "new ϵ = $ϵ" "old ϵ = $(da.state.ϵ)"

    if isnan(ϵ) || isinf(ϵ)
        @warn "Incorrect ϵ = $ϵ; ϵ_previous = $(da.state.ϵ) is used instead."
        ϵ = da.state.ϵ
        x_bar = da.state.x_bar
        H_bar = da.state.H_bar
    end

    da.state.ϵ = ϵ
    da.state.x_bar = x_bar
    da.state.H_bar = H_bar
end

function adapt!(da::DualAveraging, θ::AbstractVector{<:Real}, α::AbstractFloat)
    adapt_stepsize!(da, α)
end
