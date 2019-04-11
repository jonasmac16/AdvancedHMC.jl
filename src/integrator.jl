####
#### Numerical methods for simulating Hamiltonian trajectory.
####


abstract type AbstractIntegrator end

struct Leapfrog{T<:AbstractFloat} <: AbstractIntegrator
    ϵ   ::  T
end

# Create a `Leapfrog` with a new `ϵ`
function (::Leapfrog)(ϵ::AbstractFloat)
    return Leapfrog(ϵ)
end

function lf_momentum(ϵ::T,
        h::Hamiltonian, θ::AbstractVector{T},
        r::AbstractVector{T}; termination::Tt = Termination()
    ) where {T<:Real, Tt}
    _∂H∂θ = ∂H∂θ(h, θ)
    termination = combine(Termination(_∂H∂θ), termination)
    if !is_terminated(termination)
        r = r - ϵ * _∂H∂θ
    end
    return r, termination
end

function lf_position(ϵ::T,
        h::Hamiltonian, θ::AbstractVector{T},
        r::AbstractVector{T}; termination::Tt = Termination()
        ) where {T<:Real, Tt}
    _∂H∂r = ∂H∂r(h, r)
    termination = combine(Termination(_∂H∂r), termination)
    # Only update θ if previous udpates to r has no numerical issue.
    if !is_terminated(termination)
        θ = θ + ϵ * _∂H∂r
    end
    return θ, termination
end

# TODO: double check the function below to see if it is type stable or not
function step(lf::Leapfrog{F},
        h::Hamiltonian, θ::AbstractVector{T},
        r::AbstractVector{T}, n_steps::Int=1
    ) where {F<:AbstractFloat,T<:Real}
    fwd = n_steps > 0 # simulate hamiltonian backward when n_steps < 0
    _n_steps = abs(n_steps)
    ϵ = fwd ? lf.ϵ : - lf.ϵ

    r_new, t = lf_momentum(ϵ/2, h, θ, r)
    for i = 1:_n_steps
        θ_new, t = lf_position(ϵ, h, θ, r_new; termination=t)
        r_new, t = lf_momentum(i == n_steps ? ϵ / 2 : ϵ, h,
                                                θ_new, r_new; termination=t)
        if !is_terminated(t)
            # Here r has half more step leapfrog when 1 < i < _n_steps.
            θ, r = θ_new, r_new
        else
            # Reverse half leapfrog step from r when breaking
            #  the loop immaturely.
            if i > 1 && i < _n_steps
                r, _ = lf_momentum(-lf.ϵ / 2, h, θ, r)
            end
            break
        end
    end

    return θ, r, !t
end

###
### Utility function.
###

# Termporary function before formally introducing a `Termination` type.
Termination(v::AbstractVector{<:Real}=[0.], t::Tt=true) where Tt = !is_valid(v)
Termination(t::Tt=true) where Tt = t
is_terminated(t::Tt) where Tt = t
combine(t1::Tt, t2::Tt) where Tt = t1 && t2

function ValueTermination(f::Function, args...)
    res = f(args...);
    res, Termination(!is_valid(res))
end

function is_valid(v::AbstractVector{<:Real})
    if any(isnan, v) || any(isinf, v)
        return false
    else
        return true
    end
end
