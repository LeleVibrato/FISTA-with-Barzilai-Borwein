import LinearAlgebra: norm, eigvals
using SparseArrays
using Plots
using LaTeXStrings

struct FISTA{T<:AbstractFloat}
    A::AbstractMatrix{T}
    b::AbstractVector{T}
    μ::T
    x_tol::T
    x_dimension::Integer
    f_tol::T
    g_tol::T
    iterations::Integer
end

function FISTA(A::AbstractMatrix{T},
    b::AbstractVector{T},
    μ::T;
    x_tol::T=1e-12,
    f_tol::T=1e-12,
    g_tol::T=1e-12,
    iterations::Integer=1_000) where {T<:AbstractFloat}
    FISTA{T}(
        A,
        b,
        μ,
        x_tol,
        size(A)[2],
        f_tol,
        g_tol,
        iterations
    )
end

mutable struct OptimizationResults{T<:AbstractFloat}
    initial_x::AbstractVector{T}
    minimizer::AbstractVector{T}
    minimum::T
    f_historical_values::AbstractVector{T}
    iterations::Integer
    iteration_converged::Bool
end

function OptimizationResults(prob::FISTA{T}) where {T<:AbstractFloat}
    OptimizationResults{T}(
        randn(T, prob.x_dimension),
        randn(T, prob.x_dimension),
        Inf,
        zeros(T, prob.iterations),
        prob.iterations,
        false
    )
end

mutable struct IterationState{T<:AbstractFloat}
    x::AbstractVector{T}
    x_previous::AbstractVector{T}
    f_x::T
    f_x_previous::T
    g::AbstractVector{T}
    g_previous::AbstractVector{T}
    α::T
    const α₀::T
end

function IterationState(prob::FISTA{T}) where {T<:AbstractFloat}
    x_previous = randn(T, prob.x_dimension)
    g_previous = gradient(prob, x_previous)
    x = randn(T, prob.x_dimension)
    g = gradient(prob, x)
    initial_α = eigvals(prob.A' * prob.A)[end]

    state = IterationState{T}(
        x,
        x_previous,
        objective_value(prob, x),
        objective_value(prob, x_previous),
        g,
        g_previous,
        initial_α,
        initial_α
    )
    return state
end

function proximal(prob::FISTA, state::IterationState, x::AbstractArray)
    @. sign(x) * max(abs(x) - state.α * prob.μ, 0)
end

function gradient(prob::FISTA, y::AbstractArray)
    prob.A' * (prob.A * y .- prob.b)
end

function objective_value(prob::FISTA, x::AbstractArray)
    0.5 * norm(prob.A * x .- prob.b, 2)^2 + prob.μ * norm(x, 1)
end

function update_state!(state::IterationState, prob::FISTA, k::Integer)
    θ = (k - 1) / (k + 2)
    y = state.x + θ * (state.x - state.x_previous)
    y .= y .- state.α .* gradient(prob, y)
    state.x_previous .= state.x
    state.g_previous .= state.g
    state.x .= proximal(prob, state, y)
    state.g .= gradient(prob, state.x)
end

function update_results!(results::OptimizationResults, state::IterationState,
    iteration::Integer, prob::FISTA)
    current_objective_value = objective_value(prob, state.x)
    results.f_historical_values[iteration] = current_objective_value
    if current_objective_value < results.minimum
        results.minimizer .= state.x
        results.minimum = current_objective_value
    end
end

function assess_convergence(prob::FISTA, state::IterationState)
    is_f_x_converged = abs(state.f_x - state.f_x_previous) < prob.f_tol
    is_g_converged = norm(state.g - state.g_previous, 2) < prob.g_tol
    if is_f_x_converged || is_g_converged
        true
    else
        false
    end
end

mutable struct BBMethod
    Q
    gamma
    rhols
    Cval
    BBMethod(Q=1.0, gamma=0.85, rhols=1e-6; Cval) = new(Q, gamma, rhols, Cval)
end

function set_step!(state::IterationState, iteration::Integer)
    dx = state.x .- state.x_previous
    dg = state.g .- state.g_previous
    dxg = abs(dx' * dg)
    if dxg > 0
        if iteration % 2 == 0
            state.α = (dx' * dx) / dxg
        else
            state.α = dxg / (dg' * dg)
        end
    else
        state.α = state.α₀
    end
    state.α = max(min(state.α, 1e12), 1e-12)
end

function linear_search!(state::IterationState,
    bb::BBMethod, prob::FISTA, k::Integer)

    θ = (k - 1) / (k + 2)
    y = state.x + θ * (state.x - state.x_previous)
    w = y .- state.α .* gradient(prob, y)
    x_tmp = proximal(prob, state, w)
    f_tmp = objective_value(prob, x_tmp)

    loop_count = 1
    while f_tmp > bb.Cval - 0.5 * state.α * bb.rhols * norm(x_tmp - y, 2)^2 && loop_count <= 10
        state.α *= 0.2
        loop_count += 1
        w .= y .- state.α .* gradient(prob, y)
        x_tmp = proximal(prob, state, w)
        f_tmp = objective_value(prob, x_tmp)
    end
    Qp = bb.Q
    bb.Q = bb.gamma * Qp + 1
    bb.Cval = (bb.gamma * Qp * bb.Cval + f_tmp) / bb.Q
end

function optimize(A, b, μ; iterations=1_000)
    prob = FISTA(A, b, μ; iterations)
    state = IterationState(prob)
    results = OptimizationResults(prob)
    results.initial_x = state.x
    initial_f_x = objective_value(prob, results.initial_x)
    bb = BBMethod(Cval=state.f_x)

    for k = 1:prob.iterations
        if k > 2 && assess_convergence(prob, state)
            results.iteration_converged = true
            results.iterations = k
            break
        end
        set_step!(state, k)
        linear_search!(state, bb, prob, k)
        update_state!(state, prob, k)
        update_results!(results, state, k, prob)
    end

    return results
end

function test()
    m = 512
    n = 1024
    A = randn(m, n)
    u = sprandn(n, 0.1)
    b = A * u
    μ = 1e-3
    results = optimize(A, b, μ; iterations=1000)

    it = 100
    plot(1:it,
        ((results.f_historical_values.-results.minimum)./results.minimum)[1:it],
        dpi=300, yscale=:log10, label="FISTA with Barzilai Borwein"
    )
    ylabel!(L"$(f(x^k) - f^{*})/f^{*}$")
    xlabel!("Iterations")
end

@time test()