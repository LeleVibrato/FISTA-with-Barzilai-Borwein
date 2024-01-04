import LinearAlgebra: norm
using SparseArrays
using Plots
using LaTeXStrings

# """
# State:
#     x
#     x_previous
#     f_x
#     f_x_previous
#     g
#     g_previous

# Options:
#     x_tol = nothing,
#     f_tol = nothing,
#     g_tol = nothing,
#     x_abstol::Real = 0.0,
#     x_reltol::Real = 0.0,
#     f_abstol::Real = 0.0,
#     f_reltol::Real = 0.0,
#     g_abstol::Real = 1e-8,
#     g_reltol::Real = 1e-8,
#     iterations::Int = 1_000,


# 最后从options中构造并返回
# MultivariateOptimizationResults:
#     method::O
#     initial_x::Tx
#     minimizer::Tx
#     minimum::Tf
#     iterations::Int
#     iteration_converged::Bool
#     x_converged::Bool
#     x_abstol::Tf
#     x_reltol::Tf
#     x_abschange::Tc
#     x_relchange::Tc
#     f_converged::Bool
#     f_abstol::Tf
#     f_reltol::Tf
#     f_abschange::Tc
#     f_relchange::Tc
#     g_converged::Bool
#     g_abstol::Tf
#     g_residual::Tc


# assess_convergence 写一个函数判断是否收敛   

# 进行线性搜索
# state.alpha, ϕalpha = method.linesearch!(d, state.x, state.s, state.alpha,
#                    state.x_ls, phi_0, dphi_0)

# """

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
end

function IterationState(prob::FISTA{T}) where {T<:AbstractFloat}
    x_previous = randn(T, prob.x_dimension)
    g_previous = gradient(prob, x_previous)
    x = randn(T, prob.x_dimension)
    g = gradient(prob, x)
    initial_α = 1e-8

    state = IterationState{T}(
        x,
        x_previous,
        objective_value(prob, x),
        objective_value(prob, x_previous),
        g,
        g_previous,
        initial_α
    )
    return state
end

function proximal(prob::FISTA, state::IterationState, x::AbstractArray)
    sign.(x) .* max.(abs.(x) .- state.α * prob.μ, 0)
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

function update_results!(results::OptimizationResults, state::IterationState, iteration::Integer, prob::FISTA)
    current_objective_value = objective_value(prob, state.x)
    # println(current_objective_value)
    results.f_historical_values[iteration] = current_objective_value

    if current_objective_value < results.minimum
        results.minimizer .= state.x
        results.minimum = current_objective_value
    end
    println(results.minimum)
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

mutable struct BarzilaiBorweinMethod
    γ
    η
    Q
    C
    BarzilaiBorweinMethod(γ=1e-4, η=0.85, Q=1.0; C) = new(γ, η, Q, C)
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
        state.α = 1e-8
    end

    state.α = max(min(state.α, 1e20), 1e-20)
end

# function linear_search!(state::IterationState, 
#     ls_state::BarzilaiBorweinMethod, prob::FISTA)

# end

function optimize(A, b, μ; iterations=1_000)
    prob = FISTA(A, b, μ; iterations)
    state = IterationState(prob)
    results = OptimizationResults(prob)
    results.initial_x = state.x
    initial_f_x = objective_value(prob, results.initial_x)
    # linear_search_state = BarzilaiBorweinMethod(C=initial_f_x)

    for k = 1:prob.iterations
        # linear_search!(state, linear_search_state, prob)
        update_state!(state, prob, k)
        update_results!(results, state, k, prob)
        if k > 2 && assess_convergence(prob, state)
            results.iteration_converged = true
            results.iterations = k
            println(k)
            break
        end
        set_step!(state, k)
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
    println(results.iteration_converged)

    it = 100

    plot(1:it,
        ((results.f_historical_values.-results.minimum)./results.minimum)[1:it],
        dpi=300, yscale=:log10
    )
    ylabel!(L"$(f(x^k) - f^{*})/f^{*}$")
    xlabel!("Iterations")

end

test()
