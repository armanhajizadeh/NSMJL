# ============================================================================
#  Step 5 — gLV residual-penalty (PINN) variant
#
#  The contrast to the embedded NSM: here the RHS is a *black box*
#     dx/dt = NN([x; u(t)])                        (no mechanism, no x-multiply)
#  and physics enters only through a residual term added to the loss, which
#  penalises departure from generalized Lotka-Volterra dynamics
#     f_gLV(x) = x .* (r + A·x)
#
#  loss = (1-λ)·data_rmse + λ·mean_c ‖ NN([x_c; u]) − f_gLV(x_c) ‖²
#
#  Same data, same solver, same ForwardDiff path as the embedded model, so the
#  two are directly comparable (fit quality, and — via step 3b — evidence).
#  This is the "pNODE" formulation from the gNODE paper: gLV in the LOSS, not
#  the architecture.
# ============================================================================

using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff
using Statistics

struct BBConfig{F}
    n_x::Int          # augmented state dim (= n_s + n_m)
    n_u::Int
    n_h::Int
    n_layers::Int
    inputs::F         # inputs(t) -> vector length n_u
end

function bb_init(n_x, n_u, n_h, n_layers; Tp::Type=Float64)
    ComponentArray(
        Wi = zeros(Tp, n_h, n_x + n_u), bi = zeros(Tp, n_h),
        Wh = zeros(Tp, n_h, n_h, n_layers - 1), bh = zeros(Tp, n_h, n_layers - 1),
        Wo = zeros(Tp, n_x, n_h), bo = zeros(Tp, n_x),
        r  = zeros(Tp, n_x), A = zeros(Tp, n_x, n_x),
    )
end

# black-box neural network derivative at state x with input uvec
function bb_deriv(cfg::BBConfig, p, x, uvec)
    h = tanh.(p.Wi * vcat(x, uvec) .+ p.bi)
    @inbounds for i in 1:(cfg.n_layers - 1)
        h = tanh.(view(p.Wh, :, :, i) * h .+ view(p.bh, :, i)) ./ 10
    end
    return (p.Wo * h .+ p.bo) ./ 10
end

# gLV vector field used by the residual
glv_deriv(p, x) = x .* (p.r .+ p.A * x)

function bb_rhs!(du, x, p, t, cfg::BBConfig)
    du .= bb_deriv(cfg, p, x, cfg.inputs(t))
    return nothing
end

function bb_endpoint(cfg::BBConfig, p, u0, tf, uin;
                     solver=Tsit5(), reltol=1e-8, abstol=1e-8)
    T = promote_type(eltype(p), eltype(u0)); u0T = convert(Vector{T}, u0)
    scfg = BBConfig(cfg.n_x, cfg.n_u, cfg.n_h, cfg.n_layers, _ -> uin)
    f!(du, x, pp, t) = bb_rhs!(du, x, pp, t, scfg)
    prob = ODEProblem(f!, u0T, (zero(T), T(tf)), p)
    solve(prob, solver; reltol=reltol, abstol=abstol,
          save_everystep=false, save_start=false).u[end]
end

# data term: mean per-sample endpoint RMSE
function bb_data_rmse(cfg::BBConfig, p, data::AbstractVector{Sample}; kw...)
    return mean(data) do s
        pred = bb_endpoint(cfg, p, s.u0, s.tf, s.uin; kw...)
        err  = map((tr, pr) -> isnan(tr) ? zero(pr) : tr - pr, s.target, pred)
        sqrt(mean(abs2, err))
    end
end

# physics residual at collocation points (each sample's IC and observation)
function bb_residual(cfg::BBConfig, p, data::AbstractVector{Sample})
    tot = zero(eltype(p)); cnt = 0
    for s in data
        for c in (s.u0, s.target)
            any(isnan, c) && continue
            d = bb_deriv(cfg, p, c, s.uin) .- glv_deriv(p, c)
            tot += mean(abs2, d); cnt += 1
        end
    end
    return tot / cnt
end

# combined objective
function residual_loss(cfg::BBConfig, p, data::AbstractVector{Sample}; lambda=0.5, kw...)
    return (1 - lambda) * bb_data_rmse(cfg, p, data; kw...) +
           lambda * bb_residual(cfg, p, data)
end

"""
    fit_residual!(cfg, p, data; lambda, lr, max_epochs, rtol, patience, ...)

Full-batch Adam on the combined data + gLV-residual loss (mutates `p`). Stops on
a convergence threshold: halts once the relative improvement in the loss stays
below `rtol` for `patience` consecutive checks, or at `max_epochs`. Returns the
loss history.
"""
function fit_residual!(cfg::BBConfig, p, data::AbstractVector{Sample};
                       lambda=0.5, lr=5e-3, max_epochs=2000, rtol=1e-4,
                       patience=3, check_every=10, clip=1000.0,
                       solver=Tsit5(), reltol=1e-7, abstol=1e-7, verbose=true)
    ax = getaxes(p); z = getdata(p)
    m = zero(z); v = zero(z); t = 0
    hist = Float64[]; prev = Inf; stall = 0; epoch = 0
    obj(zz) = residual_loss(cfg, ComponentArray(zz, ax), data;
                            lambda=lambda, solver=solver, reltol=reltol, abstol=abstol)
    while epoch <= max_epochs
        if epoch % check_every == 0
            val = obj(z); push!(hist, val)
            verbose && println("epoch $epoch  loss $(round(val, digits=6))")
            if (prev - val) / prev < rtol
                stall += 1
                if stall >= patience
                    verbose && println("converged at epoch $epoch")
                    break
                end
            else
                stall = 0
            end
            prev = val
        end
        g = ForwardDiff.gradient(obj, z)
        t += 1; mc = 1 - 0.9^t; vc = 1 - 0.999^t
        @inbounds for i in eachindex(z)
            gi   = abs(g[i]) < clip ? g[i] : zero(eltype(g))
            m[i] = 0.9   * m[i] + 0.1   * gi
            v[i] = 0.999 * v[i] + 0.001 * gi^2
            z[i] -= lr * (m[i] / mc) / (sqrt(v[i] / vc) + 1e-8)
        end
        epoch += 1
    end
    return hist
end

# full trajectory from the black-box model (for plotting / comparison)
function bb_predict_point(cfg::BBConfig, p, u0, t_eval, inputs;
                          solver=Vern9(), reltol=1e-8, abstol=1e-8)
    scfg = BBConfig(cfg.n_x, cfg.n_u, cfg.n_h, cfg.n_layers, inputs)
    f!(du, x, pp, t) = bb_rhs!(du, x, pp, t, scfg)
    prob = ODEProblem(f!, collect(float.(u0)), (float(first(t_eval)), float(last(t_eval))), p)
    sol  = solve(prob, solver; saveat=t_eval, reltol=reltol, abstol=abstol)
    return permutedims(reduce(hcat, sol.u))
end
