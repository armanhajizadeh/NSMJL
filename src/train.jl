# ============================================================================
#  Step 2 — training layer  (port of nsm/nsm.py :: rmse, fit_rmse; and
#  nsm/nsm_system.py :: root_mean_squared_error)
#
#  Data model (from utilities.py :: process_df): each training Sample is an
#  (initial condition -> one later observation) pair, integrated 0 -> tf.
#  Loss per sample = sqrt(mean((target - endpoint)^2)); dataset loss = mean
#  over samples. fit_rmse! is per-sample Adam with gradient clipping at 1000,
#  matching the reference.
#
#  GRADIENTS: reverse-mode adjoint via Zygote + SciMLSensitivity
#  (InterpolatingAdjoint + ReverseDiffVJP). Cost is (roughly) independent of
#  parameter count, so this is the win at 25 species; for tiny models the old
#  ForwardDiff path can still be competitive. predict_endpoint forwards
#  `sensealg` into solve so the same function serves forward eval (sensealg
#  omitted) and the adjoint (sensealg supplied by the fitter).
#
#  Perturbation note: `uin` here is a static per-sample input vector (as in the
#  reference data model). To train with a TIME-DEPENDENT perturbation, replace
#  the `_ -> uin` closure below with the sample's own u(t) — the rest is unchanged.
# ============================================================================

using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff
using Zygote, SciMLSensitivity        # reverse-mode adjoint through the ODE solve
using Random
using Statistics

struct Sample
    u0::Vector{Float64}       # [s_ic; m_ic]
    target::Vector{Float64}   # [s_obs; m_obs] at t = tf  (NaN entries are ignored)
    tf::Float64               # integration horizon
    uin::Vector{Float64}      # static input / perturbation vector (length n_u)
end

# integrate one sample 0 -> tf and return the endpoint state [s; m].
# Forward eval: call with sensealg=nothing (default). Under Zygote/SciMLSensitivity,
# the fitter passes a reverse-mode `sensealg`, which `solve` uses for the adjoint.
function predict_endpoint(cfg::NSMConfig, p, u0, tf, uin;
                          solver=Tsit5(), reltol=1e-8, abstol=1e-8, sensealg=nothing)
    T   = promote_type(eltype(p), eltype(u0))
    u0T = convert(Vector{T}, u0)
    # per-sample config carrying this sample's (static) input schedule
    scfg = NSMConfig(cfg.n_s, cfg.n_m, cfg.n_u, cfg.n_h, cfg.n_layers,
                     cfg.s_cap, cfg.m_cap, _ -> uin)
    f!(du, u, pp, t) = nsm_rhs!(du, u, pp, t, scfg)
    prob = ODEProblem(f!, u0T, (zero(T), T(tf)), p)
    sol  = solve(prob, solver; reltol=reltol, abstol=abstol, sensealg=sensealg,
                 save_everystep=false, save_start=false)
    return sol.u[end]
end

# RMSE for one sample (NaN targets contribute zero error, as in the reference)
function sample_rmse(cfg::NSMConfig, p, u0, target, tf, uin; kw...)
    pred = predict_endpoint(cfg, p, u0, tf, uin; kw...)
    err  = map((tr, pr) -> isnan(tr) ? zero(pr) : tr - pr, target, pred)
    return sqrt(mean(abs2, err))
end

# dataset RMSE = mean of per-sample RMSE
function rmse(cfg::NSMConfig, p, data::AbstractVector{Sample}; kw...)
    return mean(s -> sample_rmse(cfg, p, s.u0, s.target, s.tf, s.uin; kw...), data)
end

"""
    fit_rmse!(cfg, p, data; lr, max_epochs, clip, solver, reltol, abstol, ...)

Per-sample Adam minimisation of the RMSE (mutates `p` in place). Gradients are
taken through the ODE solve with a reverse-mode adjoint (Zygote +
SciMLSensitivity). Faithful to nsm.py::fit_rmse (SGD over shuffled samples,
Adam moments, |grad|>clip zeroed).

Stops on a **convergence threshold** rather than a fixed epoch count: every
`check_every` epochs it measures the loss and stops once the relative
improvement `(prev-cur)/prev` stays below `rtol` for `patience` consecutive
checks, or when `max_epochs` is reached. If `valid` is given, convergence is
judged on that held-out set (recommended); otherwise on the training loss.
Returns the loss history (the monitored loss, one entry per check).
"""
function fit_rmse!(cfg::NSMConfig, p, data::AbstractVector{Sample};
                   lr=1e-2, max_epochs=5000, rtol=1e-4, patience=3, check_every=10,
                   clip=1000.0, valid=nothing,
                   solver=Tsit5(), reltol=1e-7, abstol=1e-7,
                   verbose=true, rng=Random.default_rng())
    ax = getaxes(p)
    z  = getdata(p)                    # mutating z mutates p
    m  = zero(z); v = zero(z); t = 0
    sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))   # reverse-mode VJP
    monitor = valid === nothing ? data : valid
    hist = Float64[]; prev = Inf; stall = 0; epoch = 0
    while epoch <= max_epochs
        if epoch % check_every == 0
            # logging is a plain forward eval — no sensealg
            r = rmse(cfg, ComponentArray(z, ax), monitor;
                     solver=solver, reltol=reltol, abstol=abstol)
            push!(hist, r)
            verbose && println("epoch $epoch  $(valid === nothing ? "train" : "valid") rmse $(round(r, digits=6))")
            if (prev - r) / prev < rtol
                stall += 1
                if stall >= patience
                    verbose && println("converged at epoch $epoch")
                    break
                end
            else
                stall = 0
            end
            prev = r
        end
        for s in shuffle(rng, data)
            g = first(Zygote.gradient(
                    zz -> sample_rmse(cfg, ComponentArray(zz, ax),
                                      s.u0, s.target, s.tf, s.uin;
                                      solver=solver, reltol=reltol, abstol=abstol,
                                      sensealg=sense),
                    z))
            t += 1
            mc = 1 - 0.9^t; vc = 1 - 0.999^t
            @inbounds for i in eachindex(z)
                gi   = abs(g[i]) < clip ? g[i] : zero(eltype(g))
                m[i] = 0.9   * m[i] + 0.1   * gi
                v[i] = 0.999 * v[i] + 0.001 * gi^2
                z[i] -= lr * (m[i] / mc) / (sqrt(v[i] / vc) + 1e-8)
            end
        end
        epoch += 1
    end
    return hist
end
