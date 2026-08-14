# ============================================================================
#  Step 4 — prediction with uncertainty
#  (port of nsm/nsm.py :: predict_point, predict_sample; and
#   nsm_system.py :: runODE_teval)
#
#  predict_point   — full trajectory at the posterior mean (point forecast).
#  predict_sample  — draw N parameter sets from the posterior, integrate each,
#                    return the ensemble of trajectories.
#  credible_bands  — reduce the ensemble to (mean, lower, upper) per time/output.
#
#  `inputs` is a function u(t): pass `t -> uvec` for a static input (matches the
#  reference / validates against JAX), or any u(t) for a TIME-DEPENDENT
#  perturbation schedule — the trajectory and its uncertainty respond to it.
#
#  (Scaling s_scale/m_scale from the reference is omitted here; data is assumed
#  already in model units, as in the synthetic datasets.)
# ============================================================================

using ComponentArrays
using OrdinaryDiffEq
using Random
using Statistics

# full trajectory at parameters `p`, saved at t_eval (t_eval[1] should be 0).
# Returns a (length(t_eval) × (n_s+n_m)) matrix.
function predict_point(cfg::NSMConfig, p, u0, t_eval, inputs;
                       solver=Vern9(), reltol=1e-8, abstol=1e-8)
    scfg = NSMConfig(cfg.n_s, cfg.n_m, cfg.n_u, cfg.n_h, cfg.n_layers,
                     cfg.s_cap, cfg.m_cap, inputs)
    f!(du, u, pp, t) = nsm_rhs!(du, u, pp, t, scfg)
    tspan = (float(first(t_eval)), float(last(t_eval)))
    prob  = ODEProblem(f!, collect(float.(u0)), tspan, p)
    sol   = solve(prob, solver; saveat=t_eval, reltol=reltol, abstol=abstol)
    return permutedims(reduce(hcat, sol.u))         # (n_time, n_obs)
end

"""
    predict_sample(cfg, post, u0, t_eval, inputs; n_sample=100, rng, ...)

Draw `n_sample` parameter sets from the posterior, integrate each, and return
an (n_sample × n_time × n_obs) array of trajectories.
"""
function predict_sample(cfg::NSMConfig, post::VarPosterior, u0, t_eval, inputs;
                        n_sample=100, rng=Random.default_rng(),
                        solver=Vern9(), reltol=1e-8, abstol=1e-8)
    ax = post.ax; d = length(post.mu); n_obs = cfg.n_s + cfg.n_m
    out = Array{Float64}(undef, n_sample, length(t_eval), n_obs)
    for k in 1:n_sample
        z = post.mu .+ exp.(post.log_s) .* randn(rng, d)
        p = ComponentArray(z, ax)
        out[k, :, :] = predict_point(cfg, p, u0, t_eval, inputs;
                                     solver=solver, reltol=reltol, abstol=abstol)
    end
    return out
end

"""
    credible_bands(ens; lo=0.05, hi=0.95)

Reduce an (n_sample × n_time × n_obs) ensemble to per-(time,output) summaries.
Returns a named tuple `(mean, lower, upper)`, each (n_time × n_obs).
"""
function credible_bands(ens::AbstractArray{<:Real,3}; lo=0.05, hi=0.95)
    _, nt, no = size(ens)
    mean_ = Array{Float64}(undef, nt, no)
    lower = Array{Float64}(undef, nt, no)
    upper = Array{Float64}(undef, nt, no)
    for t in 1:nt, j in 1:no
        col = @view ens[:, t, j]
        mean_[t, j] = mean(col)
        lower[t, j] = quantile(col, lo)
        upper[t, j] = quantile(col, hi)
    end
    return (mean=mean_, lower=lower, upper=upper)
end
