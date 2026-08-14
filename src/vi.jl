# ============================================================================
#  Step 3b — variational posterior + EM + evidence
#  EXACT port of nsm/nsm.py :: fit_posterior, fit_posterior_EM, update_hypers,
#  approx_evidence  and  nsm/nsm_system.py :: T, log_abs_det.
#
#  Variational inference = minimize KL[q(theta|mu,sigma) || p(theta|D)].
#  By the identity  log p(D) = ELBO(mu,sigma) + KL[q||p]  with log p(D) constant
#  in (mu,sigma), this is IDENTICALLY maximizing the ELBO (paper Eq 14->15->16):
#      ELBO = E_q[log p(D|theta)] + E_q[log p(theta|alpha)] + H[q]
#           = -E_q[nll]          - E_q[(1/2) alpha (theta-theta_prior)^2] + sum(log sigma)
#  Reparameterization  z = mu + exp(log_s).*y,  y ~ N(0,I)      (== nsm_system.T)
#  Entropy term        sum(log_s)                               (== log_abs_det)
#
#  Convergence (== nsm.py::fit_posterior): every check_every epochs evaluate the
#  ELBO at the mean, fit a line to the last 10 (normalized) values, stop once
#  |slope| < tol for `patience` consecutive checks; retain the best-ELBO iterate.
# ============================================================================

using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff
using Random
using LinearAlgebra
using Statistics
using Zygote, SciMLSensitivity        # used only when ad == :reverse

mutable struct VarPosterior
    mu::Vector{Float64}
    log_s::Vector{Float64}
    ax
end

"""VarPosterior(p; log_s_init=log(0.05)) — uniform-std init (fallback)."""
function VarPosterior(p::ComponentArray; log_s_init=log(0.05))
    z = copy(getdata(p))
    VarPosterior(z, fill(float(log_s_init), length(z)), getaxes(p))
end

# per-parameter posterior init std, EXACTLY nsm.py::init_params `params_std`
# (field order matches init_params):  C,P = 1/n_s ; d_m & all biases = 1 ;
# Wi = 1/sqrt(n_x) ; Wh, Wo = 1/sqrt(n_h).
function init_std(cfg::NSMConfig; Tp::Type=Float64)
    n_s, n_m, n_h = cfg.n_s, cfg.n_m, cfg.n_h
    n_x = cfg.n_s + cfg.n_m + cfg.n_u
    n_o = cfg.n_s + cfg.n_m
    nl  = cfg.n_layers - 1
    ComponentArray(
        C   = fill(Tp(1/n_s),        n_m, n_s),
        P   = fill(Tp(1/n_s),        n_m, n_s),
        d_m = fill(Tp(1.0),          n_m),
        Wi  = fill(Tp(1/sqrt(n_x)),  n_h, n_x),
        bi  = fill(Tp(1.0),          n_h),
        Wh  = fill(Tp(1/sqrt(n_h)),  n_h, n_h, nl),
        bh  = fill(Tp(1.0),          n_h, nl),
        Wo  = fill(Tp(1/sqrt(n_h)),  n_o, n_h),
        bo  = fill(Tp(1.0),          n_o),
    )
end

"""VarPosterior(p, cfg) — init at mean `p` with nsm.py per-parameter std (log_s = log(std))."""
function VarPosterior(p::ComponentArray, cfg::NSMConfig)
    z = copy(getdata(p))
    VarPosterior(z, log.(getdata(init_std(cfg))), getaxes(p))
end

zmean(post::VarPosterior) = ComponentArray(post.mu, post.ax)

# negative ELBO at a fixed reparam draw y (flat) = nsm_system single-sample MC of Eq 16.
# z = mu + exp(log_s).*y ;  -ELBO = nlp(z) - sum(log_s).
function neg_elbo_flat(cfg::NSMConfig, lmbda, ax, d, data, prior, alpha, nu2, sigma2, y; kw...)
    mu    = @view lmbda[1:d]
    log_s = @view lmbda[d+1:2*d]
    z = mu .+ exp.(log_s) .* y
    p = ComponentArray(z, ax)
    return nlp(cfg, p, data, prior, alpha, nu2, sigma2; kw...) - sum(log_s)
end

"""Deterministic ELBO at the posterior mean: Σ log_s − nlp(mu)."""
function approx_evidence(cfg::NSMConfig, post::VarPosterior, data, prior, alpha, nu2, sigma2; kw...)
    p = ComponentArray(post.mu, post.ax)
    return sum(post.log_s) - nlp(cfg, p, data, prior, alpha, nu2, sigma2; kw...)
end

# utilities.py :: check_convergence — OLS slope of the normalized ELBO trace
function _elbo_slope(fv::AbstractVector{<:Real})
    f = filter(!isnan, fv)
    length(f) < 3 && return 1.0
    mx = maximum(abs.(f)); mx == 0 && return 1.0
    y = f ./ mx
    x = collect(0.0:length(y)-1)
    xm = mean(x); ym = mean(y)
    denom = sum((x .- xm).^2)
    denom == 0 && return 1.0
    return sum((x .- xm) .* (y .- ym)) / denom
end

"""
    fit_posterior!(cfg, post, data, prior; ad=:forward, alpha, nu2, sigma2,
                   lr=1e-3, max_epochs=100000, tol=1e-3, patience=5, ...)

Per-sample stochastic Adam on lmbda=[mu; log_s], EXACTLY as nsm.py::fit_posterior:
each epoch shuffles the data and takes one Adam step per sample using
`nll_i + log_prior/N - Σlog_s/N` with a fresh reparameterisation draw per sample
(prior & entropy split across the N samples so an epoch sums to the full-batch
gradient). Run to ELBO-slope convergence (keeps the best-ELBO iterate).
Maximizing this ELBO == minimizing KL[q||p]. Returns the ELBO history.
"""
function fit_posterior!(cfg::NSMConfig, post::VarPosterior, data, prior;
                        ad::Symbol=:forward,
                        alpha=1.0, nu2=1e-3, sigma2=1e-2,
                        lr=1e-3, max_epochs=100000, tol=1e-3, patience=5,
                        check_every=10, clip=1000.0,
                        solver=Tsit5(), reltol=1e-7, abstol=1e-7,
                        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                        verbose=true, rng=Random.default_rng())
    ax = post.ax; d = length(post.mu)
    lmbda = vcat(post.mu, post.log_s)
    m = zero(lmbda); v = zero(lmbda); t = 0
    sk      = (solver=solver, reltol=reltol, abstol=abstol)
    sk_grad = ad === :reverse ? (solver=solver, reltol=reltol, abstol=abstol, sensealg=sensealg) : sk
    N = length(data)
    # per-sample negative-ELBO contribution at a fresh draw y (== nsm.py per-sample SGD):
    #   nll_i  +  log_prior/N  -  sum(log_s)/N   (prior & entropy split across the N samples)
    _samp(l, s, y) = begin
        mu = @view l[1:d]; ls = @view l[d+1:2*d]
        p = ComponentArray(mu .+ exp.(ls) .* y, ax)
        nll_sample(cfg, p, s.u0, s.target, s.tf, s.uin, nu2, sigma2; sk_grad...) +
            log_prior(p, prior, alpha) / N - sum(ls) / N
    end
    f = Float64[]; best = copy(lmbda); passes = 0; fails = 0; epoch = 0
    while epoch <= max_epochs && passes < patience
        if epoch % check_every == 0
            post.mu .= @view lmbda[1:d]; post.log_s .= @view lmbda[d+1:2*d]
            ev = approx_evidence(cfg, post, data, prior, alpha, nu2, sigma2; sk...)
            push!(f, ev)
            slope = length(f) > 2 ? _elbo_slope(f[max(1, end-9):end]) : 1.0
            passes = abs(slope) < tol ? passes + 1 : 0
            ev >= maximum(f) && (best .= lmbda)                     # retain best-ELBO iterate
            if slope < 0 && length(f) >= 2 && f[end] < f[end-1] && epoch > 100
                fails += 1
            else
                fails = 0
            end
            if fails == patience                                    # diverging -> restore best
                lmbda .= best; break
            end
            verbose && println("epoch $epoch  ELBO $(round(ev, digits=3))  slope $(round(slope, digits=4))")
        end
        for s in shuffle(rng, data)                  # per-sample Adam (nsm.py fit_posterior)
            y = randn(rng, d)
            gfun(l) = _samp(l, s, y)
            g = ad === :reverse ? first(Zygote.gradient(gfun, lmbda)) : ForwardDiff.gradient(gfun, lmbda)
            t += 1; mc = 1 - 0.9^t; vc = 1 - 0.999^t
            @inbounds for i in eachindex(lmbda)
                gi = abs(g[i]) < clip ? g[i] : zero(eltype(g))
                m[i] = 0.9   * m[i] + 0.1   * gi
                v[i] = 0.999 * v[i] + 0.001 * gi^2
                lmbda[i] -= lr * (m[i] / mc) / (sqrt(v[i] / vc) + 1e-8)
            end
        end
        epoch += 1
    end
    post.mu .= @view lmbda[1:d]; post.log_s .= @view lmbda[d+1:2*d]
    return f
end

"""
    update_hypers!(cfg, post, data, prior; n_sample=100)

Empirical-Bayes EM step (paper Eqs 18-19).
  ν², σ²  — per-output least squares of  error² ≈ ν² + σ²·target²   (clipped pred,
            only outputs with target>0)                              (Eq 19)
  α       — zero-mean prior precision  α_k = 1 / E[θ_k²] = 1/(μ² + σ²)  (Eq 18)
"""
function update_hypers!(cfg::NSMConfig, post::VarPosterior, data, prior;
                        n_sample=100, solver=Tsit5(), reltol=1e-7, abstol=1e-7,
                        rng=Random.default_rng())
    ax = post.ax; d = length(post.mu); n_obs = cfg.n_s + cfg.n_m
    Y2 = [Float64[] for _ in 1:n_obs]; Z = [Float64[] for _ in 1:n_obs]
    for _ in 1:n_sample
        z = post.mu .+ exp.(post.log_s) .* randn(rng, d)
        p = ComponentArray(z, ax)
        for s in data
            pred = predict_endpoint(cfg, p, s.u0, s.tf, s.uin;
                                    solver=solver, reltol=reltol, abstol=abstol)
            (any(isnan, pred) || any(isinf, pred)) && continue
            predc = max.(pred, 0.0)                       # clip pred (nsm.py update_hypers)
            for j in 1:n_obs
                tj = s.target[j]
                if !isnan(tj) && tj > 0                   # only outputs with true>0
                    push!(Y2[j], tj^2); push!(Z[j], (tj - predc[j])^2)
                end
            end
        end
    end
    nu2 = fill(1e-4, n_obs); sigma2 = fill(1e-4, n_obs)
    for j in 1:n_obs
        isempty(Y2[j]) && continue
        B = hcat(ones(length(Y2[j])), Y2[j])
        a = (B' * B) \ (B' * Z[j])                        # [ν²_j; σ²_j]
        nu2[j]    = max(a[1], 1e-4)
        sigma2[j] = max(a[2], 1e-4)
    end
    # paper Eq 18 (zero-mean Gaussian prior): alpha_k = 1 / E[theta_k^2] = 1 / (mu^2 + sigma^2).
    # sigma^2 = exp(2 log_s) > 0 always, so the denominator is strictly positive (no floor needed).
    alpha = 1.0 ./ (post.mu .^ 2 .+ exp.(post.log_s) .^ 2)
    return nu2, sigma2, ComponentArray(alpha, ax)
end

"""
    fit_posterior_EM!(cfg, post, data, prior; max_iterations=10, patience=1, ...)

nsm.py::fit_posterior_EM. Fit the posterior to convergence, then alternate
update_hypers! and (converged) fit_posterior! until the evidence stops improving
(`patience` consecutive non-improvements) or `max_iterations`.
"""
function fit_posterior_EM!(cfg::NSMConfig, post::VarPosterior, data, prior;
                           alpha=1.0, nu2=1e-3, sigma2=1e-2,
                           max_iterations=10, patience=1, n_sample_hypers=100,
                           solver=Tsit5(), reltol=1e-7, abstol=1e-7,
                           verbose=true, kw...)
    ske = (solver=solver, reltol=reltol, abstol=abstol)
    verbose && println("EM: initial posterior...")
    fit_posterior!(cfg, post, data, prior; alpha=alpha, nu2=nu2, sigma2=sigma2,
                   solver=solver, reltol=reltol, abstol=abstol, verbose=verbose, kw...)

    ev = approx_evidence(cfg, post, data, prior, alpha, nu2, sigma2; ske...)
    best_ev = ev; best_mu = copy(post.mu); best_ls = copy(post.log_s)
    best_a  = alpha; best_nu2 = nu2; best_sig2 = sigma2
    prev = ev; fails = 0; it = 0; ev_hist = Float64[ev]
    verbose && println("EM init: log evidence $(round(ev, digits=3))")

    while fails <= patience && it < max_iterations
        it += 1
        verbose && println("EM iter $it: hyperparameters...")
        nu2, sigma2, alpha = update_hypers!(cfg, post, data, prior; n_sample=n_sample_hypers, ske...)
        verbose && println("EM iter $it: posterior...")
        fit_posterior!(cfg, post, data, prior; alpha=alpha, nu2=nu2, sigma2=sigma2,
                       solver=solver, reltol=reltol, abstol=abstol, verbose=verbose, kw...)
        ev = approx_evidence(cfg, post, data, prior, alpha, nu2, sigma2; ske...)   # recompute at the mean
        push!(ev_hist, ev)
        verbose && println("EM iter $it: log evidence $(round(ev, digits=3))")
        if ev > best_ev                                       # retain best-evidence configuration
            best_ev = ev; best_mu .= post.mu; best_ls .= post.log_s
            best_a = alpha; best_nu2 = nu2; best_sig2 = sigma2
        end
        (ev <= prev) ? (fails += 1) : (fails = 0)
        prev = max(prev, ev)                                  # nsm.py: previdence = max(previdence, log_evidence)
    end
    post.mu .= best_mu; post.log_s .= best_ls                 # restore the best posterior
    verbose && println("EM done: best log evidence $(round(best_ev, digits=3))")
    return (; alpha=best_a, nu2=best_nu2, sigma2=best_sig2, evidence=ev_hist, best_evidence=best_ev)
end
