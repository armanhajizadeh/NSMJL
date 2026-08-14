# ============================================================================
#  Step 3a — Bayesian objective  (paper Eqs 8-13; nsm/nsm_system.py kernels).
#
#  Negative log posterior:
#     nlp(z) = log_prior(z) + Σ_i  nll_i(z)
#
#     log_prior(z) = Σ  α (z - z_prior)^2 / 2                    (Gaussian prior)
#         paper Eq 13/16: prior is centered at ZERO for ALL parameters
#         ("a Gaussian prior centered at zero ... driving unnecessary
#          parameters to the prior mean"). z_prior = 0 for every parameter.
#
#     nll_i(z) = Σ_k [ err_k^2 / (2 var_k) + log(var_k)/2 ]      (heteroscedastic, Eq 8)
#         err  = target - endpoint     (NaN targets -> 0 error)
#         var  = ν² + σ² · max(pred,0)^2
#
#  Gradients flow through the ODE solve via ForwardDiff.
# ============================================================================

using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff

# prior mean vector: ZERO for every parameter (paper Eq 13/16, zero-mean Gaussian prior)
function prior_params(cfg::NSMConfig; Tp::Type=Float64)
    pr = init_params(cfg; Tp=Tp)
    pr .= zero(Tp)
    return pr
end

# Gaussian log-prior term  Σ α (z - z_prior)^2 / 2   (α scalar or per-param)
log_prior(p, prior, alpha) = sum(alpha .* (p .- prior) .^ 2) / 2

# heteroscedastic negative log-likelihood for one sample (Eq 8)
function nll_sample(cfg::NSMConfig, p, u0, target, tf, uin, nu2, sigma2; kw...)
    pred = predict_endpoint(cfg, p, u0, tf, uin; kw...)
    err  = map((tr, pr) -> isnan(tr) ? zero(pr) : tr - pr, target, pred)
    var  = nu2 .+ sigma2 .* max.(pred, zero(eltype(pred))) .^ 2
    return sum(err .^ 2 ./ var ./ 2 .+ log.(var) ./ 2)
end

# negative log posterior over the whole dataset
function nlp(cfg::NSMConfig, p, data::AbstractVector{Sample}, prior, alpha, nu2, sigma2; kw...)
    val = log_prior(p, prior, alpha)
    for s in data
        val += nll_sample(cfg, p, s.u0, s.target, s.tf, s.uin, nu2, sigma2; kw...)
    end
    return val
end

# gradient of nlp w.r.t. parameters (flat), via ForwardDiff through the solves
function grad_nlp(cfg::NSMConfig, p, data::AbstractVector{Sample}, prior, alpha, nu2, sigma2; kw...)
    ax = getaxes(p)
    z  = getdata(p)
    gflat = ForwardDiff.gradient(
        zz -> nlp(cfg, ComponentArray(zz, ax), data, prior, alpha, nu2, sigma2; kw...), z)
    return ComponentArray(gflat, ax)
end

"""
    fit_map!(cfg, p, data, prior; alpha, nu2, sigma2, lr, max_epochs, rtol, patience, ...)

Full-batch Adam minimisation of `nlp` (the MAP / posterior-mode estimate;
mutates `p`). Stops on a convergence threshold: halts once the relative
improvement in `nlp` stays below `rtol` for `patience` consecutive checks, or at
`max_epochs`. Returns the nlp history.
"""
function fit_map!(cfg::NSMConfig, p, data::AbstractVector{Sample}, prior;
                  alpha=1.0, nu2=1e-3, sigma2=1e-2,
                  lr=1e-2, max_epochs=2000, rtol=1e-4, patience=3, check_every=10,
                  clip=1000.0, solver=Tsit5(), reltol=1e-7, abstol=1e-7,
                  verbose=true)
    ax = getaxes(p); z = getdata(p)
    m = zero(z); v = zero(z); t = 0
    hist = Float64[]; prev = Inf; stall = 0; epoch = 0
    obj(zz) = nlp(cfg, ComponentArray(zz, ax), data, prior, alpha, nu2, sigma2;
                  solver=solver, reltol=reltol, abstol=abstol)
    while epoch <= max_epochs
        if epoch % check_every == 0
            val = obj(z); push!(hist, val)
            verbose && println("epoch $epoch  nlp $(round(val, digits=4))")
            if abs(prev - val) / (abs(prev) + 1e-8) < rtol
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
