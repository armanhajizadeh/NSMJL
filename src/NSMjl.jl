module NSMjl

using ComponentArrays
using OrdinaryDiffEq
using ForwardDiff
using Random
using Statistics
using LinearAlgebra

include("system.jl")
include("train.jl")
include("bayes.jl")
include("vi.jl")
include("predict.jl")
include("residual.jl")
include("dataio.jl")

export NSMConfig, init_params, nsm_rhs!, solve_forward, g, g_inv       # step 1
export Sample, predict_endpoint, sample_rmse, rmse, fit_rmse!          # step 2
export prior_params, log_prior, nll_sample, nlp, grad_nlp, fit_map!    # step 3a
export VarPosterior, zmean, neg_elbo_flat, approx_evidence,            # step 3b
       fit_posterior!, update_hypers!, fit_posterior_EM!
export predict_point, predict_sample, credible_bands                 # step 4
export BBConfig, bb_init, bb_deriv, glv_deriv, bb_endpoint, bb_data_rmse,       # step 5
       bb_residual, residual_loss, fit_residual!, bb_predict_point
export load_nsm_csv                                                  # real data

"""
    solve_forward(cfg, p, u0, tspan; saveat, solver=Vern9(), reltol=1e-8, abstol=1e-8)

Integrate the NSM system forward using `cfg.inputs(t)` as the (time-dependent)
perturbation. Returns the ODESolution.
"""
function solve_forward(cfg::NSMConfig, p, u0, tspan;
                       saveat, solver=Vern9(), reltol=1e-8, abstol=1e-8)
    f!(du, u, pp, t) = nsm_rhs!(du, u, pp, t, cfg)
    prob = ODEProblem(f!, u0, tspan, p)
    solve(prob, solver; saveat=saveat, reltol=reltol, abstol=abstol)
end

end # module
