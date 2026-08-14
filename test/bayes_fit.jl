# ============================================================================
#  Step-3a test:  Bayesian objective + MAP fit.
#
#     julia --project=. test/bayes_fit.jl
#
#  (A) log_prior and nlp match the JAX float64 reference.
#  (B) grad_nlp (autodiff through the solves) matches exact JAX gradients.
#  (C) fit_map! reduces nlp and RMSE.
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, ForwardDiff, Statistics, Random, Test

include("ref_params.jl")    # dims, caps, C_RAW/P_RAW/DM_RAW
include("train_data.jl")    # dataset + perturbed NN weights (z_eval)
include("bayes_data.jl")    # nlp/grad/MAP reference targets

cfg = NSMConfig(N_S, N_M, N_U, N_H, N_LAYERS, S_CAP, M_CAP, t -> zeros(N_U))

# eval params z_eval (same as step 2)
p = init_params(cfg)
p.C .= C_RAW; p.P .= P_RAW; p.d_m .= DM_RAW
p.Wi .= WI_E; p.bi .= BI_E; p.Wh .= WH_E; p.bh .= BH_E; p.Wo .= WO_E; p.bo .= BO_E

data  = [Sample(DATA_U0[i, :], DATA_TARGET[i, :], DATA_TF[i], DATA_UIN[i, :])
         for i in 1:size(DATA_U0, 1)]
prior = prior_params(cfg; llim=PRIOR_LLIM)

tol = (solver=Tsit5(), reltol=1e-11, abstol=1e-11)

# ---- (A) objective values ----
lp  = log_prior(p, prior, ALPHA0)
val = nlp(cfg, p, data, prior, ALPHA0, NU2, SIGMA2; tol...)
println("(A) log_prior = $(round(lp, digits=6))   ref = $(round(LOGPRIOR_REF, digits=6))")
println("    nlp       = $(round(val, digits=6))   ref = $(round(NLP_REF, digits=6))")

# ---- (B) gradient of nlp vs exact JAX ----
gca = grad_nlp(cfg, p, data, prior, ALPHA0, NU2, SIGMA2; tol...)
checks = [("C[1,1]",   gca.C[1,1],     GNLP_C_1_1),
          ("P[2,3]",   gca.P[2,3],     GNLP_P_2_3),
          ("d_m[1]",   gca.d_m[1],     GNLP_DM_1),
          ("Wi[1,1]",  gca.Wi[1,1],    GNLP_WI_1_1),
          ("bi[1]",    gca.bi[1],      GNLP_BI_1),
          ("Wh[1,1,1]",gca.Wh[1,1,1],  GNLP_WH_1_1_1),
          ("Wo[1,1]",  gca.Wo[1,1],    GNLP_WO_1_1),
          ("bo[1]",    gca.bo[1],      GNLP_BO_1)]
println("(B) grad nlp (julia vs jax):")
for (nm, jv, rv) in checks
    println("    d/d$nm : $(round(jv, digits=5))   vs   $(round(rv, digits=5))")
end
grel = maximum(abs(jv - rv) / (abs(rv) + 1e-3) for (_, jv, rv) in checks)
println("    max relative gradient error = $grel")

# ---- (C) MAP fit ----
pfit = copy(p)
rmse0 = rmse(cfg, pfit, data; tol...)
histmap = fit_map!(cfg, pfit, data, prior; alpha=ALPHA0, nu2=NU2, sigma2=SIGMA2,
                   lr=1e-2, max_epochs=60, solver=Tsit5(), reltol=1e-7, abstol=1e-7,
                   check_every=20)
rmseF = rmse(cfg, pfit, data; tol...)
println("(C) nlp  $(round(histmap[1], digits=3)) -> $(round(histmap[end], digits=3))")
println("    rmse $(round(rmse0, digits=5)) -> $(round(rmseF, digits=5))")

@testset "step 3a: prior, nlp, grad, MAP" begin
    @test abs(lp - LOGPRIOR_REF) < 1e-6      # log_prior is exact param arithmetic
    @test abs(val - NLP_REF)     < 1e-2      # ODE-dependent
    @test grel < 1e-3                         # autodiff of nlp through the solver
    @test histmap[end] < histmap[1]           # MAP reduces nlp
    @test rmseF < rmse0                        # ...and improves the fit
end
