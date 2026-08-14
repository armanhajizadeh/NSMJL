# ============================================================================
#  Step-3b test:  variational posterior + EM + evidence.
#
#     julia --project=. test/vi_fit.jl
#
#  (A) evidence (Σlog_s − nlp(mu)) and neg-ELBO at a FIXED draw match JAX.
#  (B) gradient of neg-ELBO w.r.t. [mu; log_s] matches exact JAX gradients.
#  (C) alpha EM update matches the closed-form reference.
#  (D) fit_posterior! increases the evidence; update_hypers! returns valid hypers.
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, ForwardDiff, Statistics, Random, Test

include("ref_params.jl")
include("train_data.jl")
include("bayes_data.jl")     # ALPHA0, NU2, SIGMA2, PRIOR_LLIM
include("vi_data.jl")        # LOGS_VAL, EVIDENCE_REF, NEGELBO_REF, grads, alpha, Y_*

cfg = NSMConfig(N_S, N_M, N_U, N_H, N_LAYERS, S_CAP, M_CAP, t -> zeros(N_U))

# posterior mean = z_eval
p = init_params(cfg)
p.C .= C_RAW; p.P .= P_RAW; p.d_m .= DM_RAW
p.Wi .= WI_E; p.bi .= BI_E; p.Wh .= WH_E; p.bh .= BH_E; p.Wo .= WO_E; p.bo .= BO_E

data  = [Sample(DATA_U0[i, :], DATA_TARGET[i, :], DATA_TF[i], DATA_UIN[i, :])
         for i in 1:size(DATA_U0, 1)]
prior = prior_params(cfg; llim=PRIOR_LLIM)

post = VarPosterior(p; log_s_init=LOGS_VAL)
ax = post.ax; d = length(post.mu)

# fixed reparameterisation draw y (same per-component values as JAX)
yca = init_params(cfg)
yca.C .= Y_C; yca.P .= Y_P; yca.d_m .= Y_DM
yca.Wi .= Y_WI; yca.bi .= Y_BI; yca.Wh .= Y_WH; yca.bh .= Y_BH; yca.Wo .= Y_WO; yca.bo .= Y_BO
yflat = copy(getdata(yca))

tol = (solver=Tsit5(), reltol=1e-11, abstol=1e-11)

# ---- (A) evidence + neg-ELBO at fixed y ----
ev  = approx_evidence(cfg, post, data, prior, ALPHA0, NU2, SIGMA2; tol...)
lmbda = vcat(post.mu, post.log_s)
nel = neg_elbo_flat(cfg, lmbda, ax, d, data, prior, ALPHA0, NU2, SIGMA2, yflat; tol...)
println("(A) evidence  = $(round(ev, digits=5))   ref = $(round(EVIDENCE_REF, digits=5))")
println("    neg_elbo  = $(round(nel, digits=5))   ref = $(round(NEGELBO_REF, digits=5))")

# ---- (B) gradient of neg-ELBO wrt [mu; log_s] at fixed y ----
gl = ForwardDiff.gradient(
        l -> neg_elbo_flat(cfg, l, ax, d, data, prior, ALPHA0, NU2, SIGMA2, yflat; tol...), lmbda)
gmu = ComponentArray(gl[1:d],     ax)
gls = ComponentArray(gl[d+1:2d],  ax)
checks = [("mu:C[1,1]",  gmu.C[1,1],  GELBO_MU_C_1_1),
          ("ls:C[1,1]",  gls.C[1,1],  GELBO_LS_C_1_1),
          ("mu:bo[1]",   gmu.bo[1],   GELBO_MU_BO_1),
          ("ls:bo[1]",   gls.bo[1],   GELBO_LS_BO_1)]
println("(B) grad neg_elbo (julia vs jax):")
for (nm, jv, rv) in checks
    println("    d/d$nm : $(round(jv, digits=5))   vs   $(round(rv, digits=5))")
end
grel = maximum(abs(jv - rv) / (abs(rv) + 1e-3) for (_, jv, rv) in checks)

# ---- (C) alpha EM update (deterministic) ----
_, _, alpha_ca = update_hypers!(cfg, post, data, prior; n_sample=1)  # alpha is deterministic in mu,log_s
aC = alpha_ca.C[1,1]; aBO = alpha_ca.bo[1]
println("(C) alpha C[1,1] = $(round(aC, digits=5)) vs $(round(ALPHA_C_1_1, digits=5)); " *
        "bo[1] = $(round(aBO, digits=5)) vs $(round(ALPHA_BO_1, digits=5))")

# ---- (D) fit_posterior! raises evidence; update_hypers! gives valid hypers ----
Random.seed!(0)
hist = fit_posterior!(cfg, post, data, prior; alpha=ALPHA0, nu2=NU2, sigma2=SIGMA2,
                      lr=5e-3, epochs=40, solver=Tsit5(), reltol=1e-7, abstol=1e-7, print_every=20)
nu2h, sig2h, _ = update_hypers!(cfg, post, data, prior; n_sample=30)
println("(D) evidence $(round(hist[1], digits=3)) -> $(round(hist[end], digits=3))")
println("    nu2 (min/max)    = $(round(minimum(nu2h), digits=5)) / $(round(maximum(nu2h), digits=5))")
println("    sigma2 (min/max) = $(round(minimum(sig2h), digits=5)) / $(round(maximum(sig2h), digits=5))")

@testset "step 3b: VI, ELBO, evidence, EM" begin
    @test abs(ev - EVIDENCE_REF) < 1e-2       # evidence matches JAX
    @test abs(nel - NEGELBO_REF) < 1e-2       # reparameterised ELBO matches JAX
    @test grel < 1e-3                          # ELBO gradient w.r.t. lmbda correct
    @test abs(aC - ALPHA_C_1_1) < 1e-5         # alpha update closed form
    @test abs(aBO - ALPHA_BO_1) < 1e-4
    @test hist[end] > hist[1]                  # posterior optimisation raises evidence
    @test all(nu2h .> 0) && all(sig2h .> 0)    # EM noise fit valid
    @test all(isfinite, nu2h) && all(isfinite, sig2h)
end
