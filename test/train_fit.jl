# ============================================================================
#  Step-2 test:  loss value + gradient-through-solver + learning.
#
#     julia --project=. test/train_fit.jl
#
#  (A) RMSE at the eval params matches the JAX float64 reference.
#  (B) ForwardDiff gradients through the ODE solve match exact JAX gradients
#      (8 named parameters, sample 1).
#  (C) fit_rmse! reduces the loss (per-sample Adam), confirming the training
#      loop learns end to end.
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, ForwardDiff, Statistics, Random, Test

include("ref_params.jl")    # dims, caps, C_RAW/P_RAW/DM_RAW
include("train_data.jl")    # dataset, perturbed NN weights, grad + learning targets

# cfg.inputs is unused during training (each Sample carries its own input), so
# any placeholder is fine here.
cfg = NSMConfig(N_S, N_M, N_U, N_H, N_LAYERS, S_CAP, M_CAP, t -> zeros(N_U))

# eval params: CRM raw at reference, NN weights perturbed (matches gen_step2.py)
p = init_params(cfg)
p.C .= C_RAW; p.P .= P_RAW; p.d_m .= DM_RAW
p.Wi .= WI_E; p.bi .= BI_E; p.Wh .= WH_E; p.bh .= BH_E; p.Wo .= WO_E; p.bo .= BO_E

# assemble dataset
data = [Sample(DATA_U0[i, :], DATA_TARGET[i, :], DATA_TF[i], DATA_UIN[i, :])
        for i in 1:size(DATA_U0, 1)]

# ---- (A) loss value ----
r = rmse(cfg, p, data; solver=Tsit5(), reltol=1e-10, abstol=1e-10)
println("(A) rmse(z_eval) = $(round(r, digits=8))   ref = $(round(RMSE_REF, digits=8))")

# ---- (B) gradient through the solver vs exact JAX reference (sample 1) ----
ax = getaxes(p); z = getdata(p)
s1 = data[1]
gflat = ForwardDiff.gradient(
    zz -> sample_rmse(cfg, ComponentArray(zz, ax), s1.u0, s1.target, s1.tf, s1.uin;
                      solver=Tsit5(), reltol=1e-10, abstol=1e-10), z)
gca = ComponentArray(gflat, ax)
checks = [("C[1,1]",   gca.C[1,1],     GRAD_C_1_1),
          ("P[2,3]",   gca.P[2,3],     GRAD_P_2_3),
          ("d_m[1]",   gca.d_m[1],     GRAD_DM_1),
          ("Wi[1,1]",  gca.Wi[1,1],    GRAD_WI_1_1),
          ("bi[1]",    gca.bi[1],      GRAD_BI_1),
          ("Wh[1,1,1]",gca.Wh[1,1,1],  GRAD_WH_1_1_1),
          ("Wo[1,1]",  gca.Wo[1,1],    GRAD_WO_1_1),
          ("bo[1]",    gca.bo[1],      GRAD_BO_1)]
println("(B) gradient check (julia vs jax):")
for (nm, jv, rv) in checks
    println("    d/d$nm : $(round(jv, digits=7))   vs   $(round(rv, digits=7))")
end
gerr = maximum(abs(jv - rv) for (_, jv, rv) in checks)
println("    max abs gradient error = $gerr")

# ---- (C) learning: fit from z_eval, loss must drop ----
pfit = copy(p)
Random.seed!(1)
hist = fit_rmse!(cfg, pfit, data; lr=1e-2, max_epochs=60,
                 solver=Tsit5(), reltol=1e-7, abstol=1e-7, check_every=20)
println("(C) rmse: $(round(hist[1], digits=6)) -> $(round(hist[end], digits=6))")

@testset "step 2: loss, gradient, learning" begin
    @test abs(r - RMSE_REF) < 1e-5          # loss kernel + data pipeline correct
    @test gerr < 1e-4                        # autodiff through the solver correct
    @test hist[end] < hist[1]                # training reduces the loss
    @test hist[end] < 0.5 * hist[1]          # ...substantially
end
