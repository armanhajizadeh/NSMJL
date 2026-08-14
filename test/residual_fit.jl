# ============================================================================
#  Step-5 test:  gLV residual-penalty (PINN) variant.
#
#     julia --project=. test/residual_fit.jl
#
#  (A) combined loss (and its data / residual parts) matches JAX.
#  (B) gradient of the combined loss matches exact JAX gradients (8 params).
#  (C) fit_residual! reduces the loss.
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, ForwardDiff, Statistics, Test

include("ref_params.jl")      # N_U, N_H, N_LAYERS
include("train_data.jl")      # dataset
include("residual_data.jl")   # LAM5, NX5, refs + params

cfg = BBConfig(NX5, N_U, N_H, N_LAYERS, t -> zeros(N_U))

p = bb_init(NX5, N_U, N_H, N_LAYERS)
p.Wi .= WI5; p.bi .= BI5; p.Wh .= WH5; p.bh .= BH5; p.Wo .= WO5; p.bo .= BO5
p.r .= R5; p.A .= A5

data = [Sample(DATA_U0[i, :], DATA_TARGET[i, :], DATA_TF[i], DATA_UIN[i, :])
        for i in 1:size(DATA_U0, 1)]

tol = (solver=Tsit5(), reltol=1e-11, abstol=1e-11)

# ---- (A) loss value and parts ----
Lval = residual_loss(cfg, p, data; lambda=LAM5, tol...)
Dval = bb_data_rmse(cfg, p, data; tol...)
Rval = bb_residual(cfg, p, data)
println("(A) loss=$(round(Lval,6)) ref=$(round(LOSS5_REF,6)) | data=$(round(Dval,6)) ref=$(round(DATA5_REF,6)) | res=$(round(Rval,6)) ref=$(round(RES5_REF,6))")

# ---- (B) gradient vs exact JAX ----
ax = getaxes(p); z = getdata(p)
gflat = ForwardDiff.gradient(
    zz -> residual_loss(cfg, ComponentArray(zz, ax), data; lambda=LAM5, tol...), z)
gca = ComponentArray(gflat, ax)
checks = [("Wi[1,1]",  gca.Wi[1,1],   G5_WI_1_1),
          ("bi[1]",    gca.bi[1],     G5_BI_1),
          ("Wh[1,1,1]",gca.Wh[1,1,1], G5_WH_1_1_1),
          ("bh[1,1]",  gca.bh[1,1],   G5_BH_1_1),
          ("Wo[1,1]",  gca.Wo[1,1],   G5_WO_1_1),
          ("bo[1]",    gca.bo[1],     G5_BO_1),
          ("r[1]",     gca.r[1],      G5_R_1),
          ("A[1,1]",   gca.A[1,1],    G5_A_1_1)]
println("(B) grad residual_loss (julia vs jax):")
for (nm, jv, rv) in checks
    println("    d/d$nm : $(round(jv, digits=6))   vs   $(round(rv, digits=6))")
end
grel = maximum(abs(jv - rv) / (abs(rv) + 1e-4) for (_, jv, rv) in checks)

# ---- (C) learning ----
pfit = copy(p)
hist = fit_residual!(cfg, pfit, data; lambda=LAM5, lr=5e-3, epochs=60,
                     solver=Tsit5(), reltol=1e-7, abstol=1e-7, print_every=20)
println("(C) loss $(round(hist[1],5)) -> $(round(hist[end],5))")

@testset "step 5: gLV residual PINN" begin
    @test abs(Lval - LOSS5_REF) < 1e-4
    @test abs(Dval - DATA5_REF) < 1e-4
    @test abs(Rval - RES5_REF)  < 1e-6
    @test grel < 1e-3
    @test hist[end] < hist[1]
    @test hist[end] < 0.8 * hist[1]
end
