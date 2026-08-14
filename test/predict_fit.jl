# ============================================================================
#  Step-4 test:  prediction with uncertainty.
#
#     julia --project=. test/predict_fit.jl
#
#  (A) predict_point at the posterior MEAN matches the JAX trajectory.
#  (B) predict_point at a fixed posterior DRAW matches the JAX trajectory.
#  (C) a time-dependent perturbation u(t) changes the trajectory (the feature).
#  (D) predict_sample + credible_bands: ordered bands, mean ≈ point forecast.
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, Statistics, Random, Test

include("ref_params.jl")
include("train_data.jl")
include("vi_data.jl")          # LOGS_VAL, Y_* (fixed draw)
include("predict_data.jl")     # U0_TEST, T_EVAL, UIN_STATIC, TRAJ_MEAN, TRAJ_SAMPLE1

cfg = NSMConfig(N_S, N_M, N_U, N_H, N_LAYERS, S_CAP, M_CAP, t -> zeros(N_U))

# posterior mean = z_eval
p = init_params(cfg)
p.C .= C_RAW; p.P .= P_RAW; p.d_m .= DM_RAW
p.Wi .= WI_E; p.bi .= BI_E; p.Wh .= WH_E; p.bh .= BH_E; p.Wo .= WO_E; p.bo .= BO_E
post = VarPosterior(p; log_s_init=LOGS_VAL)
ax = post.ax; d = length(post.mu)

# fixed reparameterisation draw y (same as JAX)
yca = init_params(cfg)
yca.C .= Y_C; yca.P .= Y_P; yca.d_m .= Y_DM
yca.Wi .= Y_WI; yca.bi .= Y_BI; yca.Wh .= Y_WH; yca.bh .= Y_BH; yca.Wo .= Y_WO; yca.bo .= Y_BO
yflat = copy(getdata(yca))

static_u = t -> UIN_STATIC
tol = (solver=Vern9(), reltol=1e-10, abstol=1e-10)

# ---- (A) point forecast at posterior mean ----
traj_mean = predict_point(cfg, p, U0_TEST, T_EVAL, static_u; tol...)
errA = maximum(abs.(traj_mean .- TRAJ_MEAN))
println("(A) predict_point(mean) max abs error vs JAX = $errA")

# ---- (B) forecast at a fixed posterior draw ----
zs = post.mu .+ exp.(post.log_s) .* yflat
ps = ComponentArray(zs, ax)
traj_s = predict_point(cfg, ps, U0_TEST, T_EVAL, static_u; tol...)
errB = maximum(abs.(traj_s .- TRAJ_SAMPLE1))
println("(B) predict_point(sample) max abs error vs JAX = $errB")

# ---- (C) time-dependent perturbation changes the trajectory ----
pulse_u = t -> [exp(-((t - 6.0)^2) / (2 * 1.5^2)), 0.2]   # Gaussian antibiotic course
traj_pulse = predict_point(cfg, p, U0_TEST, T_EVAL, pulse_u; tol...)
pert_effect = maximum(abs.(traj_pulse .- traj_mean))
println("(C) max trajectory change from u(t) perturbation = $(round(pert_effect, digits=5))")

# ---- (D) posterior ensemble + credible bands ----
Random.seed!(0)
ens = predict_sample(cfg, post, U0_TEST, T_EVAL, static_u; n_sample=100,
                     solver=Tsit5(), reltol=1e-7, abstol=1e-7)
bands = credible_bands(ens; lo=0.05, hi=0.95)
ordered = all(bands.lower .<= bands.mean .+ 1e-9) && all(bands.mean .<= bands.upper .+ 1e-9)
mean_gap = maximum(abs.(bands.mean .- traj_mean))
println("(D) bands ordered = $ordered ; |ensemble mean - point| max = $(round(mean_gap, digits=4))")
println("    mean band width (final row) = $(round(mean(bands.upper[end,:] .- bands.lower[end,:]), digits=4))")

@testset "step 4: prediction + uncertainty" begin
    @test size(traj_mean) == size(TRAJ_MEAN)
    @test errA < 1e-5                       # point forecast matches JAX
    @test errB < 1e-5                       # sampled forecast matches JAX
    @test pert_effect > 1e-3                # u(t) perturbation has a real effect
    @test ordered                           # lower ≤ mean ≤ upper everywhere
    @test mean_gap < 0.05                   # ensemble mean ≈ point forecast
    @test all(bands.upper .>= bands.lower)  # non-negative band widths
end
