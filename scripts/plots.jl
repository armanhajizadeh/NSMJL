# ============================================================================
#  Makie validation plots.  Run from the package root:
#     julia --project=. scripts/plots.jl
#  Writes PNGs into figures/.
#
#  fig1_posterior_bands.pdf    embedded NSM posterior predictive (mean + 90%)
#  fig2_perturbation.pdf       trajectory with vs without a time-dependent u(t)
#  fig3_embedded_vs_residual.pdf   embedded NSM vs gLV-residual PINN (Option 1)
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, Statistics, Random
using CairoMakie

const TESTDIR = joinpath(@__DIR__, "..", "test")
include(joinpath(TESTDIR, "ref_params.jl"))
include(joinpath(TESTDIR, "train_data.jl"))
include(joinpath(TESTDIR, "vi_data.jl"))
include(joinpath(TESTDIR, "residual_data.jl"))

figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
labels = ["s1","s2","s3","s4","m1","m2","m3"]

# ---- embedded model: posterior at mean z_eval, log_s = LOGS_VAL ----
cfg = NSMConfig(N_S, N_M, N_U, N_H, N_LAYERS, S_CAP, M_CAP, t -> zeros(N_U))
p = init_params(cfg)
p.C .= C_RAW; p.P .= P_RAW; p.d_m .= DM_RAW
p.Wi .= WI_E; p.bi .= BI_E; p.Wh .= WH_E; p.bh .= BH_E; p.Wo .= WO_E; p.bo .= BO_E
post = VarPosterior(p; log_s_init=LOGS_VAL)

u0t = [0.35, 0.45, 0.25, 0.40, 0.9, 0.5, 0.7]
te  = collect(range(0, 24; length=49))
static_u = t -> [0.5, 0.2]

Random.seed!(7)
ens   = predict_sample(cfg, post, u0t, te, static_u; n_sample=80,
                       solver=Tsit5(), reltol=1e-7, abstol=1e-7)
bands = credible_bands(ens; lo=0.05, hi=0.95)

# ---- Figure 1: posterior predictive bands ----
fig1 = Figure(size=(1300, 640))
for j in 1:7
    ax = Axis(fig1[fldmod1(j,4)...], title=labels[j], xlabel = j>3 ? "time" : "")
    band!(ax, te, bands.lower[:,j], bands.upper[:,j], color=(:dodgerblue, 0.25))
    lines!(ax, te, bands.mean[:,j], color=:dodgerblue, linewidth=2)
end
Label(fig1[0, :], "Embedded NSM — posterior predictive (mean + 90% band)", fontsize=15)
save(joinpath(figdir, "fig1_posterior_bands.pdf"), fig1)

# ---- Figure 2: time-dependent perturbation effect ----
pulse_u = t -> [exp(-((t - 6.0)^2) / (2 * 1.5^2)), 0.2]
tj_static = predict_point(cfg, p, u0t, te, static_u; solver=Vern9(), reltol=1e-9, abstol=1e-9)
tj_pulse  = predict_point(cfg, p, u0t, te, pulse_u;  solver=Vern9(), reltol=1e-9, abstol=1e-9)
fig2 = Figure(size=(1300, 640))
for j in 1:7
    ax = Axis(fig2[fldmod1(j,4)...], title=labels[j], xlabel = j>3 ? "time" : "")
    lines!(ax, te, tj_static[:,j], color=:gray,   linewidth=2, label="no perturbation")
    lines!(ax, te, tj_pulse[:,j],  color=:crimson, linewidth=2, label="u(t) pulse @ t=6")
    j == 1 && axislegend(ax; position=:rt, framevisible=false)
end
Label(fig2[0, :], "Time-dependent perturbation u(t) bends the trajectory", fontsize=15)
save(joinpath(figdir, "fig2_perturbation.pdf"), fig2)

# ---- residual (black-box + gLV) model: load params, fit, integrate ----
bcfg = BBConfig(NX5, N_U, N_H, N_LAYERS, t -> zeros(N_U))
pr = bb_init(NX5, N_U, N_H, N_LAYERS)
pr.Wi .= WI5; pr.bi .= BI5; pr.Wh .= WH5; pr.bh .= BH5; pr.Wo .= WO5; pr.bo .= BO5
pr.r .= R5; pr.A .= A5
data = [Sample(DATA_U0[i,:], DATA_TARGET[i,:], DATA_TF[i], DATA_UIN[i,:]) for i in 1:size(DATA_U0,1)]
fit_residual!(bcfg, pr, data; lambda=LAM5, lr=5e-3, max_epochs=1000,
              check_every=10, rtol=1e-4, patience=3,
              solver=Tsit5(), reltol=1e-7, abstol=1e-7)
tj_res = bb_predict_point(bcfg, pr, u0t, te, static_u; solver=Vern9(), reltol=1e-9, abstol=1e-9)

# ---- Figure 3: embedded vs residual ----
fig3 = Figure(size=(1300, 640))
for j in 1:7
    ax = Axis(fig3[fldmod1(j,4)...], title=labels[j], xlabel = j>3 ? "time" : "")
    lines!(ax, te, tj_static[:,j], color=:dodgerblue, linewidth=2, label="embedded NSM")
    lines!(ax, te, tj_res[:,j],    color=:orange, linewidth=2, linestyle=:dash, label="gLV-residual PINN")
    j == 1 && axislegend(ax; position=:rt, framevisible=false)
end
Label(fig3[0, :], "Embedded NSM vs gLV-residual PINN (Option 1) — same data, same solver", fontsize=15)
save(joinpath(figdir, "fig3_embedded_vs_residual.pdf"), fig3)

println("wrote figures to $figdir")
