# ============================================================================
#  Real-data experiment: embedded NSM vs gLV-residual PINN on Cdiff.
#     julia --project=. scripts/experiment_realdata.jl
#
#  Fits both models on the Cdiff training fold and scores them on the HELD-OUT
#  test fold (ground truth). Reports:
#    - held-out RMSE and relative RMSE (||err||/||true||)
#    - per-output Pearson r (the paper's metric, Fig 2/3)
#    - fraction of unphysical (negative) predictions
#  Writes:
#    figures/fig4_realdata_groundtruth.pdf   pooled predicted-vs-observed
#    figures/fig5_pred_vs_obs_embedded.pdf   per-output panels (embedded)
#    figures/fig6_correlation_bars.pdf       per-output Pearson r, both models
#
#  Set N_SUB = nothing to use all 399 training samples (slower but full).
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, Statistics, Random
using CairoMakie

const DATADIR = joinpath(@__DIR__, "..", "data")
species   = ["BT","BV","CH","BU","CS","CA","DP","DSM"]
mediators = ["MS001","MS008","MS014"]
inputs    = String[]
labels    = vcat(species, mediators)

train, dims, scales = load_nsm_csv(joinpath(DATADIR, "cdiff_train_0.csv"),
                                   species, mediators, inputs)
ss, ms = scales
test, _, _ = load_nsm_csv(joinpath(DATADIR, "cdiff_test_0.csv"),
                          species, mediators, inputs; s_scale=ss, m_scale=ms)
n_s, n_m, n_u = dims
println("loaded: $(length(train)) train samples, $(length(test)) held-out; dims=$(dims)")

N_SUB = 120
Random.seed!(0)
fit_set = N_SUB === nothing ? train : train[randperm(length(train))[1:N_SUB]]

n_h, n_layers = 12, 2
s_cap = fill(1.5, n_s); m_cap = fill(1.5, n_m)
sk = (solver=Tsit5(), reltol=1e-6, abstol=1e-6)

# ---- fit both models ----
cfg = NSMConfig(n_s, n_m, n_u, n_h, n_layers, s_cap, m_cap, t -> Float64[])
pe  = init_params(cfg)
println("fitting embedded NSM..."); fit_rmse!(cfg, pe, fit_set; lr=5e-3, max_epochs=1000, check_every=10, rtol=1e-4, patience=3, sk...)

bcfg = BBConfig(n_s + n_m, n_u, n_h, n_layers, t -> Float64[])
pr   = bb_init(n_s + n_m, n_u, n_h, n_layers)
println("fitting gLV-residual PINN..."); fit_residual!(bcfg, pr, fit_set; lambda=0.5, lr=5e-3, max_epochs=1000, check_every=10, rtol=1e-4, patience=3, sk...)

# ---- collect per-output predictions on held-out data ----
unscale(v) = vcat(v[1:n_s] ./ ss, v[n_s+1:end] ./ ms)
function collect_preds(predict_fn, data)
    no = n_s + n_m
    O = [Float64[] for _ in 1:no]; P = [Float64[] for _ in 1:no]
    neg = 0; tot = 0; rmses = Float64[]
    for s in data
        q  = predict_fn(s)
        pu = unscale(q); tu = unscale(s.target)
        e  = [isnan(t) ? 0.0 : t - p for (t, p) in zip(tu, pu)]
        push!(rmses, sqrt(mean(abs2, e)))
        neg += count(<(-1e-6), q); tot += length(q)
        for j in 1:no
            isnan(tu[j]) || (push!(O[j], tu[j]); push!(P[j], pu[j]))
        end
    end
    return O, P, mean(rmses), 100neg/tot
end
emb_pred(s) = predict_endpoint(cfg,  pe, s.u0, s.tf, s.uin; sk...)
res_pred(s) = bb_endpoint(bcfg, pr, s.u0, s.tf, s.uin; sk...)

pearson(o, p) = (length(o) < 3 || std(o) < 1e-9 || std(p) < 1e-9) ? NaN : cor(o, p)
function rrmse(O, P)   # ||err|| / ||true|| pooled over all outputs
    num = sum(sum(abs2, O[j] .- P[j]) for j in eachindex(O))
    den = sum(sum(abs2, O[j])          for j in eachindex(O))
    return sqrt(num / den)
end

Oe, Pe, rmse_e, neg_e = collect_preds(emb_pred, test)
Or, Pr, rmse_r, neg_r = collect_preds(res_pred, test)
re = [pearson(Oe[j], Pe[j]) for j in eachindex(labels)]
rr = [pearson(Or[j], Pr[j]) for j in eachindex(labels)]

println("\n=== GROUND-TRUTH COMPARISON (Cdiff test_0, $(length(test)) held-out) ===")
println("Embedded NSM        RMSE $(round(rmse_e,digits=3))  rRMSE $(round(rrmse(Oe,Pe),digits=3))  mean r $(round(mean(filter(!isnan,re)),digits=3))  neg $(round(neg_e,digits=1))%")
println("gLV-residual PINN   RMSE $(round(rmse_r,digits=3))  rRMSE $(round(rrmse(Or,Pr),digits=3))  mean r $(round(mean(filter(!isnan,rr)),digits=3))  neg $(round(neg_r,digits=1))%")

# ---- figures ----
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)

# fig5: per-output predicted vs observed (embedded), paper Fig 2/3 style
fig5 = Figure(size=(1300, 950))
for j in 1:length(labels)
    row, col = fldmod1(j, 4)
    o = Oe[j]; p = Pe[j]; rj = pearson(o, p)
    ax = Axis(fig5[row, col], title="$(labels[j])   r=$(isnan(rj) ? "NA" : string(round(rj,digits=2)))",
              xlabel = row == 3 ? "measured" : "", ylabel = col == 1 ? "predicted" : "")
    if !isempty(o)
        lo = min(0.0, minimum(o), minimum(p)) - 0.02
        hi = max(maximum(o), maximum(p), 0.02) + 0.02
        scatter!(ax, o, p, markersize=7, color=(:dodgerblue, 0.6))
        lines!(ax, [lo, hi], [lo, hi], color=:black, linestyle=:dash)
        hlines!(ax, [0.0], color=:red, linewidth=0.6)
        limits!(ax, lo, hi, lo, hi)
    end
end
Label(fig5[0, :], "Embedded NSM - predicted vs measured per output (Cdiff held-out)", fontsize=14)
save(joinpath(figdir, "fig5_pred_vs_obs_embedded.pdf"), fig5)

# fig6: per-output Pearson r bar chart, both models
fig6 = Figure(size=(1250, 520))
ax = Axis(fig6[1, 1], xticks=(1:length(labels), labels), xticklabelrotation=pi/4,
          ylabel="Pearson r (predicted vs measured)",
          title="Per-output prediction correlation on held-out Cdiff")
barplot!(ax, (1:length(labels)) .- 0.2, replace(re, NaN => 0.0), width=0.4, color=:dodgerblue, label="Embedded NSM")
barplot!(ax, (1:length(labels)) .+ 0.2, replace(rr, NaN => 0.0), width=0.4, color=:orange, label="gLV-residual PINN")
hlines!(ax, [0.0], color=:black, linewidth=0.6)
vlines!(ax, [n_s + 0.5], color=:gray, linestyle=:dot)
axislegend(ax; position=:rt)
save(joinpath(figdir, "fig6_correlation_bars.pdf"), fig6)

println("wrote fig5, fig6 (PDF) to $figdir")
