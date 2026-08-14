# ============================================================================
#  FP-DP experiment — PAPER PROTOCOL.  Only the *method* is under investigation
#  (embedded NSM vs gLV-residual PINN); every other factor matches Fit_CR_NSM.
#     julia --project=. scripts/experiment_fpdp.jl
#
#  Matched to the paper's notebook:
#    - n_hidden = 16, n_layers = 1
#    - per-output scaling (load_nsm_csv scale=True default)
#    - s_cap = m_cap = 10
#    - two-stage fit:  fit_rmse! (warm start) -> fit_posterior! (Bayesian VI)
#    - nu2 = 1e-3, sigma2 = 1e-4, alpha_0 = 1e-3
#  Predictions use the posterior mean (predict at zmean(post)).
#
#  Writes figures/fig8_fpdp_groundtruth.pdf
# ============================================================================

using NSMjl, OrdinaryDiffEq, Statistics, Random
import ComponentArrays: ComponentArray, getaxes, getdata
using CairoMakie

const DATADIR = joinpath(@__DIR__, "..", "data")
species   = ["FP_abs","DP_abs"];  mediators = ["Sulfide"]
inputs    = ["Sulfate","Lactate","Glucose"];  outnames = ["FP","DP","Sulfide"]

data_all, dims, scales = load_nsm_csv(joinpath(DATADIR, "fpdp_GlLaSu.csv"),
                                      species, mediators, inputs)   # per-output scales
ss, ms = scales; n_s, n_m, n_u = dims
println("loaded $(length(data_all)) samples; dims=$(dims); s_scale=$(round.(ss,digits=2)) m_scale=$(round.(ms,digits=2))")

Random.seed!(0)
idx  = randperm(length(data_all)); ncut = round(Int, 0.30 * length(data_all))
test = data_all[idx[1:ncut]]; train = data_all[idx[ncut+1:end]]
println("train $(length(train)), test $(length(test))")

# ---- paper hyperparameters ----
n_h, n_layers = 16, 1
s_cap = fill(10.0, n_s); m_cap = fill(10.0, n_m)
sk = (solver=Tsit5(), reltol=1e-6, abstol=1e-6)
ALPHA0, NU2, SIGMA2 = 1e-3, 1e-3, 1e-4

# ---- embedded NSM: two-stage (RMSE warm start -> Bayesian posterior) ----
cfg = NSMConfig(n_s, n_m, n_u, n_h, n_layers, s_cap, m_cap, t -> zeros(n_u))
pe  = init_params(cfg)
println("embedded — stage 1: fit_rmse! (warm start)")
fit_rmse!(cfg, pe, train; lr=1e-3, max_epochs=5000, check_every=25, rtol=1e-5, patience=5, sk...)
println("embedded — stage 2: fit_posterior! (Bayesian VI)  [optional polish; stage 1 alone already reaches paper-level r]")
prior = prior_params(cfg)
post  = VarPosterior(pe; log_s_init=log(0.05))
fit_posterior!(cfg, post, train, prior; alpha=ALPHA0, nu2=NU2, sigma2=SIGMA2,
               lr=1e-3, epochs=100, print_every=25, sk...)
pe_final = ComponentArray(post.mu, getaxes(pe))     # posterior mean
# To skip the (slow) VI stage, set  pe_final = pe  and delete the block above.

# ---- residual PINN (matched architecture + training; point estimate) ----
bcfg = BBConfig(n_s + n_m, n_u, n_h, n_layers, t -> zeros(n_u))
pr   = bb_init(n_s + n_m, n_u, n_h, n_layers)
println("residual PINN: fit_residual!")
fit_residual!(bcfg, pr, train; lambda=0.5, lr=1e-3, max_epochs=5000, check_every=25, rtol=1e-5, patience=5, sk...)

# ---- evaluate on held-out (per-output unscale) ----
unscale(v) = vcat(v[1:n_s] ./ ss, v[n_s+1:end] ./ ms)
pearson(o, p) = (length(o) < 3 || std(o) < 1e-9 || std(p) < 1e-9) ? NaN : cor(o, p)
function collect_preds(predict_fn, dat)
    no = n_s + n_m; O = [Float64[] for _ in 1:no]; P = [Float64[] for _ in 1:no]
    errs = Float64[]; neg = 0; tot = 0
    for s in dat
        q = predict_fn(s); pu = unscale(q); tu = unscale(s.target)
        push!(errs, sqrt(mean(x -> isnan(x) ? 0.0 : x^2, tu .- pu)))
        neg += count(<(-1e-6), q); tot += length(q)
        for j in 1:no; isnan(tu[j]) || (push!(O[j], tu[j]); push!(P[j], pu[j])); end
    end
    O, P, mean(errs), 100neg/tot
end
emb_pred(s) = predict_endpoint(cfg,  pe_final, s.u0, s.tf, s.uin; sk...)
res_pred(s) = bb_endpoint(bcfg, pr, s.u0, s.tf, s.uin; sk...)

Oe, Pe_, rmse_e, neg_e = collect_preds(emb_pred, test)
Or, Pr_, rmse_r, neg_r = collect_preds(res_pred, test)
re = [pearson(Oe[j], Pe_[j]) for j in 1:(n_s+n_m)]
rr = [pearson(Or[j], Pr_[j]) for j in 1:(n_s+n_m)]

println("\n=== FP-DP held-out (PAPER PROTOCOL; only the method differs) ===")
println("Embedded NSM (CRM, 2-stage) RMSE $(round(rmse_e,digits=4))  neg $(round(neg_e,digits=1))%  r: " *
        join(["$(outnames[j])=$(round(re[j],digits=2))" for j in 1:3], "  "))
println("gLV-residual PINN           RMSE $(round(rmse_r,digits=4))  neg $(round(neg_r,digits=1))%  r: " *
        join(["$(outnames[j])=$(round(rr[j],digits=2))" for j in 1:3], "  "))

# ---- figure ----
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
fig = Figure(size=(1150, 760))
for (row, (Ps, mdl, c)) in enumerate([(Pe_, "Embedded NSM (CRM, 2-stage)", :dodgerblue),
                                      (Pr_, "gLV-residual PINN", :orange)])
    Os = row == 1 ? Oe : Or
    for j in 1:3
        o = Os[j]; p = Ps[j]; rj = pearson(o, p)
        ax = Axis(fig[row, j], title="$mdl — $(outnames[j])  r=$(isnan(rj) ? "NA" : string(round(rj,digits=2)))",
                  xlabel = row == 2 ? "measured" : "", ylabel = j == 1 ? "predicted" : "")
        if !isempty(o)
            lo = min(0.0, minimum(o), minimum(p)) - 0.005; hi = max(maximum(o), maximum(p)) + 0.01
            scatter!(ax, o, p, markersize=9, color=(c, 0.7))
            lines!(ax, [lo, hi], [lo, hi], color=:black, linestyle=:dash)
            limits!(ax, lo, hi, lo, hi)
        end
    end
end
Label(fig[0, :], "FP-DP held-out (paper protocol): embedded NSM vs gLV-residual PINN", fontsize=14)
save(joinpath(figdir, "fig8_fpdp_groundtruth.pdf"), fig)
println("wrote figures/fig8_fpdp_groundtruth.pdf")
