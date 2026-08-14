# ============================================================================
#  Embedded CRM NSM on FP-DP — EXACT reproduction of the paper's Fig 3d.
#  Matches VenturelliLab/Thompson_et_al_2025 to the operation:
#    FPDP/NSM_fpdp_kfold/kfold_fpdp.py, FPDP_kfold_stats.ipynb,
#    nsm/nsm.py (init_params, __init__ scaling, fit_posterior_EM, predict),
#    nsm/nsm_system.py (T, log_abs_det, log_prior_lmbda, log_likelihood_lmbda).
#
#     julia --project=. scripts/run_nsm.jl
#
#  EXACT protocol:
#   * Data: data/fpdp_GlLaSu.csv == BC026_FPDP_GlLaSu_fmt.csv (150 rows,
#     times 0/6.97/10.28/18.66/36.58, one row per timepoint).
#   * Leave-one-CO-CULTURE-out: hold out one Pair_* per fold (10 folds);
#     monocultures always in training and never scored.
#   * Scaling (per fold, nsm.py __init__): ss = 1/max over TRAINING TARGET
#     species; ms = 1/max over TRAINING TARGET mediators (NaN -> 1). Inputs
#     are NOT scaled. Held-out data scaled by the same training scale.
#   * Posterior init: VarPosterior(p, cfg) => per-parameter std == init_params.
#   * Fit: init_params -> fit_posterior_EM! to ELBO-slope convergence
#     (KL minimization), paper nu2=1e-3, sigma2=1e-2. No RMSE warmstart.
#   * Predict each held co-culture free-running from t0 (posterior mean); pool
#     later timepoints; per-output Pearson r (scale-invariant).
#   * n_hidden=15, n_layers=1.
#
#  Writes figures/fig_nsm_fpdp_loo.pdf
# ============================================================================

const USE_EM  = true     # true = fit_posterior_EM! (paper); false = single fit_posterior! to convergence
const N_DRAWS = 100

using NSMjl, OrdinaryDiffEq, Statistics, Random, DelimitedFiles
import ComponentArrays: ComponentArray, getaxes
using CairoMakie

Random.seed!(1234)

const DATADIR  = joinpath(@__DIR__, "..", "data")
const DATAFILE = "BC026_FPDP_GlLaSu_fmt.csv"     # the paper's exact FP-DP file
species  = ["FP_abs","DP_abs"];  mediators = ["Sulfide"]
inputs   = ["Sulfate","Lactate","Glucose"];  outnames = ["FP","DP","Sulfide"]
cols3    = [RGBf(0.20,0.63,0.31), RGBf(0.20,0.68,0.86), RGBf(1.0,0.50,0.05)]

raw, header = readdlm(joinpath(DATADIR, DATAFILE), ','; header=true)
hdr = strip.(String.(vec(header))); cols = Dict(h=>j for (j,h) in enumerate(hdr))
# robust numeric parse: empty/non-numeric cells (the blank input columns) -> NaN
tonum(x) = x isa Number ? Float64(x) : (v = tryparse(Float64, strip(string(x))); v === nothing ? NaN : v)
cv(n) = tonum.(raw[:, cols[n]])
treat = String.(raw[:, cols["Treatments"]]); tcol = cv("Time")

# ---- DATA INTEGRITY CHECK: confirm this is the paper's BC026 file ----------
# (halts BEFORE the expensive fit if the wrong/old file is present)
let
    need = ["Treatments","Time","Sulfate","Lactate","Glucose","FP_abs","DP_abs","Sulfide"]
    missing_cols = [c for c in need if !haskey(cols, c)]
    utimes = sort(unique(round.(tcol, digits=2)))
    ntr = length(unique(treat)); nrow = size(raw, 1)
    expected_t = [0.0, 6.97, 10.28, 18.66, 36.58]
    println("DATA CHECK  file=$DATAFILE  rows=$nrow  treatments=$ntr  timepoints=$utimes")
    ok = isempty(missing_cols) && nrow == 150 && ntr == 30 &&
         length(utimes) == 5 && all(any(abs.(utimes .- t) .< 1e-2) for t in expected_t)
    if !ok
        error("""
        Data does NOT match the paper's BC026 fingerprint — refusing to run.
          expected: 150 rows, 30 treatments, times $(expected_t), columns $(need)
          got:      $nrow rows, $ntr treatments, times $(utimes)$(isempty(missing_cols) ? "" : ", MISSING cols $(missing_cols)")
        Put BC026_FPDP_GlLaSu_fmt.csv at data/$DATAFILE (or edit DATAFILE).""")
    end
    println("DATA CHECK  OK — matches paper BC026 (150 rows, 30 treatments, 5 timepoints).")
end
# ----------------------------------------------------------------------------

Smat = hcat((cv(c) for c in species)...); Mmat = hcat((cv(c) for c in mediators)...)
Umat = hcat((cv(c) for c in inputs)...)
n_s, n_m, n_u = length(species), length(mediators), length(inputs)
no = n_s + n_m

# ---- raw per-treatment data (unscaled; scale is applied per fold) ----
rawgroups = Dict{String,NamedTuple}()
for tr in unique(treat)
    idx = findall(==(tr), treat); idx = idx[sortperm(tcol[idx])]
    rawgroups[tr] = (t=tcol[idx], S=Smat[idx,:], M=Mmat[idx,:], U=Umat[idx[1],:])
end
treatments = sort(collect(keys(rawgroups)))
cocultures = filter(t -> startswith(t, "Pair"), treatments)
isempty(cocultures) && error("no Pair_* co-cultures found; check the Treatments column")
ex = cocultures[1]
println("$(length(treatments)) treatments; leave-one-co-culture-out over $(length(cocultures)) pairs")

# nsm.py __init__ scaling: 1 / nanmax over TRAINING TARGET (non-t0) states; NaN -> 1
function fold_scale(train_trts)
    smax = fill(-Inf, n_s); mmax = fill(-Inf, n_m)
    for tr in train_trts
        g = rawgroups[tr]
        for k in 2:length(g.t)
            for j in 1:n_s; x = g.S[k,j]; (!isnan(x) && x > smax[j]) && (smax[j] = x); end
            for j in 1:n_m; x = g.M[k,j]; (!isnan(x) && x > mmax[j]) && (mmax[j] = x); end
        end
    end
    ss = [isfinite(smax[j]) && smax[j] > 0 ? 1/smax[j] : 1.0 for j in 1:n_s]
    ms = [isfinite(mmax[j]) && mmax[j] > 0 ? 1/mmax[j] : 1.0 for j in 1:n_m]
    return ss, ms
end

# build (t0 -> later) scaled Samples for one treatment under scale (ss, ms)
function build_samples(tr, ss, ms)
    g = rawgroups[tr]; v = Sample[]
    for k in 2:length(g.t)
        u0  = vcat(g.S[1,:] .* ss, g.M[1,:] .* ms)
        tgt = vcat(g.S[k,:] .* ss, g.M[k,:] .* ms)
        push!(v, Sample(u0, tgt, g.t[k], n_u == 0 ? Float64[] : g.U))
    end
    return v
end

n_h, n_layers = 15, 1
s_cap = fill(10.0, n_s); m_cap = fill(10.0, n_m)
sk = (solver=Tsit5(), reltol=1e-6, abstol=1e-6)      # ad defaults to :forward
pear(o,p) = (length(o)<3 || std(o)<1e-9 || std(p)<1e-9) ? NaN : cor(o,p)

Oe = [Float64[] for _ in 1:no]; Pe = [Float64[] for _ in 1:no]; neg = 0; tot = 0
ex_post = nothing; ex_pmean = nothing; ex_ss = nothing; ex_ms = nothing

t0 = time()
for (fi, held) in enumerate(cocultures)
    train_trts = [t for t in treatments if t != held]        # 29 conditions (incl. all monocultures)
    ss, ms = fold_scale(train_trts)                          # per-fold training-target scale
    train  = reduce(vcat, [build_samples(t, ss, ms) for t in train_trts])
    unscale(v) = vcat(v[1:n_s] ./ ss, v[n_s+1:end] ./ ms)

    cfg = NSMConfig(n_s, n_m, n_u, n_h, n_layers, s_cap, m_cap, t->zeros(n_u))
    p   = init_params(cfg)
    post  = VarPosterior(p, cfg)                             # per-parameter init std (nsm.py)
    prior = prior_params(cfg)
    if USE_EM
        fit_posterior_EM!(cfg, post, train, prior; alpha=1.0, nu2=1e-3, sigma2=1e-2,
                          max_iterations=10, patience=1, n_sample_hypers=100,
                          lr=1e-3, verbose=true, sk...)
    else
        fit_posterior!(cfg, post, train, prior; alpha=1.0, nu2=1e-3, sigma2=1e-2,
                       lr=1e-3, verbose=true, sk...)
    end
    pmean = ComponentArray(post.mu, getaxes(p))

    for s in build_samples(held, ss, ms)                    # held-out co-culture, same scale
        q = predict_endpoint(cfg, pmean, s.u0, s.tf, s.uin; sk...)
        pu = unscale(q); tu = unscale(s.target)
        global neg += count(<(-1e-6), q); global tot += length(q)
        for j in 1:no
            inoc = j <= n_s ? (s.u0[j] > 0) : true          # score species only if inoculated (repo guard)
            (inoc && !isnan(tu[j])) && (push!(Oe[j], tu[j]); push!(Pe[j], pu[j]))
        end
    end
    if held == ex
        global ex_post = (post=post, cfg=cfg); global ex_pmean = pmean
        global ex_ss = ss; global ex_ms = ms
    end
    println("  fold $fi/$(length(cocultures)) held out $held  ($(round(time()-t0,digits=0))s)")
end
println("done in $(round(time()-t0, digits=1))s")

re = [pear(Oe[j], Pe[j]) for j in 1:no]
println("\n=== FP-DP leave-one-co-culture-out (NSM, exact paper protocol) ===")
println("neg $(round(100neg/tot, digits=1))%  r: " *
        join(["$(outnames[j])=$(round(re[j], digits=2))" for j in 1:no], "  "))

# ---- 4-panel figure (Pair_1 fold's posterior; scatter pools the 10 co-cultures) ----
g = rawgroups[ex]
u0  = vcat(g.S[1,:] .* ex_ss, g.M[1,:] .* ex_ms); uin = g.U
Xsc = hcat(g.S .* ex_ss', g.M .* ex_ms')            # scaled measured trajectory
tf  = maximum(g.t); te = collect(range(0, tf+2; length=100)); inp_fn = t -> uin
ens      = predict_sample(ex_post.cfg, ex_post.post, u0, te, inp_fn; n_sample=N_DRAWS, solver=Tsit5(), reltol=1e-6, abstol=1e-6)
meanpred = predict_point(ex_post.cfg, ex_pmean, u0, te, inp_fn; solver=Vern9(), reltol=1e-8, abstol=1e-8)

figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
fig = Figure(size=(430,1000))
for j in 1:3
    ax = Axis(fig[j,1], title = j==1 ? "NSM — held-out co-culture ($ex)" : "", xlabel = j==3 ? "time (h)" : "")
    for k in 1:size(ens,1); lines!(ax, te, ens[k,:,j], color=(cols3[j],0.06)); end
    lines!(ax, te, meanpred[:,j], color=cols3[j], linewidth=3)
    scatter!(ax, g.t, Xsc[:,j], color=cols3[j], strokecolor=:black, strokewidth=1, markersize=11)
    j==3 && hlines!(ax,[0.0],color=(:gray,0.5),linewidth=0.6)
end
axs = Axis(fig[4,1], xlabel="measured", ylabel="predicted", aspect=1)
allv = vcat(reduce(vcat,Oe),reduce(vcat,Pe)); lo=-0.01; hi=maximum(allv)+0.04
lines!(axs,[lo,hi],[lo,hi],color=(:gray,0.5),linestyle=:dash)
for j in 1:3
    scatter!(axs, Oe[j], Pe[j]; color=(cols3[j],0.75), strokecolor=:black, strokewidth=0.5,
             markersize=10, label="$(outnames[j]) r=$(round(pear(Oe[j],Pe[j]),digits=2))")
end
limits!(axs,lo,hi,lo,hi); axislegend(axs; position=:rb, labelsize=10)
rowsize!(fig.layout, 4, Relative(0.32))
save(joinpath(figdir,"fig_nsm_fpdp_loo.pdf"), fig)
println("wrote figures/fig_nsm_fpdp_loo.pdf")
