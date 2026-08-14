# ============================================================================
#  Fig 4b analog: prediction correlation vs amount of training data.
#     julia --project=. scripts/experiment_datafraction.jl
#
#  For each training fraction, fits embedded NSM and gLV-residual PINN on a
#  random subset and scores held-out mean per-output Pearson r. Repeats NREP
#  times per fraction (mean +/- std). Reproduces the paper's headline result:
#  the physics-embedded model stays accurate even with little training data.
#
#  Output: figures/fig7_datafraction.pdf
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, Statistics, Random
using CairoMakie

const DATADIR = joinpath(@__DIR__, "..", "data")
species   = ["BT","BV","CH","BU","CS","CA","DP","DSM"]
mediators = ["MS001","MS008","MS014"]
inputs    = String[]

train, dims, scales = load_nsm_csv(joinpath(DATADIR, "cdiff_train_0.csv"),
                                   species, mediators, inputs)
ss, ms = scales
test, _, _ = load_nsm_csv(joinpath(DATADIR, "cdiff_test_0.csv"),
                          species, mediators, inputs; s_scale=ss, m_scale=ms)
n_s, n_m, n_u = dims
n_h, n_layers = 12, 2
s_cap = fill(1.5, n_s); m_cap = fill(1.5, n_m)
sk = (solver=Tsit5(), reltol=1e-6, abstol=1e-6)

FRACS   = [0.1, 0.25, 0.5, 1.0]
NREP    = 2
POOL    = min(length(train), 160)
Random.seed!(3)
pool_idx = randperm(length(train))[1:POOL]

unscale(v) = vcat(v[1:n_s] ./ ss, v[n_s+1:end] ./ ms)
pearson(o, p) = (length(o) < 3 || std(o) < 1e-9 || std(p) < 1e-9) ? NaN : cor(o, p)

# held-out mean per-output Pearson r for a prediction function
function meanr(predict_fn)
    no = n_s + n_m
    O = [Float64[] for _ in 1:no]; P = [Float64[] for _ in 1:no]
    for s in test
        pu = unscale(predict_fn(s)); tu = unscale(s.target)
        for j in 1:no
            isnan(tu[j]) || (push!(O[j], tu[j]); push!(P[j], pu[j]))
        end
    end
    rs = filter(!isnan, [pearson(O[j], P[j]) for j in 1:no])
    return mean(rs)
end

emb_re = zeros(length(FRACS), NREP); res_re = zeros(length(FRACS), NREP)
for (fi, f) in enumerate(FRACS)
    k = max(8, round(Int, f * POOL))
    for rep in 1:NREP
        sub = train[pool_idx[randperm(POOL)[1:k]]]

        cfg = NSMConfig(n_s, n_m, n_u, n_h, n_layers, s_cap, m_cap, t -> Float64[])
        pe  = init_params(cfg)
        fit_rmse!(cfg, pe, sub; lr=5e-3, max_epochs=800, check_every=10, rtol=1e-4, patience=3, verbose=false, sk...)
        emb_re[fi, rep] = meanr(s -> predict_endpoint(cfg, pe, s.u0, s.tf, s.uin; sk...))

        bcfg = BBConfig(n_s + n_m, n_u, n_h, n_layers, t -> Float64[])
        pr   = bb_init(n_s + n_m, n_u, n_h, n_layers)
        fit_residual!(bcfg, pr, sub; lambda=0.5, lr=5e-3, max_epochs=800, check_every=10, rtol=1e-4, patience=3, verbose=false, sk...)
        res_re[fi, rep] = meanr(s -> bb_endpoint(bcfg, pr, s.u0, s.tf, s.uin; sk...))
    end
    println("frac $(f) (n=$k): embedded r=$(round(mean(emb_re[fi,:]),digits=3))  residual r=$(round(mean(res_re[fi,:]),digits=3))")
end

# ---- figure ----
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
xs = FRACS .* 100
em = vec(mean(emb_re, dims=2)); es = vec(std(emb_re, dims=2))
rm = vec(mean(res_re, dims=2)); rs = vec(std(res_re, dims=2))
fig = Figure(size=(760, 520))
ax = Axis(fig[1, 1], xlabel="training data (% of pool)", ylabel="held-out mean Pearson r",
          title="Prediction correlation vs training-data amount (Cdiff held-out)")
band!(ax, xs, em .- es, em .+ es, color=(:dodgerblue, 0.2))
lines!(ax, xs, em, color=:dodgerblue, linewidth=2.5); scatter!(ax, xs, em, color=:dodgerblue, markersize=10, label="Embedded NSM")
band!(ax, xs, rm .- rs, rm .+ rs, color=(:orange, 0.2))
lines!(ax, xs, rm, color=:orange, linewidth=2.5); scatter!(ax, xs, rm, color=:orange, marker=:rect, markersize=10, label="gLV-residual PINN")
axislegend(ax; position=:rc)
save(joinpath(figdir, "fig7_datafraction.pdf"), fig)
println("wrote figures/fig7_datafraction.pdf")
