# ============================================================================
#  Clark 25-species NSM — 16h interpolation task (paper Fig 4c).
#  Matches ClarkInterpolate/NSM_clark_interpolate.ipynb:
#    train on all timepoints EXCEPT 16h; predict the (0 -> 16h) transition;
#    fit with fit_posterior_EM(); n_hidden=20, nu2=1e-3, sigma2=1e-2.
#     julia --project=. scripts/run_clark.jl
#
#  Writes figures/fig_clark_16h.pdf
#
#  NOTE ON SPEED: Clark is 25 species (~970 params, 29 outputs). Per-sample SGD
#  with ForwardDiff is CORRECT but heavy at this scale. AD = :forward is safe
#  and matches FP-DP exactly; AD = :reverse (adjoint) is the regime where
#  reverse-mode finally pays off (cost ~independent of parameter count) — try it
#  if :forward is too slow. See the flag below.
# ============================================================================

const AD = :reverse     # :forward (safe, exact) | :reverse (faster at 25 species; needs SciMLSensitivity)

using NSMjl, OrdinaryDiffEq, Statistics, Random, DelimitedFiles
import ComponentArrays: ComponentArray, getaxes
using CairoMakie

Random.seed!(1234)

const DATADIR = joinpath(@__DIR__, "..", "data")
mediators = ["Butyrate","Acetate","Lactate","Succinate"]

# ---- robust numeric parse (blank / non-numeric cells -> NaN) ----
function tonum(x)
    x isa Number && return Float64(x)
    v = tryparse(Float64, strip(string(x)))
    return v === nothing ? NaN : v
end

# ---- load Clark ----
raw, header = readdlm(joinpath(DATADIR, "clark.csv"), ','; header=true)
hdr  = strip.(String.(vec(header)))
cols = Dict(h => j for (j, h) in enumerate(hdr))
colvec(n) = tonum.(raw[:, cols[n]])

species = [h for h in hdr if endswith(h, "_OD")]          # 25 species
n_s, n_m = length(species), length(mediators)

treat = String.(raw[:, cols["Treatments"]])
time  = colvec("Time")
Smat  = hcat((colvec(c) for c in species)...)
Mmat  = hcat((colvec(c) for c in mediators)...)

# ---- scaling from TRAINING rows only (Time != 16), per paper (df_train max) ----
trainrow = time .!= 16.0
colmax(M, k, mask) = maximum(filter(!isnan, @view M[mask, k]))
ss = [1.0 / colmax(Smat, k, trainrow) for k in 1:n_s]
ms = [1.0 / colmax(Mmat, k, trainrow) for k in 1:n_m]
println("Clark: $(n_s) species, $(n_m) metabolites, $(length(unique(treat))) treatments")

# ---- build samples: train = (0 -> non-16h), test = (0 -> 16h) ----
train = Sample[]; test = Sample[]
for tr in unique(treat)
    idx = findall(==(tr), treat); idx = idx[sortperm(time[idx])]
    isempty(idx) && continue
    i0 = idx[1]
    u0 = vcat(Smat[i0, :] .* ss, Mmat[i0, :] .* ms)
    for k in 2:length(idx)
        ik = idx[k]
        tgt = vcat(Smat[ik, :] .* ss, Mmat[ik, :] .* ms)
        s = Sample(u0, tgt, time[ik], Float64[])
        (time[ik] == 16.0 ? push!(test, s) : push!(train, s))   # 16h held out
    end
end
println("train $(length(train)) samples (t=32,48), test $(length(test)) samples (t=16)")

# ---- fit embedded NSM via fit_posterior_EM! (paper protocol) ----
n_h, n_layers = 20, 1
s_cap = fill(10.0, n_s); m_cap = fill(10.0, n_m)
sk = (solver=Tsit5(), reltol=1e-6, abstol=1e-6)
cfg = NSMConfig(n_s, n_m, 0, n_h, n_layers, s_cap, m_cap, t -> Float64[])
p    = init_params(cfg)
post = VarPosterior(p, cfg)                 # per-parameter init std (nsm.py)
prior = prior_params(cfg)                   # zero-mean prior (paper Eq 13)
println("fitting NSM via fit_posterior_EM!  (25 species — SLOW; AD=$AD)...")
t0 = Base.time()
fit_posterior_EM!(cfg, post, train, prior; ad=AD, alpha=1.0, nu2=1e-3, sigma2=1e-2,
                  max_iterations=10, patience=1, n_sample_hypers=100, lr=1e-3, verbose=true, sk...)
pf = ComponentArray(post.mu, getaxes(p))
println("fit done in $(round(Base.time()-t0, digits=1))s")

# ---- predict 16h; collect metabolite predicted-vs-measured (original units) ----
O = [Float64[] for _ in 1:n_m]; P = [Float64[] for _ in 1:n_m]
for s in test
    q = predict_endpoint(cfg, pf, s.u0, s.tf, s.uin; sk...)
    for j in 1:n_m
        tv = s.target[n_s + j] / ms[j]; pv = q[n_s + j] / ms[j]
        isnan(tv) || (push!(O[j], tv); push!(P[j], pv))
    end
end
pear(o, p) = (length(o) < 3 || std(o) < 1e-9 || std(p) < 1e-9) ? NaN : cor(o, p)
rmse(o, p) = sqrt(mean(abs2, o .- p))

# ---- figure (paper Fig 4c colors) ----
colmap = Dict("Acetate"=>RGBf(0.16,0.50,0.72), "Butyrate"=>RGBf(0.94,0.66,0.36),
              "Lactate"=>RGBf(0.20,0.63,0.31), "Succinate"=>RGBf(0.80,0.25,0.24))
legorder = ["Acetate","Butyrate","Lactate","Succinate"]
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
fig = Figure(size=(560, 540))
ax  = Axis(fig[1, 1], title="NSM", xlabel="measured", ylabel="predicted", aspect=1)
allv = vcat(reduce(vcat, O), reduce(vcat, P)); lo = -0.02; hi = maximum(allv) + 0.05
lines!(ax, [lo, hi], [lo, hi], color=(:gray, 0.6), linestyle=:dash)
for name in legorder
    j = findfirst(==(name), mediators)
    r = pear(O[j], P[j]); e = rmse(O[j], P[j])
    scatter!(ax, O[j], P[j]; color=(colmap[name], 0.75), strokecolor=:black, strokewidth=0.5,
             markersize=11, label="$name, r=$(round(r,digits=3)), rmse=$(round(e,digits=2))")
end
limits!(ax, lo, hi, lo, hi)
axislegend(ax; position=:rb, framevisible=true, labelsize=10)
save(joinpath(figdir, "fig_clark_16h.pdf"), fig)

println("\n=== Clark 16h held-out metabolites (NSM, fit_posterior_EM) ===")
for name in legorder
    j = findfirst(==(name), mediators)
    println("  $name  r=$(round(pear(O[j],P[j]),digits=3))  rmse=$(round(rmse(O[j],P[j]),digits=3))")
end
println("wrote figures/fig_clark_16h.pdf")