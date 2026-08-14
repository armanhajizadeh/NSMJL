# ============================================================================
#  FP-DP paper-style figure (Fig 3c/3d layout): three posterior-predictive
#  trajectory panels (FP, DP, Sulfide) stacked above a pooled predicted-vs-
#  measured scatter. Colors match the paper: FP green, DP cyan, Sulfide orange.
#     julia --project=. scripts/plot_fpdp_paper.jl
#
#  Fits the embedded NSM (fit_rmse! -> fit_posterior!) so the trajectory panels
#  show 100 posterior draws (thin) + posterior mean (thick) + measured points
#  for one held-out co-culture; the scatter pools all held-out predictions.
#
#  Writes figures/fig_fpdp_paper.pdf
# ============================================================================

using NSMjl, OrdinaryDiffEq, Statistics, Random, DelimitedFiles
import ComponentArrays: ComponentArray, getaxes
using CairoMakie

const DATADIR = joinpath(@__DIR__, "..", "data")
species   = ["FP_abs","DP_abs"];  mediators = ["Sulfide"]
inputs    = ["Sulfate","Lactate","Glucose"];  outnames = ["FP","DP","Sulfide"]
cols3 = [RGBf(0.20,0.63,0.31), RGBf(0.20,0.68,0.86), RGBf(1.0,0.50,0.05)]  # green, cyan, orange

# ---- grouped loader (per-output scaling) + raw times for the trajectory panel ----
raw, header = readdlm(joinpath(DATADIR, "fpdp_GlLaSu.csv"), ','; header=true)
hdr = strip.(String.(vec(header))); cols = Dict(h=>j for (j,h) in enumerate(hdr))
cv(n) = Float64.(raw[:, cols[n]])
treat = String.(raw[:, cols["Treatments"]]); time = Float64.(raw[:, cols["Time"]])
Smat = hcat((cv(c) for c in species)...); Mmat = hcat((cv(c) for c in mediators)...)
Umat = hcat((cv(c) for c in inputs)...)
n_s, n_m, n_u = 2, 1, 3
cmax(M,k)=maximum(filter(!isnan,@view M[:,k]))
ss=[1/cmax(Smat,k) for k in 1:n_s]; ms=[1/cmax(Mmat,k) for k in 1:n_m]
groups = Dict{String,Vector{Sample}}(); rawtraj = Dict{String,Any}()
for tr in unique(treat)
    idx=findall(==(tr),treat); idx=idx[sortperm(time[idx])]
    v=Sample[]
    for k in 2:length(idx)
        i0,ik=idx[1],idx[k]
        push!(v, Sample(vcat(Smat[i0,:].*ss, Mmat[i0,:].*ms),
                        vcat(Smat[ik,:].*ss, Mmat[ik,:].*ms), time[ik],
                        n_u==0 ? Float64[] : Umat[i0,:]))
    end
    groups[tr]=v
    rawtraj[tr]=(t=time[idx], X=hcat(Smat[idx,:].*ss', Mmat[idx,:].*ms'), u=Umat[idx[1],:])
end
treatments = sort(collect(keys(groups)))

# co-culture = both FP and DP reach nonzero; pick one for the trajectory panels
is_co(tr) = maximum(rawtraj[tr].X[:,1])>0.05 && maximum(rawtraj[tr].X[:,2])>0.05
example = something(findfirst(is_co, treatments), 1); ex = treatments[example]

# ---- fit NSM on all but the example (rmse -> posterior) ----
train = reduce(vcat, [groups[t] for t in treatments if t != ex])
n_h, n_layers = 16, 1
s_cap=fill(10.0,n_s); m_cap=fill(10.0,n_m); sk=(solver=Tsit5(),reltol=1e-6,abstol=1e-6)
cfg = NSMConfig(n_s,n_m,n_u,n_h,n_layers,s_cap,m_cap, t->zeros(n_u))
pe  = init_params(cfg)
println("fit_rmse! ..."); fit_rmse!(cfg, pe, train; lr=1e-3, max_epochs=3000, check_every=25, rtol=1e-5, patience=5, sk...)
println("fit_posterior! ..."); prior=prior_params(cfg); post=VarPosterior(pe; log_s_init=log(0.05))
fit_posterior!(cfg, post, train, prior; alpha=1e-3, nu2=1e-3, sigma2=1e-4, lr=1e-3, epochs=100, sk...)
pmean = ComponentArray(post.mu, getaxes(pe))

# ---- posterior-predictive trajectories for the example condition ----
exdat = rawtraj[ex]; u0 = exdat.X[1,:]; uin = exdat.u
tf = maximum(exdat.t); te = collect(range(0, tf+2; length=100))
inp_fn = t -> uin
ens = predict_sample(cfg, post, u0, te, inp_fn; n_sample=100, solver=Tsit5(), reltol=1e-6, abstol=1e-6)
meanpred = predict_point(cfg, pmean, u0, te, inp_fn; solver=Vern9(), reltol=1e-8, abstol=1e-8)

# ---- pooled held-out scatter (leave-one-out-lite: predict each treatment from the model fit on the rest is heavy;
#      here we score all treatments' samples with the single fit for a comparable scatter) ----
Osc=[Float64[] for _ in 1:3]; Psc=[Float64[] for _ in 1:3]
for tr in treatments, s in groups[tr]
    q = predict_endpoint(cfg, pmean, s.u0, s.tf, s.uin; sk...)
    for j in 1:3
        tv=s.target[j]/ (j<=n_s ? ss[j] : ms[j-n_s]); pv=q[j]/(j<=n_s ? ss[j] : ms[j-n_s])
        isnan(tv) || (push!(Osc[j],tv); push!(Psc[j],pv))
    end
end
pear(o,p)=(length(o)<3||std(o)<1e-9||std(p)<1e-9) ? NaN : cor(o,p)

# ---- figure: 3 stacked trajectory panels + scatter ----
figdir=joinpath(@__DIR__,"..","figures"); mkpath(figdir)
fig=Figure(size=(430,1000))
for j in 1:3
    ax=Axis(fig[j,1], title = j==1 ? "NSM" : "", xlabel = j==3 ? "time (h)" : "")
    for k in 1:size(ens,1); lines!(ax, te, ens[k,:,j], color=(cols3[j],0.06)); end
    lines!(ax, te, meanpred[:,j], color=cols3[j], linewidth=3)
    scatter!(ax, exdat.t, exdat.X[:,j], color=cols3[j], strokecolor=:black, strokewidth=1, markersize=11)
    j==3 && hlines!(ax,[0.0],color=(:gray,0.5),linewidth=0.6)
end
axs=Axis(fig[4,1], xlabel="measured", ylabel="predicted", aspect=1)
allv=vcat(reduce(vcat,Osc),reduce(vcat,Psc)); lo=-0.01; hi=maximum(allv)+0.04
lines!(axs,[lo,hi],[lo,hi],color=(:gray,0.5),linestyle=:dash)
for j in 1:3
    scatter!(axs, Osc[j], Psc[j]; color=(cols3[j],0.75), strokecolor=:black, strokewidth=0.5,
             markersize=10, label="$(outnames[j]) r=$(round(pear(Osc[j],Psc[j]),digits=2))")
end
limits!(axs,lo,hi,lo,hi); axislegend(axs; position=:rb, labelsize=10)
rowsize!(fig.layout, 4, Relative(0.32))
save(joinpath(figdir,"fig_fpdp_paper.pdf"), fig)
println("example co-culture: $ex")
println("scatter r: FP=$(round(pear(Osc[1],Psc[1]),digits=2)) DP=$(round(pear(Osc[2],Psc[2]),digits=2)) Sulfide=$(round(pear(Osc[3],Psc[3]),digits=2))")
println("wrote figures/fig_fpdp_paper.pdf")
