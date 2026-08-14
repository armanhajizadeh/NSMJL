# ============================================================================
#  Step-1 smoke test: forward solve must match the verified SciPy reference.
#
#  Run from the package root:
#     julia --project=. test/smoke_forward.jl
#  (first run: `julia --project=.  -e 'using Pkg; Pkg.instantiate()'`)
# ============================================================================

using NSMjl, ComponentArrays, OrdinaryDiffEq, DelimitedFiles, Test

include("ref_params.jl")   # exact params + dims that produced ref_traj.csv

# Time-dependent perturbation — IDENTICAL to the Python reference:
#   channel 1: smooth Gaussian pulse centred at t=6 (e.g. an antibiotic course)
#   channel 2: constant background input 0.2 (e.g. a dietary factor)
pert_inputs(t) = [exp(-((t - 6.0)^2) / (2 * 1.5^2)), 0.2]

cfg = NSMConfig(N_S, N_M, N_U, N_H, N_LAYERS, S_CAP, M_CAP, pert_inputs)

# load the exact reference parameters into a ComponentArray
p = init_params(cfg)
p.C   .= C_RAW
p.P   .= P_RAW
p.d_m .= DM_RAW
p.Wi  .= WI
p.bi  .= BI
p.Wh  .= WH
p.bh  .= BH
p.Wo  .= WO
p.bo  .= BO

u0    = vcat(S0, M0)
tspan = (0.0, TF)

ref    = readdlm(joinpath(@__DIR__, "ref_traj.csv"), ','; header=true)[1]
t_eval = ref[:, 1]
ref_Y  = ref[:, 2:end]

sol = solve_forward(cfg, p, u0, tspan; saveat=t_eval,
                    solver=Vern9(), reltol=1e-10, abstol=1e-10)
Y = permutedims(reduce(hcat, sol.u))   # (n_time, n_s+n_m)

maxerr = maximum(abs.(Y .- ref_Y))
println("max abs error vs SciPy reference: ", maxerr)
println("final state (Julia): ", round.(Y[end, :]; digits=6))
println("final state (ref)  : ", round.(ref_Y[end, :]; digits=6))

@testset "NSM forward solve matches reference" begin
    @test size(Y) == size(ref_Y)
    @test all(Y[:, 1:N_S] .>= -1e-8)          # species stay non-negative
    @test maxerr < 1e-5
end
