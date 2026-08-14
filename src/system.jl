# ============================================================================
#  NSM core dynamical system  (port of nsm/nsm_system.py :: system, lines 10-53)
#
#  State u = [s; m]   (species abundances stacked on mediator concentrations)
#  dsdt = s .* o_s .* (1 - s/s_cap)                                   (eNODE law)
#  dmdt = softplus(o_m) .* ( (P·relu(dsdt)).*(1 - m/m_cap)            (CRM law)
#                            - m .* (C·s + d_m) )
#  where [o_s; o_m] = NN([s; m; u(t)]) is the neural network output.
#
#  The ONLY change vs. the reference is that `inputs` is a function u(t),
#  evaluated inside the RHS at every solver step  ->  time-dependent
#  perturbation. In the reference it is a constant vector.
#
#  NOTE (reverse-mode AD): this file is UNCHANGED. The in-place RHS is exactly
#  what SciMLSensitivity's ReverseDiffVJP traces for the adjoint's
#  vector-Jacobian products, so no out-of-place rewrite is needed. It also has
#  no data-dependent branching (relu/softplus/max are fine), so the compiled
#  tape (ReverseDiffVJP(true)) is safe here.
# ============================================================================

using ComponentArrays

# ---------- activations ----------
@inline softplusf(a) = log1p(exp(a))
@inline reluf(a)     = max(a, zero(a))

# ---------- positivity transform g: ℝ -> ℝ₊  (log10 form, matches reference) ----------
# raw (unconstrained) params z are stored; g is applied inside the RHS so that
# C, P, d_m are strictly positive. g_inv is used to initialise from a positive value.
@inline g(a)     = log10(1 + exp10(a))
@inline g_inv(a) = log10(exp10(a) - 1)

# ---------- fixed model configuration ----------
# `inputs` is the perturbation schedule: inputs(t)::AbstractVector of length n_u.
struct NSMConfig{V<:AbstractVector,F}
    n_s::Int          # number of species
    n_m::Int          # number of mediators (metabolites)
    n_u::Int          # number of external input / perturbation channels
    n_h::Int          # hidden width
    n_layers::Int     # total NN layers (paper convention, ≥1); hidden loop runs n_layers-1 times
    s_cap::V          # species carrying capacities   (length n_s)
    m_cap::V          # mediator caps                 (length n_m)
    inputs::F         # inputs(t) -> vector length n_u   (TIME-DEPENDENT)
end

@inline n_hidden(cfg::NSMConfig)  = cfg.n_layers - 1
@inline input_dim(cfg::NSMConfig) = cfg.n_s + cfg.n_m + cfg.n_u
@inline state_dim(cfg::NSMConfig) = cfg.n_s + cfg.n_m

# ---------- parameter initialisation (port of nsm.py :: init_params point values) ----------
# Returns a ComponentArray of RAW params. NN weights start at zero (as in the
# reference); C, P init so g(C)=1/n_s, d_m so g(d_m)=0.01.
function init_params(cfg::NSMConfig; Tp::Type=Float64)
    n_s, n_m, n_h = cfg.n_s, cfg.n_m, cfg.n_h
    n_x = input_dim(cfg); n_o = state_dim(cfg); nl = n_hidden(cfg)
    ComponentArray(
        C   = fill(Tp(g_inv(1 / n_s)), n_m, n_s),
        P   = fill(Tp(g_inv(1 / n_s)), n_m, n_s),
        d_m = fill(Tp(g_inv(0.01)),    n_m),
        Wi  = zeros(Tp, n_h, n_x),
        bi  = zeros(Tp, n_h),
        Wh  = zeros(Tp, n_h, n_h, nl),   # nl == 0 gives an empty (n_h,n_h,0) array; loop is skipped
        bh  = zeros(Tp, n_h, nl),
        Wo  = zeros(Tp, n_o, n_h),
        bo  = zeros(Tp, n_o),
    )
end

# ---------- the RHS ----------
# In-place. `p` is a ComponentArray as built by init_params; `cfg` carries dims,
# caps, and the perturbation schedule.
function nsm_rhs!(du, u, p, t, cfg::NSMConfig)
    n_s, n_m = cfg.n_s, cfg.n_m
    s = @view u[1:n_s]
    m = @view u[n_s+1:n_s+n_m]

    # neural network:  h = tanh(Wi·[s;m;u(t)] + bi), stacked hidden layers, output o
    uext = cfg.inputs(t)                               # <-- time-dependent perturbation
    h = tanh.(p.Wi * vcat(s, m, uext) .+ p.bi)
    @inbounds for i in 1:n_hidden(cfg)
        h = tanh.(view(p.Wh, :, :, i) * h .+ view(p.bh, :, i)) ./ 10
    end
    o  = (p.Wo * h .+ p.bo) ./ 10
    os = @view o[1:n_s]
    om = @view o[n_s+1:n_s+n_m]

    # species: eNODE law (abundance-multiplied, logistic cap) -> guarantees s≥0
    dsdt = s .* os .* (1 .- s ./ cfg.s_cap)

    # mediators: consumer-resource law, NN output (softplus, positive) multiplies it
    Cs  = g.(p.C) * s                                  # consumption  (C·s)
    Prd = g.(p.P) * reluf.(dsdt)                       # production   (P·relu(dsdt))
    dmdt = softplusf.(om) .* (Prd .* (1 .- m ./ cfg.m_cap) .- m .* (Cs .+ g.(p.d_m)))

    @inbounds du[1:n_s]         .= dsdt
    @inbounds du[n_s+1:n_s+n_m] .= dmdt
    return nothing
end
