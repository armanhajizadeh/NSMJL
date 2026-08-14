# ============================================================================
#  Real-data loader — Julia port of nsm/utilities.py :: process_df
#
#  Reads a Venturelli-format CSV with columns:
#     Treatments, Time, <species...>, <mediators...>, <inputs...>
#  Groups by Treatments, sorts by Time; the first timepoint is the initial
#  condition and each later timepoint becomes an (IC -> observation) Sample.
#
#  Returns Vector{Sample} plus the (n_s, n_m, n_u) dims and the scale factors
#  used (1/max over the training data), so predictions can be returned in
#  original units.
# ============================================================================

using DelimitedFiles
using Statistics

"""
    load_nsm_csv(path, species, mediators, inputs; s_scale=nothing, m_scale=nothing)

Parse a Venturelli-format CSV into `Vector{Sample}`.

`species`, `mediators`, `inputs` are the column-name lists (as strings).
If `s_scale`/`m_scale` are `nothing` they are computed **per output** as
`1/maximum` of each column (the paper's scale=True default); pass the training
scales (vectors) when loading a test file. Returns `(data, dims, scales)` where
`dims = (n_s, n_m, n_u)` and `scales = (s_scale, m_scale)` are per-output vectors.
"""
function load_nsm_csv(path::AbstractString, species, mediators, inputs;
                      s_scale=nothing, m_scale=nothing)
    raw, header = readdlm(path, ','; header=true)
    cols = Dict(strip(String(h)) => j for (j, h) in enumerate(vec(header)))
    colvec(name) = Float64.(raw[:, cols[name]])

    treat = String.(raw[:, cols["Treatments"]])
    time  = Float64.(raw[:, cols["Time"]])
    Smat  = isempty(species)   ? zeros(length(time), 0) : hcat((colvec(c) for c in species)...)
    Mmat  = isempty(mediators) ? zeros(length(time), 0) : hcat((colvec(c) for c in mediators)...)
    Umat  = isempty(inputs)    ? zeros(length(time), 0) : hcat((colvec(c) for c in inputs)...)

    n_s, n_m, n_u = length(species), length(mediators), length(inputs)
    # per-output scaling (each variable by its own max) — matches the paper's
    # scale=True default. `s_scale`/`m_scale` may be given (scalar or vector) to
    # reuse the training scales on a test file.
    colmax(M, k) = maximum(filter(!isnan, @view M[:, k]))
    ss = s_scale === nothing ? [1.0 / colmax(Smat, k) for k in 1:n_s] : s_scale
    ms = m_scale === nothing ? (n_m == 0 ? Float64[] : [1.0 / colmax(Mmat, k) for k in 1:n_m]) : m_scale

    data = Sample[]
    for tr in unique(treat)
        idx = findall(==(tr), treat)
        order = sortperm(time[idx]); idx = idx[order]
        for k in 2:length(idx)
            i0, ik = idx[1], idx[k]
            u0  = vcat(Smat[i0, :] .* ss, Mmat[i0, :] .* ms)
            tgt = vcat(Smat[ik, :] .* ss, Mmat[ik, :] .* ms)
            uin = n_u == 0 ? Float64[] : Umat[i0, :]
            push!(data, Sample(u0, tgt, time[ik], uin))
        end
    end
    return data, (n_s, n_m, n_u), (ss, ms)
end
