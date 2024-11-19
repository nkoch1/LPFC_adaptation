using DrWatson
@quickactivate "LPFC_adaptation"

using DataFrames, CSV, DelimitedFiles, JLD2
using Interpolations
using ImageFiltering, NumericalIntegration
using LsqFit
using DifferentialEquations, DiffEqCallbacks
using ProgressMeter
using StatsBase
include(srcdir("Pospischil.jl"))



# https://doi.org/10.1016%2Fs0306-4522(01)00344-x
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320220/#FD2

Table1 = DataFrame(
	Cell = ["Layer VI", "Layer III", "Layer Va", "Layer Vb"],
	Area = [34636, 20321, 55017, 93265], # (μm2)
	Rin = [58.9, 94.2, 38.9, 23.1], # (MΩ)
	ge0 = [0.012, 0.006, 0.018, 0.029], #(μS)
	σe = [0.0030, 0.0019, 0.0035, 0.0042], #(μS)
	τe = [2.7, 7.8, 2.6, 2.8], # (ms)
	gi0 = [0.057, 0.044, 0.098, 0.16], # (μS)
	σi = [0.0066, 0.0069, 0.0092, 0.01], # (μS)
	τi = [10.5, 8.8, 8.0, 8.5], # (ms)
)

Table1[!, "Area (cm)"] = Table1[!, "Area"] * (1 / 10000)^2
Table1[!, "ge0 (mS)"] = Table1[!, "ge0"] * 1 / 1000 # μS to mS
Table1[!, "σe (mS)"] = Table1[!, "σe"] * 1 / 1000 # μS to mS
Table1[!, "gi0 (mS)"] = Table1[!, "gi0"] * 1 / 1000 # μS to mS
Table1[!, "σi (mS)"] = Table1[!, "σi"] * 1 / 1000 # μS to mS
#% convert mS/cm^2 for Pospischil model
Table1[!, "ge0 (mS/cm^2)"] = Table1[!, "ge0 (mS)"] ./ Table1[!, "Area (cm)"]
Table1[!, "σe (mS/cm^2)"]  = Table1[!, "σe (mS)"] ./ Table1[!, "Area (cm)"]
Table1[!, "gi0 (mS/cm^2)"] = Table1[!, "gi0 (mS)"] ./ Table1[!, "Area (cm)"]
Table1[!, "σi (mS/cm^2)"]  = Table1[!, "σi (mS)"] ./ Table1[!, "Area (cm)"]

# get mean values for use
Θe = 1.0 / mean(Table1[!, "τe"]) # Speed of the mean reversion (scaling distance between Xt and μ) # Excitatory time constant ~2ms
μe = mean(Table1[!, "ge0 (mS/cm^2)"]) #mean of the process
σe = (sqrt(2 / mean(Table1[!, "τe"]))) * mean(Table1[!, "σe (mS/cm^2)"]) # volatility that scales standard Wiener process (dWt)
τe = mean(Table1[!, "τe"])

Θi = 1.0 / mean(Table1[!, "τi"]) # Speed of the mean reversion (scaling distance between Xt and μ) # Excitatory time constant ~2ms
μi = mean(Table1[!, "gi0 (mS/cm^2)"]) #mean of the process
σi = (sqrt(2 / mean(Table1[!, "τi"]))) * mean(Table1[!, "σi (mS/cm^2)"]) # volatility that scales standard Wiener process (dWt)
τi = mean(Table1[!, "τi"])

Ee = 0
Ei = -75

#% Read in Patch Tau
df = DataFrame(CSV.File(datadir("exp", "Summary_Decay.csv")))
τ_Patch = -1 ./ (df[isnan.(df[!, "Patch NS"]).==0, "Patch NS"] / 1000)

# read in Pospischil fits
τ_fit_df = (CSV.read(datadir("sims", "Pospischil_Patch_tau.csv"), DataFrame))
τ_fit = τ_fit_df[!, "τ_fit"]


# read in BS SDF
VGS_orig = Matrix(CSV.read(datadir("exp", "VGS_BS_SDF_groups_sel_n_70.csv"), DataFrame, header = false))
sdf_t_orig = Array(CSV.read(datadir("exp", "BS_sdf_t_groups.csv"), DataFrame, header = false)) .- 100
first500_ind = (0 .<= sdf_t_orig[1, :] .<= 500.0)
first_500 = VGS_orig[first500_ind, :]
VGS = vcat(first_500, first_500, VGS_orig)
sdf_t = vcat(sdf_t_orig[1, first500_ind] .- 1000.01, sdf_t_orig[1, first500_ind] .- 500.01, sdf_t_orig[1, :])
sdf_ind = sdf_t[:] .<= 2501.0

#%% model setup and initial conditions 
cm = 1 #uF/cm2
L = d = 56.9 #um
SA = 4 * π * (L / 10000)^2
gleak = 3.8 * 10^(-5) * 1000 # mS/cm2 
gNa = 0.058 * 1000 # mS/cm2 
gKd = 0.0039 * 1000 # mS/cm2 
gM = 7.87 * 10^(-5) * 1000 # mS/cm2 
τmax = 502.0 #502. #ms
VT = -57.9 #mV
Eleak = -70.4 # mV
ECa = 120 #mV
ENa = 50 #mV
EK = -90 #mV 
gL = 0
gT = 0
Vx = 2 # mV
I = 0.0
V0 = -60
Imag = 3

step_start = 100
step_length = 1000
post_step = 100
sim_start = 0
step_end = step_length + step_start
sim_end = step_end + post_step

samp_rate = 0.01
tspan = (0.0, sim_end)
tsteps = 0.0:samp_rate:sim_end

p_ic = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax]
ic = Pospischil_steady(V0, VT, p_ic)# run to steady state

tspan = (0.0, 1000.0);
Ee = 0
Ei = -75

#%% VGS input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u0 = [μe, μi* 0.35];
t_diff = 650.0
tspan = (-1000.0, 2500.0)
tsteps = tspan[1]+100:0.01:tspan[2]
τ_VGS = zeros(size(τ_fit))
coeff_VGS = zeros(size(τ_fit))
offset_VGS = zeros(size(τ_fit))

psth_VGS = [[] for i ∈ 1:size(τ_fit)[1]]
psth_t_VGS = [[] for i ∈ 1:size(τ_fit)[1]]


bin_size = 10.0
fit_len = 750.0
bins = collect(tsteps[1]:bin_size:tsteps[end])
@. model(x, p) = p[1] + p[2] * exp(x * p[3])

spikes_all = mapreduce(permutedims, vcat, [[[] for i ∈ 1:size(VGS)[2]] for i ∈ 1:size(τ_fit)[1]])
spike_raster_all = zeros(size(τ_fit)[1], Int(tspan[2] - tspan[1]) * 100)
I = 0.0
g = 3. 

gscale = g / maximum(VGS)

global VGS_int = [LinearInterpolation(sdf_t[sdf_ind], VGS[sdf_ind, ii]) for ii in range(1, size(VGS)[2])]
prog = Progress(size(τ_fit)[1], 1)
Threads.@threads for i in 1:size(τ_fit)[1]
	# i = 1
	p_ic = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i]]# for each model use fit tau
	ic = Pospischil_steady(V0, VT, p_ic) # for each model use fit tau

	p_exp = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i], Θe, μe, σe, τe, Θi, μi * 0.35, σi, τi, Ee, Ei]
	u0 = [ic..., μe, μi]
	prob_exp_ic = SDEProblem(Pospischil_SDE!, syn_noise!, u0, (0, 10000.0), p_exp, save_everystep = true, save_end = true, save_start = false, dtmax = samp_rate, maxiters = 1e25, seed = i)
	sol_exp_ic = solve(prob_exp_ic, saveat = tsteps)
	ic_new = sol_exp_ic[end]

	for j in range(1, size(VGS)[2])
		p_exp = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i], gscale, j, Θe, μe, σe, τe, Θi, μi * 0.35, σi, τi, Ee, Ei]
		u0_j = [ic_new..., μe, μi * 0.2]
		prob_exp = SDEProblem(Pospischil_VGS_SDE_I!, syn_noise_VGS_I!, u0_j, tspan, p_exp, saveat = samp_rate, dtmax = samp_rate, maxiters = 1e25, seed = ((j - 1) * size(τ_fit)[1]) + i)
		sol_exp = solve(prob_exp, saveat = tsteps)
		spikes, spike_t = freq_analysis_ind(sol_exp, tsteps[1], tsteps[end]; ind = 1)
		spikes_all[i, j] = spike_t
	end
	# get PSTH and fit exp decay to it
	all_spike_times = mapreduce(permutedims, hcat, spikes_all[i, :])
	test = convert(Array{Float64, 1}, vec(all_spike_times))
	h = fit(Histogram, test, bins)
	psth_t = h.edges[1][1:end-1]
	psth = h.weights
	psth_post_stim = psth[psth_t.>=0.0]
	psth_t_post_stim = psth_t[psth_t.>=0.0]
	start_ind = argmax(psth_post_stim)
	stop_ind = findlast(psth_t_post_stim .<= psth_t_post_stim[start_ind] + fit_len)
	psth_aligned = psth_post_stim[start_ind:stop_ind]
	psth_t_aligned = (psth_t_post_stim[start_ind:stop_ind] .- psth_t_post_stim[start_ind]) ./ 1000.0
	Fit = curve_fit(model, psth_t_aligned, psth_aligned, [1, 1, -500.0]; lower = [0.0, 0.0, -1000.0], upper = [maximum(psth_aligned), 500.0, 0.0])
	fit_param = deepcopy(Fit.param)


	# into arrays
	spike_raster_all[i, Int.(round.((all_spike_times[1, :] .+ 1000) .* 100))] .= 1
	τ_VGS[i] = (-1 / (fit_param[3] / 1000.0))
	coeff_VGS[i] = fit_param[2]
	offset_VGS[i] = fit_param[1]
	psth_VGS[i] = psth ./ (bin_size / 1000) ./ size(VGS)[2]
	psth_t_VGS[i] = psth_t .- t_diff
	GC.gc()
	next!(prog) # update progress bar
end



using PyPlot
plt.figure()

for i in 1:size(τ_fit)[1]
	plt.plot(psth_t_VGS[i], psth_VGS[i])
end
plt.show()

println(median(τ_VGS))

println(τ_VGS)

#%% save files
fname = "Pospischil_filter_OU_sim.jld2"
save(datadir("sims", fname), Dict("τ_VGS" => τ_VGS, "psth_VGS" => psth_VGS, "psth_t_VGS" => psth_t_VGS, "spike_raster_all" => spike_raster_all))

fname = "Pospischil_filter_OU_sim"
writedlm(datadir("sims", "$(fname)_tau.csv"), τ_VGS, ',')
writedlm(datadir("sims", "$(fname)_coeff.csv"), coeff_VGS, ',')
writedlm(datadir("sims", "$(fname)_offset.csv"), offset_VGS, ',')
AI_VGS = offset_VGS ./ (offset_VGS .+ coeff_VGS)
writedlm(datadir("sims", "$(fname)_AI.csv"), AI_VGS, ',')

df_psth = DataFrame(x = [Float64.(psth_VGS[i]) for i in 1:size(psth_VGS)[1]])
df_psth_t = DataFrame(x = [Float64.(psth_t_VGS[i]) for i in 1:size(psth_t_VGS)[1]])
CSV.write(datadir("sims", "$(fname)_psth.csv"), df_psth)
CSV.write(datadir("sims", "$(fname)_psth_t.csv"), df_psth_t)

using DataFrames;
τ_step_df = CSV.read(datadir("sims", "Pospischil_step_sim_tau.csv"), DataFrame; header = false);
τ_step = Vector{Float64}(vec(Array(τ_step_df)))

fname = "Pospischil_TAU_filter_OU_sim"
df = DataFrame("τ_step" => log10.(τ_step), "τ_VGS" => log10.(τ_VGS))
CSV.write(datadir("sims", "$(fname).csv"), df)

using DataFrames;
df = CSV.read(datadir("sims", "Pospischil_step_sim_AI.csv"), DataFrame; header = false);
AI_step = Vector{Float64}(vec(Array(df)))

fname = "Pospischil_TAU_filter_OU_sim_AI"
df = DataFrame("AI_step" => AI_step, "AI_VGS" => AI_VGS)
CSV.write(datadir("sims", "$(fname).csv"), df)
