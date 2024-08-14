using DrWatson
@quickactivate "LPFC_adaptation"
using DataFrames, CSV, DelimitedFiles, JLD2
using Interpolations
using ImageFiltering, NumericalIntegration
using LsqFit
using DifferentialEquations, DiffEqCallbacks
using ProgressMeter
using StatsBase
using DSP
include(srcdir("Pospischil.jl"))

#% Read in Patch Tau
df = DataFrame(CSV.File(datadir("exp", "Summary_Decay_fit_250_psth_10.csv")))
τ_Patch = -1 ./ (df[isnan.(df[!, "Patch NS"]).==0, "Patch NS"] / 1000)

# read in Pospischil fits
τ_fit_df = (CSV.read(datadir("sims", "Pospischil_Patch_tau.csv"), DataFrame))
τ_fit = τ_fit_df[!, "τ_fit"]

# read in BS SDF
VGS = Matrix(CSV.read(datadir("exp", "VGS_BS_SDF_groups_sel_n_70.csv"), DataFrame, header = false))
sdf_t = Array(CSV.read(datadir("exp", "BS_sdf_t_groups_new.csv"), DataFrame, header = false)) .- 100
sdf_ind = sdf_t[1, :] .<= 2501.0


#%% model setup and initial conditions 
cm = 1 # uF/cm2
L = d = 56.9 #um
SA = 4 * π * (L / 10000)^2
gleak = 3.8 * 10^(-5) * 1000 # mS/cm2 
gNa = 0.058 * 1000 # mS/cm2 
gKd = 0.0039 * 1000  # mS/cm2 
gM = 7.87 * 10^(-5) * 1000  # mS/cm2 
τmax = 502.0 #ms
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
t_diff = 650.0
tspan = (0.0, 2500.0)
tsteps = tspan[1]+100:0.01:tspan[2]


#%%
# slow IPSP
t_raster = collect(0:0.01:2501)[1:end-1]

# thalamocortical circuit model of auditory cortex in macaque
# Synapse from PV  https://doi.org/10.1016/j.celrep.2023.113378 
tau_d = 18.2
tau_r = 0.07
t_exp = 0:0.01:150
kernel = -exp.((t_exp) ./ -tau_r) .+ exp.((t_exp) ./ -tau_d)
kernel = -kernel ./ maximum(kernel)
bin_size = 10.0
fit_len = 500.0
bins = collect(tsteps[1]:bin_size:tsteps[end])
@. model(x, p) = p[1] + p[2] * exp(x * p[3])
t_diff = 650.0
tspan = (0.0, 2500.0)
tsteps = tspan[1]+100:0.01:tspan[2]
dt = unique(round.(unique((diff(sdf_t, dims = 2))), digits = 3))[1]
fs = Int(ceil(1 / (dt / 1000)))

highpass_cut = [0.1, 1, 2, 3, 4, 5]
τ_VGS_I = zeros(size(τ_fit)[1], size(highpass_cut)[1])
coeff_VGS_I = zeros(size(τ_fit)[1], size(highpass_cut)[1])
offset_VGS_I = zeros(size(τ_fit)[1], size(highpass_cut)[1])
psth_VGS_I = [[[] for i ∈ 1:size(τ_fit)[1]] for j ∈ 1:size(highpass_cut)[1]]
psth_t_VGS_I = [[[] for i ∈ 1:size(τ_fit)[1]] for j ∈ 1:size(highpass_cut)[1]]

I = 0.74 #
g = 5.6 #
gsynI = 2.45


tshift = 0.0

prog = Progress(size(highpass_cut)[1], 1)
for hc ∈ 1:size(highpass_cut)[1]
	designmethod = FIRWindow(hanning(3000; zerophase = false))

	#% convolve with IPSC kernel
	responsetype = Bandpass(highpass_cut[hc], 50; fs)
	inhib_sdf = [[] for i ∈ 1:size(VGS)[2]]
	min_q = zeros(size(VGS)[2])
	for i ∈ 1:size(VGS)[2]
		f = filt(digitalfilter(responsetype, designmethod), VGS[sdf_ind, i])
		q = imfilter(f, kernel)
		inhib_sdf[i] = q
		min_q[i] = minimum(q)
	end
	inhib_sdf = inhib_sdf ./ -minimum(min_q)



	global VGS_int = [LinearInterpolation(sdf_t[1, sdf_ind], VGS[sdf_ind, ii]) for ii in range(1, size(VGS)[2])]
	global exp_int = [LinearInterpolation(sdf_t[1, sdf_ind], inhib_sdf[j]) for j in range(1, size(VGS)[2])]
	Threads.@threads for i in 1:size(τ_fit)[1]
		spike_total_VGS_1_I = [[] for i ∈ 1:size(VGS)[2]]
		gscale = g / maximum(VGS)
		p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i]] # for each model use fit tau
		ic = Pospischil_steady(V0, VT, p) # run to steady state
		for jj in range(1, size(VGS)[2])
			p_exp = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i], gscale, gsynI, tshift, jj, jj]
			prob_exp = ODEProblem(Pospischil_VGS_exp!, ic, tspan, p_exp, dtmax = samp_rate, maxiters = 1e25)
			sol_exp = solve(prob_exp, QNDF(), saveat = tsteps)
			spikes, spike_t = freq_analysis_ind(sol_exp, tsteps[1], tsteps[end]; ind = 1) # get spike times
			spike_total_VGS_1_I[jj] = spike_t
		end
		# get PSTH and fit exp decay to it
		all_spike_times = mapreduce(permutedims, hcat, spike_total_VGS_1_I)
		test = convert(Array{Float64, 1}, vec(all_spike_times))
		h = fit(Histogram, test, bins)
		psth_t = h.edges[1][1:end-1] .- t_diff
		psth = h.weights
		psth_post_stim = psth[psth_t.>=0.0]
		psth_t_post_stim = psth_t[psth_t.>=0.0]
		start_ind = argmax(psth_post_stim)
		stop_ind = findlast(psth_t_post_stim .<= psth_t_post_stim[start_ind] + fit_len)
		psth_aligned = psth_post_stim[start_ind:stop_ind]
		psth_t_aligned = (psth_t_post_stim[start_ind:stop_ind] .- psth_t_post_stim[start_ind]) ./ 1000.0
		Fit = curve_fit(model, psth_t_aligned, psth_aligned, [1, 1, -10.0]; lower = [0.0, 0.0, -1000.0 / ((1 - 1 / exp(1)) * bin_size)], upper = [maximum(psth_aligned), 500.0, 0.0])
		fit_param = deepcopy(Fit.param)

		# into arrays
		τ_VGS_I[i, hc] = (-1 / (fit_param[3] / 1000.0))
		coeff_VGS_I[i, hc] = fit_param[2]
		offset_VGS_I[i, hc] = fit_param[1]
		psth_VGS_I[hc][i] = psth ./ (bin_size / 1000) ./ size(VGS)[2]
		psth_t_VGS_I[hc][i] = psth_t
		GC.gc()
	end
	next!(prog) # update progress bar
end


# save files
fname = "Pospischil_highpass_filter_Inh_sim_I_$(I)_g_$(g)_gi_$(gsynI)_10"
writedlm(datadir("sims", "$(fname)_tau.csv"), τ_VGS_I, ',')
writedlm(datadir("sims", "$(fname)_coeff.csv"), coeff_VGS_I, ',')
writedlm(datadir("sims", "$(fname)_offset.csv"), offset_VGS_I, ',')
AI_VGS_I = offset_VGS_I ./ (offset_VGS_I .+ coeff_VGS_I)
writedlm(datadir("sims", "$(fname)_AI.csv"), AI_VGS_I, ',')

writedlm(datadir("sims", "$(fname)_highpass_cut.csv"), highpass_cut, ',')

df_psth = DataFrame(mapreduce(permutedims, vcat, psth_VGS_I), :auto)
df_psth_t = DataFrame(mapreduce(permutedims, vcat, psth_t_VGS_I), :auto)
CSV.write(datadir("sims", "$(fname)_psth.csv"), df_psth)
CSV.write(datadir("sims", "$(fname)_psth_t.csv"), df_psth_t)



#% 5 Hz
psth_VGS_I_5 = psth_VGS_I[highpass_cut.==5][1]
psth_t_VGS_I_5 = psth_t_VGS_I[highpass_cut.==5][1]
df_psth_5 = DataFrame(x = [Float64.(psth_VGS_I_5[i]) for i in 1:size(psth_VGS_I_5)[1]])
df_psth_t_5 = DataFrame(x = [Float64.(psth_t_VGS_I_5[i]) for i in 1:size(psth_t_VGS_I_5)[1]])
CSV.write(datadir("sims", "$(fname)_psth_5.csv"), df_psth_5)
CSV.write(datadir("sims", "$(fname)_psth_t_5.csv"), df_psth_t_5)


#% 3 Hz
psth_VGS_I_3 = psth_VGS_I[highpass_cut.==3][1]
psth_t_VGS_I_3 = psth_t_VGS_I[highpass_cut.==3][1]
df_psth_3 = DataFrame(x = [Float64.(psth_VGS_I_3[i]) for i in 1:size(psth_VGS_I_3)[1]])
df_psth_t_3 = DataFrame(x = [Float64.(psth_t_VGS_I_3[i]) for i in 1:size(psth_t_VGS_I_3)[1]])
CSV.write(datadir("sims", "$(fname)_psth_3.csv"), df_psth_3)
CSV.write(datadir("sims", "$(fname)_psth_t_3.csv"), df_psth_t_3)


#% 1 Hz
psth_VGS_I_1 = psth_VGS_I[highpass_cut.==1][1]
psth_t_VGS_I_1 = psth_t_VGS_I[highpass_cut.==1][1]
df_psth_1 = DataFrame(x = [Float64.(psth_VGS_I_1[i]) for i in 1:size(psth_VGS_I_1)[1]])
df_psth_t_1 = DataFrame(x = [Float64.(psth_t_VGS_I_1[i]) for i in 1:size(psth_t_VGS_I_1)[1]])
CSV.write(datadir("sims", "$(fname)_psth_1.csv"), df_psth_1)
CSV.write(datadir("sims", "$(fname)_psth_t_1.csv"), df_psth_t_1)



#% 0.1 Hz
psth_VGS_I_0_1 = psth_VGS_I[highpass_cut.==0.1][1]
psth_t_VGS_I_0_1 = psth_t_VGS_I[highpass_cut.==0.1][1]
df_psth_0_1 = DataFrame(x = [Float64.(psth_VGS_I_0_1[i]) for i in 1:size(psth_VGS_I_0_1)[1]])
df_psth_t_0_1 = DataFrame(x = [Float64.(psth_t_VGS_I_0_1[i]) for i in 1:size(psth_t_VGS_I_0_1)[1]])
CSV.write(datadir("sims", "$(fname)_psth_0_1.csv"), df_psth_0_1)
CSV.write(datadir("sims", "$(fname)_psth_t_0_1.csv"), df_psth_t_0_1)
