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
df = DataFrame(CSV.File(datadir("exp", "Summary_Decay.csv")))
τ_Patch = -1 ./ (df[isnan.(df[!, "Patch NS"]).==0, "Patch NS"] / 1000)

# read in Pospischil fits
τ_fit_df = (CSV.read(datadir("sims", "Pospischil_Patch_tau.csv"), DataFrame))
τ_fit = τ_fit_df[!, "τ_fit"]

# read in BS SDF
VGS = Matrix(CSV.read(datadir("exp", "VGS_BS_SDF_groups_sel_n_70.csv"), DataFrame, header = false))
sdf_t = Array(CSV.read(datadir("exp", "BS_sdf_t_groups_new.csv"), DataFrame, header = false)) .- 100
sdf_ind = sdf_t[1, :] .<= 2501.0

#%% model setup and initial conditions 
cm = 1 #uF/cm2
L = d = 56.9 #um
SA = 4 * π * (L / 10000)^2
gleak = 3.8 * 10^(-5) * 1000 # mS/cm2 
gNa = 0.058 * 1000 # mS/cm2 
gKd = 0.0039 * 1000 # mS/cm2 
gM = 7.87 * 10^(-5) * 1000 # mS/cm2 
τmax = 502.0 #ms
VT = -57.9 #mV
Eleak = -70.4 # mV
ECa = 120 #mV
ENa = 50 #mV
EK = -90 #mV 
gL = 0
gT = 0
Vx = 2 # mV
I = 0.0 #
V0 = -60 #mV
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


#%% Run step input and get taus  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
τ_step = zeros(size(τ_fit))
coeff_step = zeros(size(τ_fit))
offset_step = zeros(size(τ_fit))
AI_step = zeros(size(τ_fit))

F_step = [[] for i ∈ 1:size(τ_fit)[1]]
t_step_array = [[] for i ∈ 1:size(τ_fit)[1]]
Threads.@threads for i in 1:size(τ_fit)[1]
	p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i]] # for each model use fit tau

	ic = Pospischil_steady(V0, VT, p) # for each model use fit tau
	cbs = step_I(step_start, step_end, Imag) # step input 
	prob = ODEProblem(Pospischil!, ic, tspan, p, callback = cbs, dtmax = samp_rate,
		abstol = 1e-9, reltol = 1e-9, maxiters = 1e25)
	sol_sq = solve(prob, Tsit5(), p = p, u0 = ic, callback = cbs, saveat = tsteps)
	Finst_sq, ISI_sq, spiket_sq, peakamp_sq = freq_analysis(sol_sq, step_start, step_end; ind = 1) # freq analysis

	# fit tau of firing F
	model(x, p) = p[1] * exp.(-x ./ p[2]) .+ p[3]
	p0 = [Finst_sq[1], 30.0, 0.0]
	t_mod = spiket_sq[1:end-1] .- spiket_sq[1]
	fit_sq = curve_fit(model, t_mod, Finst_sq, p0)

	#fit param into arrays
	τ_step[i] = fit_sq.param[2]
	coeff_step[i] = fit_sq.param[1]
	offset_step[i] = fit_sq.param[3]
	AI_step[i] = Finst_sq[end] / Finst_sq[1]
	F_step[i] = Finst_sq
	t_step_array[i] = t_mod
	GC.gc()
end

# save step files
writedlm(datadir("sims", "Pospischil_step_sim_tau.csv"), τ_step, ',')
writedlm(datadir("sims", "Pospischil_step_sim_coeff.csv"), coeff_step, ',')
writedlm(datadir("sims", "Pospischil_step_sim_offset.csv"), offset_step, ',')
writedlm(datadir("sims", "Pospischil_step_sim_AI.csv"), AI_step, ',')
df_F = DataFrame(x = [Float64.(F_step[i]) for i in 1:size(F_step)[1]])
df_t = DataFrame(x = [Float64.(t_step_array[i]) for i in 1:size(t_step_array)[1]])
CSV.write(datadir("sims", "Pospischil_step_sim_F.csv"), df_F)
CSV.write(datadir("sims", "Pospischil_step_sim_t.csv"), df_t)


#%%
# slow IPSP kernel
t_raster = collect(0:0.01:2501)[1:end-1]

# thalamocortical circuit model of auditory cortex in macaque
# Synapse from PV  https://doi.org/10.1016/j.celrep.2023.113378 
tau_d = 18.0
tau_r = 0.07

t_exp = 0:0.01:150
kernel = -exp.((t_exp) ./ -tau_r) .+ exp.((t_exp) ./ -tau_d)
kernel = -kernel ./ maximum(kernel)
dt = unique(round.(unique((diff(sdf_t, dims = 2))), digits = 3))[1] # find dt of data
fs = Int(ceil(1 / (dt / 1000)))

# lowpass cutoff of RC circuit = 1/(RC) for g_leak of 5 *10^-5 S/cm^2 and C = =1 uF/cm^2 cutoff = 50 Hz 
responsetype = Lowpass(50; fs)
designmethod = FIRWindow(hanning(3000; zerophase = false))

# convolve with IPSC kernel
inhib_sdf = [[] for i ∈ 1:size(VGS)[2]]
min_q = zeros(size(VGS)[2])
for i ∈ 1:size(VGS)[2]
	f = filt(digitalfilter(responsetype, designmethod), VGS[sdf_ind, i])
	q = imfilter(f, kernel)
	inhib_sdf[i] = q
	min_q[i] = minimum(q)
end
inhib_sdf = inhib_sdf ./ -minimum(min_q)


#%% VGS input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_diff = 650.0
tspan = (0.0, 2500.0)
tsteps = tspan[1]+100:0.01:tspan[2]
τ_VGS = zeros(size(τ_fit))
coeff_VGS = zeros(size(τ_fit))
offset_VGS = zeros(size(τ_fit))
psth_VGS = [[] for i ∈ 1:size(τ_fit)[1]]
psth_t_VGS = [[] for i ∈ 1:size(τ_fit)[1]]
spikes_all = mapreduce(permutedims, vcat, [[[] for i ∈ 1:size(VGS)[2]] for i ∈ 1:size(τ_fit)[1]])
spike_raster_all = zeros(size(τ_fit)[1], 2501 * 100)

bin_size = 10.0
fit_len = 500.0
bins = collect(tsteps[1]:bin_size:tsteps[end])
@. model(x, p) = p[1] + p[2] * exp(x * p[3])

I = 0.75
g = 2.5
gscale = g / maximum(VGS) # normalize so that g*VGS where VGS [0,1]

global VGS_int = [LinearInterpolation(sdf_t[1, sdf_ind], VGS[sdf_ind, ii]) for ii in range(1, size(VGS)[2])]
prog = Progress(size(τ_fit)[1], 1) # progress bar
Threads.@threads for i in 1:size(τ_fit)[1]
	p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i]] # for each model use fit tau
	ic = Pospischil_steady(V0, VT, p) # run to steady state
	for j in range(1, size(VGS)[2])
		p_exp = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i], gscale, j] # j is sweep of VGS 
		prob_exp = ODEProblem(Pospischil_VGS!, ic, tspan, p_exp, dtmax = samp_rate, maxiters = 1e25)
		sol_exp = solve(prob_exp, QNDF(), saveat = tsteps)
		spikes, spike_t = freq_analysis_ind(sol_exp, tsteps[1], tsteps[end]; ind = 1) # get spike times
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
	Fit = curve_fit(model, psth_t_aligned, psth_aligned, [1, 1, -10.0]; lower = [0.0, 0.0, -1000.0 / ((1 - 1 / exp(1)) * bin_size)], upper = [maximum(psth_aligned), 500.0, 0.0])
	fit_param = deepcopy(Fit.param)

	# into arrays
	spike_raster_all[i, Int.(round.(all_spike_times .* 100))] .= 1
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

# save files
fname = "Pospischil_filter_sim.jld2"
save(datadir("sims", fname), Dict("τ_VGS" => τ_VGS, "psth_VGS" => psth_VGS, "psth_t_VGS" => psth_t_VGS, "spike_raster_all" => spike_raster_all))

fname = "Pospischil_filter_sim"
writedlm(datadir("sims", "$(fname)_tau.csv"), τ_VGS, ',')
writedlm(datadir("sims", "$(fname)_coeff.csv"), coeff_VGS, ',')
writedlm(datadir("sims", "$(fname)_offset.csv"), offset_VGS, ',')
AI_VGS = offset_VGS ./ (offset_VGS .+ coeff_VGS)
writedlm(datadir("sims", "$(fname)_AI.csv"), AI_VGS, ',')

df_psth = DataFrame(x = [Float64.(psth_VGS[i]) for i in 1:size(psth_VGS)[1]])
df_psth_t = DataFrame(x = [Float64.(psth_t_VGS[i]) for i in 1:size(psth_t_VGS)[1]])
CSV.write(datadir("sims", "$(fname)_psth.csv"), df_psth)
CSV.write(datadir("sims", "$(fname)_psth_t.csv"), df_psth_t)


#%% VGS input + Inhibition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
τ_VGS_I = zeros(size(τ_fit))
coeff_VGS_I = zeros(size(τ_fit))
offset_VGS_I = zeros(size(τ_fit))

psth_VGS_I = [[] for i ∈ 1:size(τ_fit)[1]]
psth_t_VGS_I = [[] for i ∈ 1:size(τ_fit)[1]]
median_τ_ind = findfirst(median(τ_fit) .== τ_fit)

tshift = 0.0

I = 0.74
g = 5.6
gsynI = 2.45

global VGS_int = [LinearInterpolation(sdf_t[1, sdf_ind], VGS[sdf_ind, ii]) for ii in range(1, size(VGS)[2])]
global exp_int = [LinearInterpolation(sdf_t[1, sdf_ind], inhib_sdf[j]) for j in range(1, size(VGS)[2])]
prog = Progress(size(τ_fit)[1], 1) # progress bar
Threads.@threads for i in 1:size(τ_fit)[1]
	spike_total_VGS_1_I = [[] for i ∈ 1:size(VGS)[2]]
	gscale = g / maximum(VGS)
	p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i]] # for each model use fit tau
	ic = Pospischil_steady(V0, VT, p) # run to steady state
	for jj in range(1, size(VGS)[2])
		p_exp = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τ_fit[i], gscale, gsynI, tshift, jj, jj] # jj is sweep of VGS / exp input
		prob_exp = ODEProblem(Pospischil_VGS_exp!, ic, tspan, p_exp, dtmax = samp_rate, maxiters = 1e25) # run to steady state
		sol_exp = solve(prob_exp, QNDF(), saveat = tsteps)
		spikes, spike_t = freq_analysis_ind(sol_exp, tsteps[1], tsteps[end]; ind = 1)
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
	τ_VGS_I[i] = (-1 / (fit_param[3] / 1000.0))
	coeff_VGS_I[i] = fit_param[2]
	offset_VGS_I[i] = fit_param[1]
	psth_VGS_I[i] = psth ./ (bin_size / 1000) ./ size(VGS)[2]
	psth_t_VGS_I[i] = psth_t
	GC.gc()
	next!(prog) # update progress bar
end


using PyPlot
plt.figure()
for i in 1:size(τ_fit)[1]
	plt.plot(psth_t_VGS_I[i], psth_VGS_I[i])
end
plt.show()

# save files
fname = "Pospischil_filter_Inh_sim"
writedlm(datadir("sims", "$(fname)_tau.csv"), τ_VGS_I, ',')
writedlm(datadir("sims", "$(fname)_coeff.csv"), coeff_VGS_I, ',')
writedlm(datadir("sims", "$(fname)_offset.csv"), offset_VGS_I, ',')
AI_VGS_I = offset_VGS_I ./ (offset_VGS_I .+ coeff_VGS_I)

writedlm(datadir("sims", "$(fname)_AI.csv"), AI_VGS_I, ',')
df_psth = DataFrame(x = [Float64.(psth_VGS_I[i]) for i in 1:size(psth_VGS_I)[1]])
df_psth_t = DataFrame(x = [Float64.(psth_t_VGS_I[i]) for i in 1:size(psth_t_VGS_I)[1]])
CSV.write(datadir("sims", "$(fname)_psth.csv"), df_psth)
CSV.write(datadir("sims", "$(fname)_psth_t.csv"), df_psth_t)
println(median(τ_VGS_I))
println(median(AI_VGS_I))

#% save summary files

fname = "Pospischil_TAU_filter_Inh_sim"
df = DataFrame("τ_step" => log10.(τ_step), "τ_VGS" => log10.(τ_VGS), "τ_VGS_I" => log10.(τ_VGS_I))
CSV.write(datadir("sims", "$(fname).csv"), df)

df_coeff = DataFrame("coeff_step" => coeff_step, "coeff_VGS" => coeff_VGS, "coeff_VGS_I" => coeff_VGS_I)
CSV.write(datadir("sims", "$(fname)_coeff.csv"), df_coeff)

df_offset = DataFrame("offset_step" => offset_step, "offset_VGS" => offset_VGS, "offset_VGS_I" => offset_VGS_I)
CSV.write(datadir("sims", "$(fname)_offset.csv"), df_offset)

df_AI = DataFrame("AI_step" => AI_step, "AI_VGS" => AI_VGS, "AI_VGS_I" => AI_VGS_I)
CSV.write(datadir("sims", "$(fname)_AI.csv"), df_AI)

println(median(τ_VGS_I))
println(median(AI_VGS_I))
