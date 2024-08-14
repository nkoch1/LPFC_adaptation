using DrWatson
@quickactivate "LPFC_adaptation"
using CSV, DataFrames, JLD2
using LsqFit
using DifferentialEquations, DiffEqCallbacks
using Optimization, OptimizationBBO
using ProgressMeter
include(srcdir("Pospischil.jl"))


# model setup and initial conditions 
cm = 1 #uF/cm2
L = d = 56.9 #um
SA = 4 * π * (L / 10000)^2

gleak = 3.8 * 10^(-5) * 1000#  * SA # S/cm2  *cm2
gNa = 0.058 * 1000 #  * SA #S/cm2  *cm2
gKd = 0.0039 * 1000 #  * SA  #S/cm2  *cm2
gM = 7.87 * 10^(-5) * 1000 #  * SA  # S/cm2 *cm2

τmax = 4004.0 #502. #ms

VT = -57.9 #mV
Eleak = -70.4 # mV

# defaults
ECa = 120 #mV
ENa = 50 #mV
EK = -90 #mV 
gL = 0
gT = 0
Vx = 2 # mV
I = 0.0 #
p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax]

# initial questions
V0 = -60
ic = Pospischil_steady(V0, VT, p)

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
cbs = step_I(step_start, step_end, Imag)
prob = ODEProblem(Pospischil!, ic, tspan, p, callback = cbs, dtmax = samp_rate, # dt=1e-10,
	# saveat=tsteps, 
	abstol = 1e-9, reltol = 1e-9, maxiters = 1e25)
sol_sq = solve(prob, Tsit5(), p = p, u0 = ic, callback = cbs, saveat = tsteps)#
Finst_sq, ISI_sq, spiket_sq, peakamp_sq = freq_analysis(sol_sq, step_start, step_end; ind = 1)



# fit tau of surface
model(x, p) = p[1] * exp.(-x ./ p[2]) .+ p[3]
p0 = [Finst_sq[1], 30.0, 0.0]
t_mod = spiket_sq[1:end-1] .- spiket_sq[1]
fit_sq = curve_fit(model, t_mod, Finst_sq, p0)
tau_step = fit_sq.param[2]


#%% Read in Patch Tau
df = DataFrame(CSV.File("./data/exp/Summary_Decay.csv"))
τ_Patch = -1 ./ (df[isnan.(df[!, "Patch NS"]).==0, "Patch NS"] / 1000)


#%%
global Tend = 1200.0
global Maxiters = 30000
τ_fit = zeros(size(τ_Patch))
Threads.@threads for i in 1:size(τ_Patch)[1]
	tau_des = τ_Patch[i]
	println("---- $(i) ---- $(tau_des) -------")
	function loss(p_in, p_in2)
		p_p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, p_in[1]]
		ic = Pospischil_steady(V0, VT, p_p)
		cbs = step_I(step_start, step_end, Imag)
		local prob = ODEProblem(Pospischil!, ic, tspan, p, callback = cbs, dtmax = samp_rate, abstol = 1e-9, reltol = 1e-9, maxiters = 1e25)
		local sol = solve(prob, Tsit5(), p = p_p, u0 = ic, callback = cbs, saveat = tsteps)#
		if sol.retcode == ReturnCode.Success
			Finst, ISI, spiket, peakamp = freq_analysis(sol, step_start, step_end; ind = 1)
			if size(Finst)[1] >= 4
				# fit tau of surface
				model(x, p) = p[1] .* exp.(-x ./ p[2]) .+ p[3]
				p0 = [Finst[1], tau_des / 33 * 100, 0.0]
				t_mod = spiket[1:end-1] .- spiket_sq[1]
				fit = curve_fit(model, t_mod, Finst, p0)
				tau_step = fit.param[2]
				loss = sum(abs2, tau_step .- tau_des)
			else
				loss = 10000.0
			end
		else
			loss = 10000.0
		end
		GC.gc()
		return loss
	end
	local p = [tau_des]
	local opt_prob = Optimization.OptimizationProblem(loss, [tau_des / 33 * 100], p, lb = [0], ub = [400000.0])
	local sol_prob = solve(opt_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = Maxiters, maxtime = Tend,
		verbose = true, TraceInterval = 50.0, TraceMode = :verbose, PopulationSize = 15, TargetFitness = 1.0, FitnessTolerance = 1e-3, abstol = 1e-3)
	println("$(i): $(tau_des)")
	if SciMLBase.successful_retcode(sol_prob) | (sol_prob.original.elapsed_time < Tend)
		τ_fit[i] = sol_prob.u[1]
		println("save")
	end
end
# save fitted parameter array
@save datadir("sims", "Pospischil_Patch_tau_init.jld2") τ_fit
CSV.write(datadir("sims", "Pospischil_Patch_tau_init.csv"), (; τ_fit))

#%%

@load datadir("sims", "Pospischil_Patch_tau_init.jld2") τ_fit
#%% unsuccessful fits, fit again with longer time
ind_0 = findall(τ_fit .== 0.0)

global Tend = 4800.0
global Maxiters = 300000
Threads.@threads for i in 1:size(ind_0)[1]
	tau_des = τ_Patch[ind_0[i]]
	println("---- $(ind_0[i]) ---- $(tau_des) -------")
	function loss(p_in, p_in2)
		p_p = [I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, p_in[1]]
		cbs = step_I(step_start, step_end, Imag)

		ic_p = Pospischil_steady(V0, VT, p_p)
		prob = ODEProblem(Pospischil!, ic_p, tspan, p, callback = cbs, dtmax = samp_rate, abstol = 1e-9, reltol = 1e-9, maxiters = 1e25)
		sol = solve(prob, Tsit5(), p = p_p, u0 = ic, callback = cbs, saveat = tsteps)
		if sol.retcode == ReturnCode.Success
			Finst, ISI, spiket, peakamp = freq_analysis(sol, step_start, step_end; ind = 1)
			if size(Finst)[1] >= 4
				# fit tau of surface
				model(x, p) = p[1] .* exp.(-x ./ p[2]) .+ p[3]
				p0 = [Finst[1], tau_des / 33 * 100, 0.0]
				t_mod = spiket[1:end-1] .- spiket_sq[1]
				fit = curve_fit(model, t_mod, Finst, p0)
				tau_step = fit.param[2]

				loss = sum(abs2, tau_step .- tau_des)
			else
				loss = 10000.0
			end
		else
			loss = 10000.0
		end
		GC.gc()
		return loss
	end
	local p = [tau_des]
	local opt_prob = Optimization.OptimizationProblem(loss, [tau_des / 33 * 100], p, lb = [0], ub = [400000.0])

	local sol_prob = solve(opt_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = Maxiters, maxtime = Tend,
		verbose = true, TraceInterval = 10.0, TraceMode = :verbose, PopulationSize = 15, TargetFitness = 1.0, FitnessTolerance = 1e-2,
		abstol = 1e-2,
	)
	τ_fit[ind_0[i]] = sol_prob.u[1]
end


#%% save fitted parameter array
@save datadir("sims", "Pospischil_Patch_tau.jld2") τ_fit
CSV.write(datadir("sims", "Pospischil_Patch_tau.csv"), (; τ_fit))
