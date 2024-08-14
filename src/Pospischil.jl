"""
Model from Pospischil et al (2008)
Pospischil, M., Toledo-Rodriguez, M., Monier, C. et al. Minimal Hodgkin–Huxley
  type models for different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441 (2008).
  https://doi.org/10.1007/s00422-008-0263-8

"""

using DifferentialEquations, DiffEqCallbacks
using Peaks, Statistics

##### gating functions #####
αm(V, VT) = -0.32 * (V - VT - 13) / (exp(-(V - VT - 13) / 4) - 1)
βm(V, VT) = 0.28 * (V - VT - 40) / (exp((V - VT - 40) / 5) - 1)
m∞(V, VT) = αm(V, VT) / (αm(V, VT) + βm(V, VT))

αh(V, VT) = 0.128 * exp(-(V - VT - 17) / 18)
βh(V, VT) = 4 / (1 + exp(-(V - VT - 40) / 5))
h∞(V, VT) = αh(V, VT) / (αh(V, VT) + βh(V, VT))

αn(V, VT) = -0.032 * (V - VT - 15) / (exp(-(V - VT - 15) / 5) - 1)
βn(V, VT) = 0.5 * exp(-(V - VT - 10) / 40)
n∞(V, VT) = αn(V, VT) / (αn(V, VT) + βn(V, VT))

p∞(V) = 1 / (1 + exp(-(V + 35) / 10))
τp(V, τmax) = τmax / (3.3 * exp((V + 35) / 20) + exp(-(V + 35) / 20))


αq(V) = 0.055 * (-27 - V) / (exp((-27 - V) / 3.8) - 1)
βq(V) = 0.94 * exp((-75 - V) / 17)
q∞(V) = αq(V) / (αq(V) + βq(V))

αr(V) = 0.000457 * exp((-13 - V) / 50)
βr(V) = 0.0065 / (exp((-15 - V) / 28) + 1)
r∞(V) = αr(V) / (αr(V) + βr(V))

s∞(V, Vx) = 1 / (1 + exp(-(V + Vx + 57) / 6.2))
u∞(V, Vx) = 1 / (1 + exp((V + Vx + 81) / 4))
τu(V, Vx) = 30.8 + (211.4 + exp((V + Vx + 113.2) / 5)) / (3.7 * (1 + exp((V + Vx + 84) / 3.2)))



function Pospischil!(dy, y, p, t)
	"""
	Model from Pospischil et al (2008)
	Pospischil, M., Toledo-Rodriguez, M., Monier, C. et al. Minimal Hodgkin–Huxley
	type models for different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441 (2008).
	https://doi.org/10.1007/s00422-008-0263-8

	"""
	x = @view y[:]
	dx = @view dy[:]

	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax = p
	V, m, h, n, p, q, r, u = x

	dx[1] = dV = (I
				  - gleak * (V - Eleak)
				  - gKd * n^4 * (V - EK)
				  - gNa * m^3 * h * (V - ENa)
				  - gM * p * (V - EK)
				  - gL * q^2 * r * (V - ECa)
				  - gT * s∞(V, Vx)^2 * u * (V - ECa)
	) / cm

	dx[2] = dm = αm(V, VT) * (1 - m) - βm(V, VT) * m
	dx[3] = dh = αh(V, VT) * (1 - h) - βh(V, VT) * h
	dx[4] = dn = αn(V, VT) * (1 - n) - βn(V, VT) * n
	dx[5] = dp = (p∞(V) - p) / τp(V, τmax)
	dx[6] = dq = αq(V) * (1 - q) - βq(V) * q
	dx[7] = dr = αr(V) * (1 - r) - βr(V) * r
	dx[8] = du = (u∞(V, Vx) - u) / τu(V, Vx)

end


function Pospischil_SDE!(dy, y, p, t)
	"""
	Model from Pospischil et al (2008) for use with syn_noise
	Pospischil, M., Toledo-Rodriguez, M., Monier, C. et al. Minimal Hodgkin–Huxley
	type models for different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441 (2008).
	https://doi.org/10.1007/s00422-008-0263-8

	"""
	x = @view y[:]
	dx = @view dy[:]

	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax, Θe, μe, σe, τe, Θi, μi, σi, τi, Ee, Ei = p
	V, m, h, n, p, q, r, u, ge, gi = x

	dx[1] = dV = (I
				  - gleak * (V - Eleak)
				  - gKd * n^4 * (V - EK)
				  - gNa * m^3 * h * (V - ENa)
				  - gM * p * (V - EK)
				  - gL * q^2 * r * (V - ECa)
				  - gT * s∞(V, Vx)^2 * u * (V - ECa)
				  - ge * (V - Ee) - gi * (V - Ei)
	) / cm

	dx[2] = dm = αm(V, VT) * (1 - m) - βm(V, VT) * m
	dx[3] = dh = αh(V, VT) * (1 - h) - βh(V, VT) * h
	dx[4] = dn = αn(V, VT) * (1 - n) - βn(V, VT) * n
	dx[5] = dp = (p∞(V) - p) / τp(V, τmax)
	dx[6] = dq = αq(V) * (1 - q) - βq(V) * q
	dx[7] = dr = αr(V) * (1 - r) - βr(V) * r
	dx[8] = du = (u∞(V, Vx) - u) / τu(V, Vx)

	dx[9] = Θe * (μe - ge) # OU term excit
	dx[10] = Θi * (μi - gi) # OU term inhib
end

function syn_noise!(du, u, p, t)
	"""
	add synaptic input as OU process
	"""
	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax, Θe, μe, σe, τe, Θi, μi, σi, τi, Ee, Ei = p
	du[1:8] .= 0.0
	du[9] = sqrt((2.0 * σe^2.0 / τe))
	du[10] = sqrt((2.0 * σi^2.0 / τi))
end


function Pospischil_steady(V0, VT, p)
	"""
	get steady state of Pospischil! 
	"""
	u0 = [V0, m∞(V0, VT), h∞(V0, VT), n∞(V0, VT), p∞(V0), q∞(V0), r∞(V0), u∞(V0, Vx)]
	ss_cb = TerminateSteadyState()
	prob_ic = ODEProblem(Pospischil!, u0, (0, 100000), p, callback = ss_cb, save_everystep = false, save_end = true, save_start = false)
	sol_ic = solve(prob_ic, maxiters = 1e25)
	ic = sol_ic.u[1]
	return ic
	nothing
end


function step_I(step_start, step_end, I_mag)
	"""
	create step current callback
	"""
	current_step = PresetTimeCallback(step_start, integrator -> integrator.p[1] = I_mag)
	current_step_off = PresetTimeCallback(step_end, integrator -> integrator.p[1] = 0.0)

	cbs = CallbackSet(current_step, current_step_off)
	return cbs
	nothing
end

function freq_analysis(s, step_start, step_end; ind = 1)
	"""
	run frequency analysis to get Finst
	"""
	t_ind = step_start .<= s.t .<= step_end
	xpks = argmaxima(s[ind, t_ind])
	(peaks, proms) = peakproms(xpks, s[ind, t_ind], strict = true, minprom = 25, maxprom = nothing)
	if length(peaks) >= 2 # if more that 2 peaks
		peakamp = s[ind, t_ind][peaks]
		spiket = s.t[t_ind][peaks]
		ISI = zeros(length(peaks) - 1)
		for i in range(2, length(peaks))
			ISI[i-1] = (spiket[i] - spiket[i-1])
		end
		Finst = 1 ./ (ISI / 1000)
	else
		Finst = []
		ISI = []
		spiket = []
		peakamp = []
	end
	return Finst, ISI, spiket, peakamp
end

function freq_analysis_ind(s, step_start, step_end; ind = 1)
	"""
	run frequency analysis to get spike indices and times
	"""
	t_ind = step_start .<= s.t .<= step_end
	xpks = argmaxima(s[ind, t_ind])
	(peaks, proms) = peakproms(xpks, s[ind, t_ind], strict = true, minprom = 25, maxprom = nothing)
	spiket = s.t[t_ind][peaks]
	return peaks, spiket
end


function Pospischil_VGS!(dy, y, p, t)
	"""
	Model from Pospischil et al (2008) for use with VGS derived input
	Pospischil, M., Toledo-Rodriguez, M., Monier, C. et al. Minimal Hodgkin–Huxley
	type models for different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441 (2008).
	https://doi.org/10.1007/s00422-008-0263-8

	"""
	x = @view y[:]
	dx = @view dy[:]

	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax, gsyn, VGS_ind = p
	V, m, h, n, p, q, r, u = x

	dx[1] = dV = (I + gsyn * VGS_int[Int(VGS_ind)](t)
				  - gleak * (V - Eleak)
				  - gKd * n^4 * (V - EK)
				  - gNa * m^3 * h * (V - ENa)
				  - gM * p * (V - EK)
				  - gL * q^2 * r * (V - ECa)
				  - gT * s∞(V, Vx)^2 * u * (V - ECa)
	) / cm

	dx[2] = dm = αm(V, VT) * (1 - m) - βm(V, VT) * m
	dx[3] = dh = αh(V, VT) * (1 - h) - βh(V, VT) * h
	dx[4] = dn = αn(V, VT) * (1 - n) - βn(V, VT) * n
	dx[5] = dp = (p∞(V) - p) / τp(V, τmax)
	dx[6] = dq = αq(V) * (1 - q) - βq(V) * q
	dx[7] = dr = αr(V) * (1 - r) - βr(V) * r
	dx[8] = du = (u∞(V, Vx) - u) / τu(V, Vx)

end


function Pospischil_VGS_SDE_I!(dy, y, p, t)
	"""
	Model from Pospischil et al (2008) for use with VGS derived input and syn_noise_VGS_I
	Pospischil, M., Toledo-Rodriguez, M., Monier, C. et al. Minimal Hodgkin–Huxley
	type models for different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441 (2008).
	https://doi.org/10.1007/s00422-008-0263-8

	"""
	x = @view y[:]
	dx = @view dy[:]
	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax, gsyn, VGS_ind, Θe, μe, σe, τe, Θi, μi, σi, τi, Ee, Ei = p
	V, m, h, n, p, q, r, u, ge, gi = x

	dx[1] = dV = (I + gsyn * VGS_int[Int(VGS_ind)](t)
				  - gleak * (V - Eleak)
				  - gKd * n^4 * (V - EK)
				  - gNa * m^3 * h * (V - ENa)
				  - gM * p * (V - EK)
				  - gL * q^2 * r * (V - ECa)
				  - gT * s∞(V, Vx)^2 * u * (V - ECa)
				  - ge * (V - Ee) - gi * (V - Ei)
	) / cm

	dx[2] = dm = αm(V, VT) * (1 - m) - βm(V, VT) * m
	dx[3] = dh = αh(V, VT) * (1 - h) - βh(V, VT) * h
	dx[4] = dn = αn(V, VT) * (1 - n) - βn(V, VT) * n
	dx[5] = dp = (p∞(V) - p) / τp(V, τmax)
	dx[6] = dq = αq(V) * (1 - q) - βq(V) * q
	dx[7] = dr = αr(V) * (1 - r) - βr(V) * r
	dx[8] = du = (u∞(V, Vx) - u) / τu(V, Vx)

	dx[9] = Θe * (μe - ge) # OU term excit
	dx[10] = Θi * (μi - gi) # OU term inhib
end




function syn_noise_VGS_I!(du, u, p, t)
	"""
	add synaptic input as OU process
	"""
	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax, gsyn, VGS_ind, Θe, μe, σe, τe, Θi, μi, σi, τi, Ee, Ei = p
	du[1:8] .= 0.0
	du[9] = sqrt((2.0 * σe^2.0 / τe))
	du[10] = sqrt((2.0 * σi^2.0 / τi))
end



function Pospischil_VGS_exp!(dy, y, p, t)
	"""
	Model from Pospischil et al (2008) for use with VGS derived and exp input
	Pospischil, M., Toledo-Rodriguez, M., Monier, C. et al. Minimal Hodgkin–Huxley
	type models for different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441 (2008).
	https://doi.org/10.1007/s00422-008-0263-8

	"""
	x = @view y[:]
	dx = @view dy[:]

	I, gleak, gKd, gNa, gM, gL, gT, ENa, EK, ECa, cm, VT, Vx, τmax, gsyn, gsynI, tshift, VGS_ind, inh_ind = p
	V, m, h, n, p, q, r, u = x
	if (t - tshift) < 0
		I_syn = 0.0
	else
		I_syn = gsynI * exp_int[Int(inh_ind)](t)
	end
	dx[1] = dV = (I + gsyn * VGS_int[Int(VGS_ind)](t) + I_syn
				  - gleak * (V - Eleak)
				  - gKd * n^4 * (V - EK)
				  - gNa * m^3 * h * (V - ENa)
				  - gM * p * (V - EK)
				  - gL * q^2 * r * (V - ECa)
				  - gT * s∞(V, Vx)^2 * u * (V - ECa)
	) / cm

	dx[2] = dm = αm(V, VT) * (1 - m) - βm(V, VT) * m
	dx[3] = dh = αh(V, VT) * (1 - h) - βh(V, VT) * h
	dx[4] = dn = αn(V, VT) * (1 - n) - βn(V, VT) * n
	dx[5] = dp = (p∞(V) - p) / τp(V, τmax)
	dx[6] = dq = αq(V) * (1 - q) - βq(V) * q
	dx[7] = dr = αr(V) * (1 - r) - βr(V) * r
	dx[8] = du = (u∞(V, Vx) - u) / τu(V, Vx)

end



