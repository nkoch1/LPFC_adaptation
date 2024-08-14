# %%
import os
os.chdir('../../')
from src.plt_fxns import lighten_color, remove_axis_box, generate_spike_train, exp_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.gridspec as gridspec
import scipy.io as sio
import matplotlib.patches as patches
import string
from matplotlib import ticker
plt.rcParams["font.family"] = "Arial"
# %% load data

# example data
df_VGS = pd.read_csv('./data/exp/VGS_table.csv')
df_VGS_sel = df_VGS[df_VGS['selective'] == 1]
broad_id = df_VGS_sel.loc[df_VGS_sel.index[df_VGS_sel.narrow ==
                                           0], 'cellName'].tolist()
narrow_id = df_VGS_sel.loc[df_VGS_sel.index[df_VGS_sel.narrow ==
                                            1], 'cellName'].tolist()
narrow_id.remove('B20171109_DChan042_2')
BS_W = pd.read_csv('./data/exp/BS_waveform.csv', header=None)
NS_W = pd.read_csv('./data/exp/NS_waveform.csv', header=None)
W_t = np.arange(1, NS_W.shape[0]+1, 1)

df_extra = pd.read_csv('./data/exp/pfcwvfrmData2.csv')
broadVivo = 'T20170330_VChan012_3'
narrowVivo = 'B20171109_DChan012_3'

df_extra_BS = df_extra[df_extra['UnitIDs'].isin(broad_id)]
df_extra_NS = df_extra[df_extra['UnitIDs'].isin(narrow_id)]
broad_ex_ind = df_extra_BS.index[df_extra_BS['UnitIDs'] == broadVivo][0]
narrow_ex_ind = df_extra_NS.index[df_extra_NS['UnitIDs'] == narrowVivo][0]

# summary data
sdf_t = pd.read_csv('./data/exp/PSTH_t_array.csv', header=None)
VGS_broad = pd.read_csv('./data/exp/VGS_broad_psth.csv')
VGS_narrow = pd.read_csv('./data/exp/VGS_narrow_psth.csv')
VGS_narrow = VGS_narrow.drop(columns=['B20171109_DChan042_2'])
df = pd.read_csv('./data/exp/Summary_Decay.csv')
df_vgs_b = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_vgs_b['decay $\tau$'] = np.log10(
    -1 / (df.loc[df.index[~df["VGS BS"].isnull()], "VGS BS"] / 1000))
df_vgs_b['condition'] = "VGS BS"

df_vgs_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_vgs_n['decay $\tau$'] = np.log10(
    -1 / (df.loc[df.index[~df["VGS NS"].isnull()], "VGS NS"] / 1000))
df_vgs_n['condition'] = "VGS NS"
df_sum_tau = pd.concat([df_vgs_b, df_vgs_n], ignore_index=True)

df_offset = pd.read_csv('./data/exp/extracell_offset.csv')
df_coeff = pd.read_csv(
    './data/exp/extracell_coefficient.csv')
df_tau = pd.read_csv('./data/exp/extracell_decay.csv')
df_AI = df_offset / (df_offset + df_coeff)

# latency
df_lat = pd.read_csv(
    './data/exp/Latency_extraction_extracell_psth_latency_5_sd_SDF.csv')
df_lat = df_lat[df_lat.Task == 'VGS']

# Tmax
df_lat_max = pd.read_csv(
    './data/exp/Latency_extraction_extracell_psth_latency.csv')
df_lat_max = df_lat_max[df_lat_max.Task == 'VGS']


# generate simulated data for PSTH diagram
t = np.arange(0, 500, 0.01)  # generate time array for exponential fits
f_c_size = 10
f_p_size = 8
lw = 1.
c1 = (120/255, 120/255, 120/255)
c2 = (82/255, 80/255, 80/255)
numtrials = 100
numtrials_dis = 3
tend = 500
tstim = 225
stim_num_spikes = 5
nbins = 50

F0 = 40
F1 = 20
cell_1_spikes = generate_spike_train(
    numtrials, 0, F0, F1, tend, tstim, stim_num_spikes)

F0 = 30
F1 = 14
cell_2_spikes = generate_spike_train(
    numtrials, 1, F0, F1, tend, tstim, stim_num_spikes)


# %% Create Figure
np.random.seed(0)  # set seed for stripplot jitter

pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choices #####
binsize = 0.01
sdf_xlim = (-150, 600)
sdf_xlim_norm = (-250, 500)
alpha = 0.5
lw = 1.1
lw_R = 2.
broad_ex_cell = "B20171110_VChan093_4"
narrow_ex_cell = "B20171109_DChan012_3"
order_e = ['VGS BS', 'VGS NS']
palette_e = ['#40A787', '#2060A7']
tau_ylim = (0.5, 2.8)
pairs1 = [('VGS BS', 'VGS NS'),]
pair_xlabels = ['BS', 'NS']
ms = 3
waveform_xlim = (0, 1250)
waveform_ylim = (-0.052*1000, 0.04*1000)
xticks_psth = [0, 200, 400, 600]
xtick_labels_psth = ['Stim. Onset', '200', '400', '600']
xticks_psth_n = [-200, 0, 200, 400]
xtick_labels_psth_n = ['-200', '$\mathrm{T}_{\mathrm{Max}}$', '200', '400']


##### figure layout #####
fig = plt.figure(figsize=(6.5, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[
                      1.1, 0.375, 0.375], hspace=0.5,  wspace=0.5)
gs_ex = gridspec.GridSpecFromSubplotSpec(
    2, 4, height_ratios=[0.7, 1.9], wspace=0.7, hspace=0.8, subplot_spec=gs[0])
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 4, wspace=0.75, hspace=0.7, subplot_spec=gs[1])
gs_bottom2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, wspace=0.6, hspace=0.7, subplot_spec=gs[2])


gs_waveform_2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, wspace=0.25, hspace=0.1, subplot_spec=gs_ex[0, 2:])
gs_d = gridspec.GridSpecFromSubplotSpec(
    1, 2, wspace=0.25, subplot_spec=gs_waveform_2[1])
gs_0 = gridspec.GridSpecFromSubplotSpec(
    2, 1, height_ratios=[1, 1], wspace=0.2, hspace=0.05, subplot_spec=gs_d[0])
ax0_all = fig.add_subplot(gs_0[:])
ax0_R = fig.add_subplot(gs_0[0])
ax0_PSTH = fig.add_subplot(gs_0[1])

gs_1 = gridspec.GridSpecFromSubplotSpec(
    2, 1, height_ratios=[1, 1], wspace=0.2, hspace=0.05, subplot_spec=gs_d[1])
ax1_all = fig.add_subplot(gs_1[:])
ax1_R = fig.add_subplot(gs_1[0])
ax1_PSTH = fig.add_subplot(gs_1[1])

gs_waveform = gridspec.GridSpecFromSubplotSpec(
    1, 2, wspace=0.25, hspace=0.1, subplot_spec=gs_ex[:2])
ax_BS_waveform = fig.add_subplot(gs_waveform[0])
ax_NS_waveform = fig.add_subplot(gs_waveform[1])
ax_class = fig.add_subplot(gs_waveform_2[0])

gs_BS_VGS = gridspec.GridSpecFromSubplotSpec(4, 1, height_ratios=[
                                             1, 0.9, 0.05, 0.9], wspace=0.35, hspace=0.45, subplot_spec=gs_ex[1, :2])
ax_BS_VGS_Lab = fig.add_subplot(gs_ex[1, :2])
ax_BS_VGS_R = fig.add_subplot(gs_BS_VGS[0])
ax_BS_VGS_P = fig.add_subplot(gs_BS_VGS[1])
ax_BS_VGS_P_n = fig.add_subplot(gs_BS_VGS[3])

gs_NS_VGS = gridspec.GridSpecFromSubplotSpec(4, 1, height_ratios=[
                                             1, 0.9, 0.05, 0.9], wspace=0.35, hspace=0.45, subplot_spec=gs_ex[1, 2:])
ax_NS_VGS_Lab = fig.add_subplot(gs_ex[1, 2:])
ax_NS_VGS_R = fig.add_subplot(gs_NS_VGS[0])
ax_NS_VGS_P = fig.add_subplot(gs_NS_VGS[1])
ax_NS_VGS_P_n = fig.add_subplot(gs_NS_VGS[3])


ax_A = fig.add_subplot(gs_bottom[1])
ax_B = fig.add_subplot(gs_bottom[2])
ax_A_B = fig.add_subplot(gs_bottom[3])
ax_latency_max = fig.add_subplot(gs_bottom[0])
ax_AI = fig.add_subplot(gs_bottom2[0])
ax_tau = fig.add_subplot(gs_bottom2[1])
ax_latency = fig.add_subplot(gs_bottom2[2])

##### PSTH diagram #####
ax0_R.eventplot(cell_1_spikes, color=c1, linewidths=lw)
ax0_R.set_ylim((0.5, numtrials_dis+0.5))

counts0, bins0 = np.histogram(
    np.concatenate(cell_1_spikes, axis=0), bins=nbins)
ax0_PSTH.step(bins0[:-1], counts0, color=c1)

ax1_R.eventplot(cell_2_spikes,  color=c2, linewidths=lw, zorder=-1)
ax1_R.set_ylim((0.5, numtrials_dis+0.5))

counts1, bins1 = np.histogram(
    np.concatenate(cell_2_spikes, axis=0), bins=nbins)
ax1_PSTH.step(bins1[:-1], counts1, color=c2)

ax0_all.set_title("Cell 1", color=c1, y=1.02, weight='bold', fontsize=f_c_size)
ax1_all.set_title("Cell 2", color=c2, y=1.02, weight='bold', fontsize=f_c_size)
ax0_PSTH.set_xlabel('PSTH\nCell 1', color=c1, weight="bold", size=f_p_size)
ax1_PSTH.set_xlabel('PSTH\nCell 2', color=c2, weight="bold", size=f_p_size)

# add rectangles
rect0 = patches.Rectangle((-0.01, 0.5), 1.02, 0.51, linewidth=lw+1,
                          clip_on=False, edgecolor=c1, facecolor='none', linestyle="--")
rect1 = patches.Rectangle((-0.01, 0.5), 1.02, 0.51, linewidth=lw+1,
                          clip_on=False, edgecolor=c2, facecolor='none', linestyle="--")
ax0_all.add_patch(rect0)
ax1_all.add_patch(rect1)

# add arrows
xyA = [0.5, 0.475]
xyB = [0.5, 0.25]
arrow0 = patches.ConnectionPatch(xyA, xyB, coordsA=ax0_all.transData, coordsB=ax0_all.transData,
                                 color=c1, arrowstyle="-|>", mutation_scale=5, linewidth=2,)
fig.patches.append(arrow0)
xyB2 = [0.5, 0.325]
arrow1 = patches.ConnectionPatch(xyA, xyB2, coordsA=ax1_all.transData, coordsB=ax1_all.transData,
                                 color=c2, arrowstyle="-|>",   mutation_scale=5,  linewidth=2,)
fig.patches.append(arrow1)

##### Plot broad and narrow waveforms #####
ax_BS_waveform.plot(W_t, BS_W.values * 1000, color=palette_e[0])
ax_BS_waveform.set_xlabel("Time ($\\mathrm{\\mu}$s)")
ax_BS_waveform.set_ylabel("Voltage ($\\mathrm{\\mu}$V)")
ax_BS_waveform.set_title("BS")
ax_BS_waveform.set_xlim(waveform_xlim)
ax_NS_waveform.set_ylim(waveform_ylim)

ax_NS_waveform.plot(W_t, NS_W.values * 1000, color=palette_e[1])
ax_NS_waveform.set_xlabel("Time ($\\mathrm{\\mu}$s)")
ax_NS_waveform.set_title("NS")
ax_NS_waveform.set_xlim(waveform_xlim)
ax_NS_waveform.set_ylim(waveform_ylim)
ax_NS_waveform.spines["left"].set_visible(False)
ax_NS_waveform.set_yticks([])

##### width vs firing rate #####
ax_class.plot(df_extra_BS['tpwidth_2'], df_extra_BS['FRs'],
              '.', color=palette_e[0], label='BS', markersize=ms)
ax_class.plot(df_extra_NS['tpwidth_2'], df_extra_NS['FRs'],
              '.', color=palette_e[1], label='NS', markersize=ms)
ax_class.set_xlabel("Spike width ($\\mathrm{\\mu}$s)")
ax_class.set_ylabel("Firing rate (Hz)")
ax_class.set_yticks([0, 10, 20, 30])
ax_class.set_xticks([0, 250, 500])

##### BS examples #####
BS_cell_ind = np.argwhere(VGS_broad.columns == broad_ex_cell)[0][0]
t_shift_ex = sdf_t[0][np.argmax(VGS_broad[broad_ex_cell])]

# example raster
container = sio.loadmat('./data/exp/VGS_BS_ex_spike_times.mat')
VGS_BS_spikes = []
for i in range(container['VGS_BS_struct'].shape[0]):
    VGS_BS_spikes.append(container['VGS_BS_struct'][i][0][:, 0] - 750)
ax_BS_VGS_R.eventplot(VGS_BS_spikes, color=palette_e[0], linewidths=lw_R)
ax_BS_VGS_R.set_ylabel('Trials')
ax_BS_VGS_R.set_yticks([0, 70])

# PSTHs
sat = np.linspace(0.5, 0.95, VGS_broad.shape[0])
for i in range(VGS_broad.shape[1]):
    ax_BS_VGS_P.plot(sdf_t[0], VGS_broad.iloc[:, i],
                     alpha=alpha, color=lighten_color(palette_e[0], sat[i]))
ax_BS_VGS_P.plot(sdf_t[0], VGS_broad[broad_ex_cell],
                 color=palette_e[0], linewidth=lw)
ax_BS_VGS_P.set_ylabel('F (Hz)')

BS_cell_ind = np.argwhere(VGS_broad.columns == broad_ex_cell)[0][0]
t_shift_ex = sdf_t[0][np.argmax(VGS_broad[broad_ex_cell])]
y = exp_curve(t, df_coeff.loc[BS_cell_ind, 'VGS BS'],
              df_offset.loc[BS_cell_ind, 'VGS BS'],
              df_tau.loc[BS_cell_ind, 'VGS BS']/1000)
ax_BS_VGS_P.plot(t + t_shift_ex, y, color='tab:red',
                 linestyle="--", alpha=0.75)
ax_BS_VGS_P.set_xticks(xticks_psth)
ax_BS_VGS_P.set_xticklabels(xtick_labels_psth)

sat = np.linspace(0.5, 0.95, VGS_broad.shape[0])
for i in range(VGS_broad.shape[1]):
    t_shift = sdf_t[0][np.argmax(VGS_broad.iloc[:, i])]
    ax_BS_VGS_P_n.plot(sdf_t[0] - t_shift, VGS_broad.iloc[:, i] / np.max(
        VGS_broad.iloc[:, i]), alpha=alpha, color=lighten_color(palette_e[0], sat[i]))
ax_BS_VGS_P_n.plot(sdf_t[0] - t_shift_ex, VGS_broad[broad_ex_cell] /
                   np.max(VGS_broad[broad_ex_cell]), color=palette_e[0], linewidth=lw)
ax_BS_VGS_P_n.set_xlabel('Time (ms)')
ax_BS_VGS_P_n.set_ylabel('Norm F')
ax_BS_VGS_P_n.set_ylim(0, 1)
ax_BS_VGS_P_n.set_xticks(xticks_psth_n)
ax_BS_VGS_P_n.set_xticklabels(xtick_labels_psth_n)


##### NS examples #####
NS_cell_ind = np.argwhere(VGS_narrow.columns == narrow_ex_cell)[0][0]
t_shift_ex_NS = sdf_t[0][np.argmax(VGS_narrow[narrow_ex_cell])]

# example raster
container = sio.loadmat('./data/exp//VGS_NS_ex_spike_times.mat')
VGS_NS_spikes = []
for i in range(container['VGS_NS_struct'].shape[0]):
    VGS_NS_spikes.append(container['VGS_NS_struct'][i][0][:, 0] - 750)
ax_NS_VGS_R.eventplot(VGS_NS_spikes, color=palette_e[1], linewidths=lw_R)
ax_NS_VGS_R.set_ylabel('Trials')
ax_NS_VGS_R.set_yticks([0, 70])

# PSTHs
sat = np.linspace(0.5, 0.95, VGS_narrow.shape[0])
for i in range(VGS_narrow.shape[1]):
    ax_NS_VGS_P.plot(sdf_t[0], VGS_narrow.iloc[:, i],
                     alpha=alpha, color=lighten_color(palette_e[1], sat[i]))
ax_NS_VGS_P.plot(sdf_t[0], VGS_narrow[narrow_ex_cell], color=palette_e[1])
ax_NS_VGS_P.set_ylabel('F (Hz)', fontsize=10)


NS_cell_ind = np.argwhere(VGS_narrow.columns == narrow_ex_cell)[0][0]
t_shift2 = sdf_t[0][np.argmax(VGS_narrow[narrow_ex_cell])]
y2 = exp_curve(t, df_coeff.loc[NS_cell_ind, 'VGS NS'],
               df_offset.loc[NS_cell_ind, 'VGS NS'],
               df_tau.loc[NS_cell_ind, 'VGS NS']/1000)
ax_NS_VGS_P.plot(t + t_shift2, y2, color='tab:red', linestyle="--", alpha=0.75)
ax_NS_VGS_P.set_xticks(xticks_psth)
ax_NS_VGS_P.set_xticklabels(xtick_labels_psth)

sat = np.linspace(0.5, 0.95, VGS_narrow.shape[0])
for i in range(VGS_narrow.shape[1]):
    if i == 6:
        t_shift = sdf_t[0][VGS_narrow.iloc[:, i].nlargest(2).index[1]]
    else:
        t_shift = sdf_t[0][np.argmax(VGS_narrow.iloc[:, i])]
    ax_NS_VGS_P_n.plot(sdf_t[0] - t_shift, VGS_narrow.iloc[:, i] / np.max(
        VGS_narrow.iloc[:, i]), alpha=alpha, color=lighten_color(palette_e[1], sat[i]))
ax_NS_VGS_P_n.plot(sdf_t[0] - t_shift_ex_NS, VGS_narrow[narrow_ex_cell] /
                   np.max(VGS_narrow[narrow_ex_cell]), color=palette_e[1])
ax_NS_VGS_P_n.set_xlabel('Time (ms)')
ax_NS_VGS_P_n.set_ylabel('Norm F', fontsize=10)
ax_NS_VGS_P_n.set_ylim(0, 1)
ax_NS_VGS_P_n.set_xticks(xticks_psth_n)
ax_NS_VGS_P_n.set_xticklabels(xtick_labels_psth_n)

# labels
ax_BS_VGS_Lab.set_title('BS', fontsize=14)
ax_NS_VGS_Lab.set_title('NS', fontsize=14)

##### Coefficient #####
sns.boxplot(data=df_coeff[['VGS BS', 'VGS NS']], ax=ax_A, saturation=0.5,
            order=order_e, palette=palette_e,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_coeff[['VGS BS', 'VGS NS']],
              ax=ax_A, palette=palette_e,  order=order_e, s=ms)
ax_A.set_ylabel("Coefficient (Hz)")
ax_A.set_xticks(ax_A.get_xticks())
ax_A.set_xticklabels(ax_A.get_xticklabels(), rotation=0, ha='center')

annotator = Annotator(ax_A, pairs1, data=df_coeff[['VGS BS', 'VGS NS']])
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_A, test_results1 = annotator.annotate()
ax_A.set_yticks([0, 25, 50, 75])

##### Offset #####
sns.boxplot(data=df_offset[['VGS BS', 'VGS NS']], ax=ax_B, saturation=0.5,
            order=order_e, palette=palette_e,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_offset[['VGS BS', 'VGS NS']],
              ax=ax_B, palette=palette_e,  order=order_e, s=ms)
ax_B.set_ylabel("Baseline (Hz)")
ax_B.set_xticks(ax_B.get_xticks())
ax_B.set_xticklabels(ax_B.get_xticklabels(), rotation=0, ha='center')
annotator = Annotator(ax_B, pairs1, data=df_offset[['VGS BS', 'VGS NS']])
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_B, test_results1 = annotator.annotate()
ax_B.set_yticks([0, 10, 20,  30])

#####  Offset vs Coefficient #####
for i in range(0, 2):
    ax_A_B.plot(df_offset[order_e[i]], df_coeff[order_e[i]],
                '.', color=palette_e[i], label=order_e[i], markersize=ms)
ax_A_B.set_ylabel('Coefficient (Hz)')
ax_A_B.set_xlabel("Baseline (Hz)")
ax_A_B.set_yticks([0, 25, 50])

rho_BS, p_BS = stats.spearmanr(
    df_offset[order_e[0]], df_coeff[order_e[0]], nan_policy='omit')
rho_NS, p_NS = stats.spearmanr(
    df_offset[order_e[1]], df_coeff[order_e[1]], nan_policy='omit')
print("BS rho = {}, p = {}".format(rho_BS, p_BS))
print("NS rho = {}, p = {}".format(rho_NS, p_NS))


##### Adaptation index #####
sns.boxplot(data=df_AI[['VGS BS', 'VGS NS']], ax=ax_AI, saturation=0.5,
            order=order_e, palette=palette_e,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_AI[['VGS BS', 'VGS NS']], ax=ax_AI,
              palette=palette_e,  order=order_e, s=ms)
ax_AI.set_ylabel("Adaptation Index")
ax_AI.set_xticks(ax_AI.get_xticks())
ax_AI.set_xticklabels(ax_AI.get_xticklabels(), rotation=0, ha='center')
annotator = Annotator(ax_AI, pairs1, data=df_AI[['VGS BS', 'VGS NS']])
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_AI, test_results1 = annotator.annotate()
ax_AI.set_yticks([0, 0.25, 0.5, 0.75])

#####  Decay tau #####
sns.boxplot(data=np.log10(-1 / (df_tau[['VGS BS', 'VGS NS']]/1000)), ax=ax_tau, order=order_e, palette=palette_e, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=np.log10(-1 / (df_tau[['VGS BS', 'VGS NS']]/1000)),
              ax=ax_tau, order=order_e, palette=palette_e, s=ms)
ax_tau.set_ylabel("ISFA $\\tau$ (ms)")
ax_tau.spines["right"].set_visible(False)
ax_tau.spines["top"].set_visible(False)
ax_tau.set_xticks(ax_tau.get_xticks())
ax_tau.set_xticklabels(ax_tau.get_xticklabels(), rotation=0, ha='center')
ymin, ymax = ax_tau.get_ylim()
tick_range = np.arange(np.floor(ymin), ymax)
ax_tau.yaxis.set_ticks(tick_range)
ax_tau.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
annotator = Annotator(ax_tau, pairs1, data=np.log10(-1 /
                      (df_tau[['VGS BS', 'VGS NS']]/1000)))
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_tau, test_results1 = annotator.annotate()
# 10, 100, 1000 ticklabe;s
ax_tau.set_yticklabels(["{:.0f}".format(i) for i in 10**(ax_tau.get_yticks())])
ax_tau.set_ylim(tau_ylim)

##### Latency #####
sns.boxplot(data=df_lat, x='Type', y='Latency', ax=ax_latency, order=order_e, palette=palette_e, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_lat, x='Type', y='Latency',  ax=ax_latency,
              order=order_e, palette=palette_e, s=ms)
annotator = Annotator(ax_latency, pairs1, data=df_lat, x='Type', y='Latency',)
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_latency, test_results1 = annotator.annotate()
ax_latency.set_xlabel("")
ax_latency.set_ylabel("Response \nLatency (ms)")

##### Tmax #####
sns.boxplot(data=df_lat_max, x='Type', y='PSTH max Latency', ax=ax_latency_max, order=order_e, palette=palette_e, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_lat_max, x='Type', y='PSTH max Latency',
              ax=ax_latency_max, order=order_e, palette=palette_e, s=ms)
annotator = Annotator(ax_latency_max, pairs1,
                      data=df_lat_max, x='Type', y='PSTH max Latency',)
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_latency_max, test_results1 = annotator.annotate()
ax_latency_max.set_xlabel("")
ax_latency_max.set_ylabel('$\mathrm{T}_{\mathrm{Max}}$ (ms)')
ax_latency_max.set_yticks([0, 200, 400, 600])

# remove axis box sides
for ax in [ax_BS_VGS_Lab, ax_NS_VGS_Lab]:
    ax = remove_axis_box(ax, ["right", "top", "left", "bottom"])

for ax in [ax_BS_VGS_R, ax_NS_VGS_R, ax_BS_VGS_P, ax_BS_VGS_P_n, ax_NS_VGS_P, ax_NS_VGS_P_n, ax_tau, ax_A, ax_B, ax_A_B, ax_latency, ax_BS_waveform, ax_NS_waveform, ax_class, ax_AI, ax_latency_max]:
    ax = remove_axis_box(ax, ["right", "top"])

for ax in [ax_BS_VGS_R, ax_NS_VGS_R, ]:
    ax = remove_axis_box(ax, ["bottom"])

for ax in [ax0_all, ax1_all, ax0_R, ax1_R, ax0_PSTH, ax1_PSTH]:
    ax = remove_axis_box(ax, ["right", "top", "left", "bottom"])

# set axis limits
for ax in [ax_BS_VGS_R, ax_NS_VGS_R, ax_BS_VGS_P, ax_NS_VGS_P]:
    ax.set_ylim(0, 70)
    ax.set_xlim(sdf_xlim)

for ax in [ax_BS_VGS_P_n,  ax_NS_VGS_P_n]:
    ax.set_xlim(sdf_xlim_norm)

for ax in [ax0_PSTH, ax1_PSTH]:
    ax.set_xlim((0, tend))
    ax.set_ylim((0, 120))

# set plot labels
for ax in [ax_tau, ax_A, ax_B, ax_latency, ax_AI, ax_latency_max]:
    ax.set_xticklabels(pair_xlabels)

##### subplot letters #####
ax_BS_waveform.text(-0.05, 1.075, string.ascii_uppercase[0],
                    transform=ax_BS_waveform.transAxes, size=12, weight='bold')
ax_class.text(-0.05, 1.075, string.ascii_uppercase[1],
              transform=ax_class.transAxes, size=12, weight='bold')
ax0_all.text(-0.45, 1.075, string.ascii_uppercase[2],
             transform=ax0_all.transAxes, size=12, weight='bold')
ax_BS_VGS_R.text(-0.025, 1.15, string.ascii_uppercase[3],
                 transform=ax_BS_VGS_R.transAxes, size=12, weight='bold')
ax_NS_VGS_R.text(-0.025, 1.15, string.ascii_uppercase[4],
                 transform=ax_NS_VGS_R.transAxes, size=12, weight='bold')
ax_latency_max.text(-0.05, 1.085, string.ascii_uppercase[5],
                    transform=ax_latency_max.transAxes, size=12, weight='bold')
ax_A.text(-0.025, 1.085, string.ascii_uppercase[6],
          transform=ax_A.transAxes, size=12, weight='bold')
ax_B.text(-0.05, 1.085, string.ascii_uppercase[7],
          transform=ax_B.transAxes, size=12, weight='bold')
ax_A_B.text(-0.05, 1.085, string.ascii_uppercase[8],
            transform=ax_A_B.transAxes, size=12, weight='bold')

ax_AI.text(-0.025, 1.085, string.ascii_uppercase[9],
           transform=ax_AI.transAxes, size=12, weight='bold')
ax_tau.text(-0.025, 1.085, string.ascii_uppercase[10],
            transform=ax_tau.transAxes, size=12, weight='bold')
ax_latency.text(-0.05, 1.085, string.ascii_uppercase[11],
                transform=ax_latency.transAxes, size=12, weight='bold')

##### save and show #####
plt.savefig('./figures/Figure_2_in_vivo_adaptation.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Figure_2_in_vivo_adaptation.pdf',
            bbox_inches='tight', dpi=300)
plt.show()
