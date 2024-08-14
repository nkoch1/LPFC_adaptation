# %%
import os
os.chdir('../../')
from src.plt_fxns import lighten_color, remove_axis_box, generate_spike_train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import matplotlib.patches as patches
import string
plt.rcParams["font.family"] = "Arial"

# %% load data
t = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_PSTH_T_n_70_fit_1000_psth_50.csv', header=None)
VGS_broad = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_VGS_BS_PSTH_n_70_fit_1000_psth_50.csv')
VGS_narrow = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_VGS_NS_PSTH_n_70_fit_1000_psth_50.csv')

df_offset = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_offset_PSTH_n_70_fit_1000_psth_50.csv')
df_coeff = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_coeff_PSTH_n_70_fit_1000_psth_50.csv')
df_tau = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_decay_PSTH_n_70_fit_1000_psth_50.csv')
df_lat = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_latency_PSTH_n_70_fit_1000_psth_50.csv')
df_lat = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_latency_PSTH_n_70_fit_1000_psth_50_sdf_5.csv')
df_tmax = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_Tmax_PSTH_n_70_fit_1000_psth_50.csv')


ind = df_tmax <= 1000
df_offset = df_offset[ind]
df_coeff = df_coeff[ind]
df_tau = df_tau[ind]
df_lat = df_lat[ind]
df_tmax = df_tmax[ind]
df_AI = df_offset / (df_offset + df_coeff)

# %% simulated example data


numtrials = 3
num_cells = 150
tend = 400
tstim = 150
stim_num_spikes = 4
nbins = 70
F0 = 20
F1 = 10

trial_1 = []
trial_2 = []
trial_3 = []
for i in range(0, num_cells):
    cell_i_spikes = generate_spike_train(numtrials, i, F0+F0*0.2*np.random.rand(), F1+F1*0.25*np.random.rand(),
                                         tend, tstim+tstim*0.2*np.random.rand(), stim_num_spikes + np.random.randint(-2, high=5, size=None, dtype=int))
    trial_1.append(cell_i_spikes[0])
    trial_2.append(cell_i_spikes[1])
    trial_3.append(cell_i_spikes[2])


F0 = 20
F1 = 10
cell_1_spikes = generate_spike_train(
    numtrials, 0, F0, F1, tend, tstim, stim_num_spikes)

F0 = 15
F1 = 7
cell_2_spikes = generate_spike_train(
    numtrials, 1, F0, F1, tend, tstim, stim_num_spikes)

# %% Create Figure

np.random.seed(0)  # set seed for stripplot jitter
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choices #####
alpha = 0.5
ms = 2
lw = 2
stim_start = 750.
palette_e = ['#7DA54B', '#007030']
order_e = ['VGS BS', 'VGS NS']
pairs1 = [('VGS BS', 'VGS NS'), ]
psth_xlim = (-500, 1720)
psth_xlim = (-500, 1500)
x_let, y_let = 0.65, 0.9
c1 = (120/255, 120/255, 120/255)
c2 = (82/255, 80/255, 80/255)
p = '#404040'
lw_R = 1.05

xticks_psth = [-500, 0, 500, 1000, 1500]
xtick_labels_psth = ['-500', 'Stim.\nOnset', '500', '1000', '1500']
xticks_psth_n = [-500, 0, 500, 1000, 1500]
xtick_labels_psth_n = [
    '-500', '$\\mathrm{T}_{\\mathrm{Max}}$', '500', '1000', '1500']

##### layout #####
fig = plt.figure(figsize=(6.5, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[
                      0.55, 0.225, 0.225], hspace=0.5,  wspace=0.7)
gs_L = gridspec.GridSpecFromSubplotSpec(9, 1, wspace=0.5, hspace=0.25, height_ratios=[
                                        0.25, 0.1, 0.25, 0.075, 0.25, 0.2, 0.25, 0.075, 0.25], subplot_spec=gs[0])
gs_R = gridspec.GridSpecFromSubplotSpec(
    4, 1,  wspace=0.8, hspace=0.6, subplot_spec=gs[1])
gs_R2 = gridspec.GridSpecFromSubplotSpec(
    3, 1,  wspace=0.8, hspace=0.6, subplot_spec=gs[2])
gs_s = gridspec.GridSpecFromSubplotSpec(3, 4, hspace=0.2, wspace=0.25, width_ratios=[
                                        0.3, 0.3, 0.2, 0.3], subplot_spec=gs_L[0])
ax0_R = fig.add_subplot(gs_s[:, 0])
ax1_R = fig.add_subplot(gs_s[:, 1])
ax2_R = fig.add_subplot(gs_s[:, 2])
ax_allR = fig.add_subplot(gs_s[:, :3])
ax_allR.patch.set_alpha(0.)
ax_PSTH0 = fig.add_subplot(gs_s[0, 3])
ax_PSTH1 = fig.add_subplot(gs_s[1, 3])
ax_PSTH2 = fig.add_subplot(gs_s[2, 3])

ax_VGS_BS = fig.add_subplot(gs_L[2])
ax_VGS_BS_s = fig.add_subplot(gs_L[4])
ax_VGS_NS = fig.add_subplot(gs_L[6])
ax_VGS_NS_s = fig.add_subplot(gs_L[8])
ax_A = fig.add_subplot(gs_R[1])
ax_B = fig.add_subplot(gs_R[2])
ax_A_B = fig.add_subplot(gs_R[3])
ax_latency_max = fig.add_subplot(gs_R[0])

ax_AI = fig.add_subplot(gs_R2[0])
ax_tau = fig.add_subplot(gs_R2[1])
ax_latency = fig.add_subplot(gs_R2[2])


## plot sim data for PSTH diagram ############################
ax0_R.eventplot(cell_1_spikes, color=c1, linewidths=lw_R)
ax0_R.set_ylim((0.5, numtrials+0.5))

ax1_R.eventplot(cell_2_spikes,  color=c2, linewidths=lw_R)
ax1_R.set_ylim((0.5, numtrials+0.5))

f_c_size = 9
f_p_size = 7
ax0_R.set_title("Cell 1", color=c1, y=1.02, fontsize=f_c_size)
ax1_R.set_title("Cell 2", color=c2, y=1.02, fontsize=f_c_size)
ax2_R.set_title("...Cell N", color='k', y=1.02, x=0.275, fontsize=f_c_size)
ax_PSTH0.text(0.55, 0.05, "Trial 1 PSTH", ha="center", va="center",
              color='k', transform=ax_PSTH0.transAxes, fontsize=f_p_size)
ax_PSTH1.text(0.55, 0.05, "Trial 2 PSTH", ha="center", va="center",
              color='k', transform=ax_PSTH1.transAxes, fontsize=f_p_size)
ax_PSTH2.text(0.55, 0.05, "Trial 3 PSTH", ha="center", va="center",
              color='k', transform=ax_PSTH2.transAxes, fontsize=f_p_size)

counts0, bins0 = np.histogram(np.concatenate(trial_1, axis=0), bins=nbins)
ax_PSTH0.step(bins0[:-1], counts0, color=p)
counts1, bins1 = np.histogram(np.concatenate(trial_2, axis=0), bins=nbins)
ax_PSTH1.step(bins1[:-1], counts1, color=p)
counts2, bins2 = np.histogram(np.concatenate(trial_3, axis=0), bins=nbins)
ax_PSTH2.step(bins2[:-1], counts2, color=p)

# add boxes
rect2 = patches.Rectangle((-0.01, 0.), 0.96, 1,  linewidth=2, clip_on=False,
                          edgecolor=p, facecolor='none', linestyle=":", zorder=10)
ax_allR.add_patch(rect2)

l1 = patches.Arrow(-0.01, 0.33, 0.96, 0, linewidth=2,
                   color=p, linestyle=":", zorder=10, width=0)
ax_allR.add_patch(l1)
l2 = patches.Arrow(-0.01, 0.66, 0.96, 0, linewidth=2,
                   color=p, linestyle=":", zorder=10, width=0)
ax_allR.add_patch(l2)


# add arrows
xyA = [0.96, 0.825]
xyB = [1.05, 0.825]
arrow0 = patches.ConnectionPatch(xyA, xyB, coordsA=ax_allR.transData, coordsB=ax_allR.transData,
                                 color=p, arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow0)

xyA2 = [0.96, 0.495]
xyB2 = [1.05, 0.495]
arrow1 = patches.ConnectionPatch(xyA2, xyB2, coordsA=ax_allR.transData, coordsB=ax_allR.transData,
                                 color=p, arrowstyle="-|>",   mutation_scale=10,  linewidth=2,)
fig.patches.append(arrow1)

xyA3 = [0.96, 0.165]
xyB3 = [1.05, 0.165]
arrow2 = patches.ConnectionPatch(xyA3, xyB3, coordsA=ax_allR.transData, coordsB=ax_allR.transData,
                                 color=p, arrowstyle="-|>",   mutation_scale=10,  linewidth=2,)
fig.patches.append(arrow2)
#######################################################


##### PSTHs #####
sat = np.linspace(0.5, 1.25, VGS_broad.shape[0])
for i in range(VGS_broad.shape[0]):
    ax_VGS_BS.plot(t.iloc[0, :] - stim_start, VGS_broad.iloc[i, :],
                   alpha=alpha, color=lighten_color(palette_e[0], sat[i]))
ax_VGS_BS.plot(t.iloc[0, :] - stim_start, VGS_broad.mean(),
               alpha=alpha, color=lighten_color(palette_e[0], 1.5))

for i in range(VGS_broad.shape[0]):
    t_shift = t.iloc[0, np.argmax(VGS_broad.iloc[i, :])]
    ax_VGS_BS_s.plot(t.iloc[0, :] - t_shift, VGS_broad.iloc[i, :],
                     alpha=alpha, color=lighten_color(palette_e[0], sat[i]))
ax_VGS_BS_s.set_xlabel('Time (ms)')
t_shift = t.iloc[0, np.argmax(VGS_broad.mean())]
ax_VGS_BS_s.plot(t.iloc[0, :] - t_shift, VGS_broad.mean(),
                 alpha=alpha, color=lighten_color(palette_e[0], 1.5))


sat = np.linspace(0.75, 1.15, VGS_narrow.shape[0])
for i in range(VGS_narrow.shape[0]):
    ax_VGS_NS.plot(t.iloc[0, :] - stim_start, VGS_narrow.iloc[i, :],
                   alpha=alpha, color=lighten_color(palette_e[1], sat[i]))
ax_VGS_NS.plot(t.iloc[0, :] - stim_start, VGS_narrow.mean(),
               alpha=alpha, color='k')  # lighten_color(palette_e[1], 1.2))


for i in range(VGS_narrow.shape[0]):
    t_shift = t.iloc[0, np.argmax(VGS_narrow.iloc[i, :])]
    ax_VGS_NS_s.plot(t.iloc[0, :] - t_shift, VGS_narrow.iloc[i, :],
                     alpha=alpha, color=lighten_color(palette_e[1], sat[i]))
ax_VGS_NS_s.set_xlabel('Time (ms)')
t_shift = t.iloc[0, np.argmax(VGS_narrow.mean())]
ax_VGS_NS_s.plot(t.iloc[0, :] - t_shift,
                 VGS_narrow.mean(), alpha=alpha, color='k')

# set label
ax_VGS_BS.set_title('BS', fontsize=12, y=0.9)
ax_VGS_NS.set_title('NS', fontsize=12, y=0.9)

##### Decay tau #####
sns.boxplot(data=np.log10(-1 / (df_tau[['VGS BS', 'VGS NS']]/1000)), ax=ax_tau, order=order_e, palette=palette_e, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=np.log10(-1 / (df_tau[['VGS BS', 'VGS NS']]/1000)),
              ax=ax_tau, order=order_e, palette=palette_e, s=ms)
ax_tau.set_ylabel("ESFA $\\tau$ (ms)")
ax_tau.set_xticks(ax_tau.get_xticks())
ax_tau.set_xticklabels(ax_tau.get_xticklabels(), rotation=0, ha='center')
ymin, ymax = ax_tau.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_tau.yaxis.set_ticks(tick_range)
tick_range_ticks = np.append([1], tick_range[:-1])
ax_tau.yaxis.set_ticks([np.log10(x) for p in tick_range_ticks for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_tau.yaxis.set_ticklabels(['$10^{%s}$' % (round(i)) for i in tick_range])
annotator = Annotator(ax_tau, pairs1, data=np.log10(-1 /
                      (df_tau[['VGS BS', 'VGS NS']]/1000)))
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_tau, test_results1 = annotator.annotate()
ax_tau.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
ax_tau.set_xticklabels(['BS', 'NS'], rotation=0, ha='center')

##### Coefficient #####
sns.boxplot(data=df_coeff, ax=ax_A, saturation=0.5, order=order_e,
            palette=palette_e,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_coeff, ax=ax_A, palette=palette_e,  order=order_e, s=ms)
ax_A.set_ylabel("Coefficient (Hz)")
ax_A.set_xticks(ax_A.get_xticks())
ax_A.set_xticklabels(ax_A.get_xticklabels(), rotation=0, ha='center')
annotator = Annotator(ax_A, pairs1, data=df_coeff)
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_A, test_results1 = annotator.annotate()
ax_A.set_xticklabels(['BS', 'NS'], rotation=0, ha='center')

##### offset #####
sns.boxplot(data=df_offset, ax=ax_B, saturation=0.5, order=order_e,
            palette=palette_e,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_offset, ax=ax_B, palette=palette_e,  order=order_e, s=ms)
ax_B.set_ylabel("Baseline (Hz)")
ax_B.set_xticks(ax_B.get_xticks())
ax_B.set_xticklabels(ax_B.get_xticklabels(), rotation=0, ha='center')
annotator = Annotator(ax_B, pairs1, data=df_offset)
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_B, test_results1 = annotator.annotate()
ax_B.set_xticklabels(['BS', 'NS'], rotation=0, ha='center')


##### Baseline vs Coefficient #####
for i in range(0, 2):
    ax_A_B.plot(df_offset[order_e[i]], df_coeff[order_e[i]],
                '.', color=palette_e[i], label=order_e[i], markersize=4)
ax_A_B.set_ylabel('Coefficient (Hz)')
ax_A_B.set_xlabel("Baseline (Hz)")

rho_BS, p_BS = stats.spearmanr(
    df_offset[order_e[0]], df_coeff[order_e[0]], nan_policy='omit')
rho_NS, p_NS = stats.spearmanr(
    df_offset[order_e[1]], df_coeff[order_e[1]], nan_policy='omit')
print("BS rho = {}, p = {}".format(rho_BS, p_BS))
print("NS rho = {}, p = {}".format(rho_NS, p_NS))

##### Latency #####
sns.boxplot(data=df_lat, ax=ax_latency, order=order_e, palette=palette_e, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_lat, ax=ax_latency,
              order=order_e, palette=palette_e, s=ms)
annotator = Annotator(ax_latency, pairs1, data=df_lat)
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_latency, test_results1 = annotator.annotate()
ax_latency.set_ylabel('Latency (ms)')
ax_latency.set_xticks(ax_latency.get_xticks())
ax_latency.set_xticklabels(
    ax_latency.get_xticklabels(), rotation=0, ha='center')
ax_latency.set_xticklabels(['BS', 'NS'], rotation=0, ha='center')
ax_latency.set_ylim(0, ax_latency.get_ylim()[1])


##### Tmax #####
sns.boxplot(data=df_tmax, ax=ax_latency_max, order=order_e, palette=palette_e, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_tmax, ax=ax_latency_max,
              order=order_e, palette=palette_e, s=ms)
annotator = Annotator(ax_latency_max, pairs1, data=df_tmax)
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_latency_max, test_results1 = annotator.annotate()
ax_latency_max.set_ylabel('$\mathrm{T}_{\mathrm{Max}}$ (ms)')
ax_latency_max.set_xticks(ax_latency_max.get_xticks())
ax_latency_max.set_xticklabels(
    ax_latency_max.get_xticklabels(), rotation=0, ha='center')
ax_latency_max.set_xticklabels(['BS', 'NS'], rotation=0, ha='center')


##### Adaptation index #####
sns.boxplot(data=df_AI, ax=ax_AI, saturation=0.5, order=order_e,
            palette=palette_e,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_AI, ax=ax_AI, palette=palette_e,  order=order_e, s=ms)
ax_AI.set_ylabel("Adaptation Index")
ax_AI.set_xticks(ax_AI.get_xticks())
ax_AI.set_xticklabels(ax_AI.get_xticklabels(), rotation=0, ha='center')
annotator = Annotator(ax_AI, pairs1, data=df_AI[['VGS BS', 'VGS NS']])
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_AI, test_results1 = annotator.annotate()
ax_AI.set_xticklabels(['BS', 'NS'], rotation=0, ha='center')


##### remove axis box sides #####
for ax in [ax_VGS_BS_s, ax_VGS_NS_s, ax_A, ax_B, ax_A_B, ax_AI, ax_tau, ax_latency, ax_latency_max]:
    ax = remove_axis_box(ax, ["right", "top"])

for ax in [ax0_R, ax1_R, ax2_R, ax_allR, ax_PSTH0, ax_PSTH1, ax_PSTH2]:
    ax = remove_axis_box(ax, ["right", "top", "left", "bottom"])

##### set axis limits & add labels #####
for ax in [ax_VGS_BS, ax_VGS_NS]:
    ax.set_xlim(psth_xlim)
    ax = remove_axis_box(ax, ["right", "top"])
    ax.set_xticks(xticks_psth)
    ax.set_xticklabels(xtick_labels_psth)

for ax in [ax_VGS_BS_s, ax_VGS_NS_s]:
    ax.set_xlim(psth_xlim)
    ax.set_ylabel('                       Avg Firing (Hz)', ha='center')
    ax.set_xticks(xticks_psth_n)
    ax.set_xticklabels(xtick_labels_psth_n)

for ax in [ax_PSTH0, ax_PSTH1, ax_PSTH2]:
    ax.set_xlim((50, tend))
    ax.set_ylim((5, 80))


##### subplot letters #####
ax_allR.text(-0.075, 1.125, string.ascii_uppercase[0],
             transform=ax_allR.transAxes, size=12, weight='bold')
ax_VGS_BS.text(-0.025, 1.075, string.ascii_uppercase[1],
               transform=ax_VGS_BS.transAxes, size=12, weight='bold')
ax_VGS_NS.text(-0.025, 1.075, string.ascii_uppercase[2],
               transform=ax_VGS_NS.transAxes, size=12, weight='bold')
ax_latency_max.text(-0.05, 1.075, string.ascii_uppercase[3],
                    transform=ax_latency_max.transAxes, size=12, weight='bold')
ax_A.text(-0.05, 1.075, string.ascii_uppercase[4],
          transform=ax_A.transAxes, size=12, weight='bold')
ax_B.text(-0.05, 1.075, string.ascii_uppercase[5],
          transform=ax_B.transAxes, size=12, weight='bold')
ax_A_B.text(-0.05, 1.075, string.ascii_uppercase[6],
            transform=ax_A_B.transAxes, size=12, weight='bold')
ax_AI.text(-0.05, 1.075, string.ascii_uppercase[7],
           transform=ax_AI.transAxes, size=12, weight='bold')
ax_latency.text(-0.05, 1.075, string.ascii_uppercase[9],
                transform=ax_latency.transAxes, size=12, weight='bold')
ax_tau.text(-0.05, 1.075, string.ascii_uppercase[8],
            transform=ax_tau.transAxes, size=12, weight='bold')


##### save and show #####
plt.savefig('./figures/Figure_3_in_vivo_pop_adaptation.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Figure_3_in_vivo_pop_adaptation.pdf',
            bbox_inches='tight', dpi=300)
plt.show()

# %%
