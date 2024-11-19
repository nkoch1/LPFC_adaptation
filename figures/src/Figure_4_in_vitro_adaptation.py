# %%
import os
os.chdir('../../')
import scipy
import string
import scipy.io as sio
import matplotlib.gridspec as gridspec
from statannotations.Annotator import Annotator
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import lighten_color, remove_axis_box
plt.rcParams["font.family"] = "Arial"

# %% load data
df_IC = pd.read_csv('./data/exp/ICTable.csv')
di = {'A': 1, 'S': 2}
df_IC['Cell type'] = df_IC['type']
df_IC['Cell type'] = df_IC['Cell type'].fillna(0)
df_IC.replace({"Cell type": di}, inplace=True)

df_IC['Cell type names'] = df_IC['type']
df_IC['Cell type names'] = df_IC['Cell type'].fillna(0)
di2 = {0: 'Unlabeled', 'A': 'Aspiny', 'S': 'Spiny'}
df_IC.replace({"Cell type names": di2}, inplace=True)
patch_BS = sio.loadmat('./data/exp/patch_BS_t_F.mat')
patch_NS = sio.loadmat('./data/exp/patch_NS_t_F.mat')


# latency df
df_laten = pd.read_csv('./data/exp/Latency_extraction_patch_spike_t.csv')
upper_ind = (df_laten['Initial Spike t'] >= 0.8125)
lower_ind = (df_laten['Initial Spike t'] < 0.8125)
df_laten.loc[df_laten.index[lower_ind], 'Initial Spike t'] = (
    df_laten.loc[df_laten.index[lower_ind], 'Initial Spike t'] - 0.656) * 1000
df_laten.loc[df_laten.index[upper_ind], 'Initial Spike t'] = (
    df_laten.loc[df_laten.index[upper_ind], 'Initial Spike t'] - 0.8125) * 1000
df = pd.read_csv('./data/exp/Patch_exp_decay_fit_NS_BS.csv')


# tau df (log10)
df_tau_BS = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_tau_BS['decay $\tau$'] = np.log10(-1/(df['Patch BS Exponent'] / 1000))
df_tau_BS['condition'] = ['BS' for i in range(
    0, len(df['Patch BS Exponent']-1))]
df_tau_BS = df_tau_BS[df_tau_BS['decay $\tau$'] <= 5]
df_tau_NS = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_tau_NS['decay $\tau$'] = np.log10(-1/(df['Patch NS Exponent'] / 1000))
df_tau_NS['condition'] = ['NS' for i in range(
    0, len(df['Patch NS Exponent']-1))]
df_tau = pd.concat([df_tau_BS, df_tau_NS])

# AI df
df_AI_BS = pd.DataFrame(columns=['AI', 'condition'])
df_AI_BS['Adaptation Index'] = df['Patch BS AI']
df_AI_BS['condition'] = ['BS' for i in range(0, len(df['Patch BS AI']-1))]
df_AI_NS = pd.DataFrame(columns=['AI', 'condition'])
df_AI_NS['Adaptation Index'] = df['Patch NS AI']
df_AI_NS['condition'] = ['NS' for i in range(0, len(df['Patch NS AI']-1))]
df_AI = pd.concat([df_AI_BS, df_AI_NS])

# Coefficient df
df_coeff_BS = pd.DataFrame(columns=['Coefficient', 'condition'])
df_coeff_BS['Coefficient'] = df['Patch BS Coefficient']
df_coeff_BS['condition'] = ['BS' for i in range(
    0, len(df['Patch BS Exponent']-1))]
df_coeff_NS = pd.DataFrame(columns=['Coefficient', 'condition'])
df_coeff_NS['Coefficient'] = df['Patch NS Coefficient']
df_coeff_NS['condition'] = ['NS' for i in range(
    0, len(df['Patch NS Exponent']-1))]
df_coeff = pd.concat([df_coeff_BS, df_coeff_NS])


# Offset df
df_off_BS = pd.DataFrame(columns=['Offset', 'condition'])
df_off_BS['Offset'] = df['Patch BS Offset']
df_off_BS['condition'] = ['BS' for i in range(
    0, len(df['Patch BS Exponent']-1))]
df_off_NS = pd.DataFrame(columns=['Offset', 'condition'])
df_off_NS['Offset'] = df['Patch NS Offset']
df_off_NS['condition'] = ['NS' for i in range(
    0, len(df['Patch NS Exponent']-1))]
df_off = pd.concat([df_off_BS, df_off_NS])


# coeff/offset df
df_cof_off_BS = pd.DataFrame(columns=['Coeff/Offset', 'condition'])
df_cof_off_BS['Coeff/Offset'] = df['Patch BS Coefficient'] / \
    df['Patch BS Offset']
df_cof_off_BS['condition'] = [
    'BS' for i in range(0, len(df['Patch BS Exponent']-1))]
df_cof_off_NS = pd.DataFrame(columns=['Offset', 'condition'])
df_cof_off_NS['Coeff/Offset'] = df['Patch NS Coefficient'] / \
    df['Patch NS Offset']
df_cof_off_NS['condition'] = [
    'NS' for i in range(0, len(df['Patch NS Exponent']-1))]
df_cof_off = pd.concat([df_cof_off_BS, df_cof_off_NS])


# %%
FS_ex_cell = 'M10_JS_A1_C22'
FS_ex_df = pd.read_json(
    './data/exp_pro/dlPFC_FS/{}_AP_phase_analysis.json'.format(FS_ex_cell))
FS_stimVal = df_IC.loc[df_IC.index[df_IC.cells ==
                                   FS_ex_cell], 'stimVal'].values[0]
FS_ex_ind = FS_ex_df.index[FS_ex_df['I mag'] == FS_stimVal]


Pyr_ex_cell = 'M05_SM_A1_C02'
Pyr_ex_df = pd.read_json(
    './data/exp_pro/dlPFC_Pyr/{}_AP_phase_analysis.json'.format(Pyr_ex_cell))
Pyr_stimVal = df_IC.loc[df_IC.index[df_IC.cells ==
                                    Pyr_ex_cell], 'stimVal'].values[0]
Pyr_ex_ind = Pyr_ex_df.index[Pyr_ex_df['I mag'] == Pyr_stimVal]

plot_dt = 0.005
plot_dt_a = 0.01
min_spike_height = 10.
prominence = 10.
distance = 0.001

Pyr_t = np.array(Pyr_ex_df.loc[Pyr_ex_ind, 't'].values[0]) - \
    Pyr_ex_df.loc[Pyr_ex_ind, 't'].values[0][0]
Pyr_V = Pyr_ex_df.loc[Pyr_ex_ind, 'V'].values[0]
peaks, peak_prop = scipy.signal.find_peaks(
    Pyr_V,  height=min_spike_height, prominence=prominence)
spike_times_Pyr = Pyr_t[peaks]
Pyr_tstart = spike_times_Pyr[-2] - plot_dt
Pyr_tend = spike_times_Pyr[-2] + plot_dt_a
Pyr_tplot = (np.array(Pyr_ex_df.loc[Pyr_ex_ind, 't'].values[0]) -
             Pyr_ex_df.loc[Pyr_ex_ind, 't'].values[0][0] - spike_times_Pyr[-2]) * 1000
Pyr_tstart_ind = np.argwhere(Pyr_t <= Pyr_tstart)[-1][0]
Pyr_tend_ind = np.argwhere(Pyr_t >= Pyr_tend)[0][0] - 1


FS_t = np.array(FS_ex_df.loc[FS_ex_ind, 't'].values[0]) - \
    FS_ex_df.loc[FS_ex_ind, 't'].values[0][0]
FS_V = FS_ex_df.loc[FS_ex_ind, 'V'].values[0]
peaks, peak_prop = scipy.signal.find_peaks(
    FS_V,  height=min_spike_height, prominence=prominence)
spike_times_FS = FS_t[peaks]
FS_tstart = spike_times_FS[-5] - plot_dt
FS_tend = spike_times_FS[-5] + plot_dt_a
FS_tplot = (np.array(FS_ex_df.loc[FS_ex_ind, 't'].values[0]) -
            FS_ex_df.loc[FS_ex_ind, 't'].values[0][0] - spike_times_FS[-5]) * 1000
FS_tstart_ind = np.argwhere(FS_t <= FS_tstart)[-1][0]
FS_tend_ind = np.argwhere(FS_t >= FS_tend)[0][0] - 1

#%%

max_F_NS = np.zeros((patch_NS['patch_NS'].shape[0], 1))
for i in range(patch_NS['patch_NS'].shape[0]):
    max_F_NS[i] = np.max(patch_NS['patch_NS'][i, 1])

max_F_NS_df = pd.DataFrame(data=max_F_NS)
# %% Create Figure
np.random.seed(0)  # set seed for stripplot jitter
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choices #####
p_col = ['tab:grey', 'tab:brown', 'tab:olive']
col = ['#D03050', '#6D398B']
order = ['BS', 'NS']
ms = 3
spike_1_diff = np.maximum(spike_times_Pyr[0], spike_times_FS[0])
patch_xlim = (-spike_1_diff*1000, 0.5*1000)
patch_ylim = (-65, 32)
pairs1 = [('BS', 'NS')]

AP_xticks = [0, 200, 400]
AP_xtick_labels = ['1$^{\mathrm{st}}$ AP', '200', '400']

##### layout #####
fig = plt.figure(figsize=(6.5, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1., 0.8], hspace=0.25,  wspace=0.6)
gs_ex = gridspec.GridSpecFromSubplotSpec(
    2, 4,  height_ratios=[0.7, 1.5], wspace=1.1, hspace=0.6, subplot_spec=gs[0])
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    2, 3, wspace=0.6, hspace=0.8, subplot_spec=gs[1])
gs_waveform = gridspec.GridSpecFromSubplotSpec(
    1, 3, wspace=0.7, hspace=0.4, subplot_spec=gs_ex[0, :])
gs_waveform_2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, wspace=0.2, hspace=0.1, subplot_spec=gs_waveform[:2])
ax_BS_waveform = fig.add_subplot(gs_waveform_2[0])
ax_NS_waveform = fig.add_subplot(gs_waveform_2[1])
ax_BS_NS_sep = fig.add_subplot(gs_waveform[2])
gs_BS_VGS = gridspec.GridSpecFromSubplotSpec(
    3, 1, wspace=0.35, hspace=0.25, subplot_spec=gs_ex[1, :2])
ax_BS_VGS_Lab = fig.add_subplot(gs_ex[1, :2])
ax_BS_VGS_Ex = fig.add_subplot(gs_BS_VGS[0])
ax_BS_VGS_P = fig.add_subplot(gs_BS_VGS[1])
ax_BS_VGS_P_n = fig.add_subplot(gs_BS_VGS[2])
gs_NS_VGS = gridspec.GridSpecFromSubplotSpec(
    3, 1, wspace=0.35, hspace=0.25, subplot_spec=gs_ex[1, 2:])
ax_NS_VGS_Lab = fig.add_subplot(gs_ex[1, 2:])
ax_NS_VGS_Ex = fig.add_subplot(gs_NS_VGS[0])
ax_NS_VGS_P = fig.add_subplot(gs_NS_VGS[1])
ax_NS_VGS_P_n = fig.add_subplot(gs_NS_VGS[2])
ax_q_coeff = fig.add_subplot(gs_bottom[0, 0])
ax_q_off = fig.add_subplot(gs_bottom[0, 1])
ax_q_A_C = fig.add_subplot(gs_bottom[0, 2])
ax_q_AI = fig.add_subplot(gs_bottom[1, 0])
ax_q_tau = fig.add_subplot(gs_bottom[1, 1])
ax_q_laten = fig.add_subplot(gs_bottom[1, 2])

##### spike width vs frequency #####
ax_BS_NS_sep.plot(df_IC.loc[:, 'widths']*1000, df_IC.loc[:,
                  'instfr'], 'o', markersize=ms,  c='tab:grey')
ax_BS_NS_sep.vlines(700, ax_BS_NS_sep.get_ylim()[0], ax_BS_NS_sep.get_ylim()[
                    1], linestyle='--', color='k', label='   BS/NS\nThreshold', zorder=0)
ax_BS_NS_sep.set_ylim(0, 180)
ax_BS_NS_sep.set_xlabel('Width (ms)')
ax_BS_NS_sep.set_ylabel('median ISI F (Hz)')
leg = ax_BS_NS_sep.legend(ncol=2, frameon=False, loc=1, bbox_to_anchor=(
    1.1, 1.3), fontsize=8, handletextpad=0.25, labelspacing=0.2)
leg.get_lines()[0].set_linewidth(1.25)


##### Patch clamp examples #####
ax_BS_waveform.plot(Pyr_tplot[Pyr_tstart_ind:Pyr_tend_ind], Pyr_ex_df.loc[Pyr_ex_ind,
                    'V'].values[0][Pyr_tstart_ind:Pyr_tend_ind], color=col[0])
ax_NS_waveform.plot(FS_tplot[FS_tstart_ind:FS_tend_ind], FS_ex_df.loc[FS_ex_ind,
                    'V'].values[0][FS_tstart_ind:FS_tend_ind], color=col[1])
ax_BS_waveform.set_xlabel('Time (ms)')
ax_BS_waveform.set_ylabel("V (mV)")
ax_NS_waveform.set_xlabel('Time (ms)')


# BS
sat = np.linspace(0.75, 1.5, patch_BS['patch_BS'].shape[0])
for i in range(patch_BS['patch_BS'].shape[0]):
    ax_BS_VGS_P.plot(patch_BS['patch_BS'][i, 0] * 1000, patch_BS['patch_BS']
                     [i, 1], color=lighten_color(col[0], sat[i]), alpha=0.5)
ax_BS_VGS_P.set_ylabel("Freq. (Hz)")

sat = np.linspace(0.75, 1.5, patch_BS['patch_BS'].shape[0])
for i in range(patch_BS['patch_BS'].shape[0]):
    ax_BS_VGS_P_n.plot(patch_BS['patch_BS'][i, 0] * 1000, patch_BS['patch_BS'][i, 1] / np.max(
        patch_BS['patch_BS'][i, 1]), color=lighten_color(col[0], sat[i]), alpha=0.5)
ax_BS_VGS_P_n.set_ylabel("Norm\nFreq.")


ax_BS_VGS_Ex.plot((np.array(Pyr_ex_df.loc[Pyr_ex_ind, 't'].values[0]) - Pyr_ex_df.loc[Pyr_ex_ind, 't'].values[0][0] - spike_times_Pyr[0]) * 1000,
                  Pyr_ex_df.loc[Pyr_ex_ind, 'V'].values[0], color=col[0])
ax_BS_VGS_Ex.set_ylabel("V (mV)")

# NS
sat = np.linspace(0.95, 1.25, patch_NS['patch_NS'].shape[0])
for i in range(patch_NS['patch_NS'].shape[0]):
    ax_NS_VGS_P.plot(patch_NS['patch_NS'][i, 0] * 1000, patch_NS['patch_NS']
                     [i, 1], color=lighten_color(col[1], sat[i]), alpha=0.5)
ax_NS_VGS_P.set_ylabel("Freq. (Hz)")

sat = np.linspace(0.95, 1.25, patch_NS['patch_NS'].shape[0])
for i in range(patch_NS['patch_NS'].shape[0]):
    ax_NS_VGS_P_n.plot(patch_NS['patch_NS'][i, 0] * 1000, patch_NS['patch_NS'][i, 1] / np.max(
        patch_NS['patch_NS'][i, 1]), color=lighten_color(col[1], sat[i]), alpha=0.5)
ax_NS_VGS_P_n.set_ylabel("Norm\nFreq.")

ax_NS_VGS_Ex.plot((np.array(FS_ex_df.loc[FS_ex_ind, 't'].values[0]) - FS_ex_df.loc[FS_ex_ind, 't'].values[0][0] - spike_times_FS[0]) * 1000,
                  FS_ex_df.loc[FS_ex_ind, 'V'].values[0], color=col[1])

##### coefficient #####
sns.boxplot(data=df_coeff, x='condition', y='Coefficient', ax=ax_q_coeff,
            palette=col, boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_coeff, x='condition', y='Coefficient',
              ax=ax_q_coeff, palette=col, s=ms)
ax_q_coeff.set_ylabel('Coefficient (Hz)')
annotator = Annotator(ax_q_coeff, pairs1, data=df_coeff,
                      x='condition',  y='Coefficient')
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_q_coeff, test_results1 = annotator.annotate()
ax_q_coeff.set_yticks([0, 50, 100, 150])

##### offset #####
sns.boxplot(data=df_off, x='condition', y='Offset', ax=ax_q_off,
            palette=col, boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_off, x='condition', y='Offset',
              ax=ax_q_off, palette=col, s=ms)
ax_q_off.set_ylabel('Baseline (Hz)')
annotator = Annotator(ax_q_off, pairs1, data=df_off,
                      x='condition',  y='Offset')
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_q_off, test_results1 = annotator.annotate()
ax_q_off.set_yticks([0, 50, 100, 150])


##### latency #####
sns.boxplot(data=df_laten, x='Type', y='Initial Spike t', ax=ax_q_laten, palette=col,
            boxprops=dict(alpha=.5), saturation=0.5, order=order, showfliers=False)
sns.stripplot(data=df_laten, x='Type', y='Initial Spike t',
              ax=ax_q_laten, palette=col, order=order, s=ms)
ax_q_laten.set_ylabel('First Spike\nLatency (ms)')
annotator = Annotator(ax_q_laten, pairs1, data=df_laten,
                      x='Type', y='Initial Spike t', )
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_q_laten, test_results1 = annotator.annotate()
ax_q_laten.set_yticks([0, 25, 50, 75])

##### AI #####
sns.boxplot(data=df_AI, x='condition', y='Adaptation Index', ax=ax_q_AI,
            palette=col, boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_AI, x='condition', y='Adaptation Index',
              ax=ax_q_AI, palette=col, s=ms)
annotator = Annotator(ax_q_AI, pairs1, data=df_AI,
                      x='condition',  y='Adaptation Index')
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_q_AI, test_results1 = annotator.annotate()
ax_q_AI.set_ylim(0, ax_q_AI.get_ylim()[1])
ax_q_AI.set_yticks([0, 1, 2, 3, 4])

#####  tau #####
sns.boxplot(data=df_tau, x='condition', y='decay $\tau$', ax=ax_q_tau,
            palette=col, boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_tau, x='condition', y='decay $\tau$',
              ax=ax_q_tau, palette=col, s=ms)
ymin, ymax = ax_q_tau.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_q_tau.yaxis.set_ticks(tick_range)
tick_range_ticks = np.append([0.5], tick_range[:-1])
ax_q_tau.yaxis.set_ticks([np.log10(x) for p in [0,1,2,3,4] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_q_tau.yaxis.set_ticklabels(['$10^{%s}$' % (round(i)) for i in tick_range])
ax_q_tau.set_ylabel('I-SFA $\\tau$ (ms)')
annotator = Annotator(ax_q_tau, pairs1, data=df_tau,
                      x='condition',  y='decay $\tau$')
annotator.configure(test='Mann-Whitney', loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_q_tau, test_results1 = annotator.annotate()
ax_q_tau.set_ylim(ymin, ymax)
#####  relationship between baseline and coefficient #####
ax_q_A_C.plot(df['Patch BS Offset'], df['Patch BS Coefficient'],
              '.', color=col[0], label="BS")
ax_q_A_C.plot(df['Patch NS Offset'], df['Patch NS Coefficient'],
              '.', color=col[1], label="NS")
ax_q_A_C.set_ylabel('Coefficient (Hz)')
ax_q_A_C.set_xlabel("Baseline (Hz)")
ax_q_A_C.set_xticks([0, 50, 100, 150])
ax_q_A_C.set_yticks([0, 50, 100, 150])

rho_BS, p_BS = stats.spearmanr(
    df['Patch BS Offset'], df['Patch BS Coefficient'], nan_policy='omit')
rho_NS, p_NS = stats.spearmanr(
    df['Patch NS Offset'], df['Patch NS Coefficient'], nan_policy='omit')
print("BS rho = {}, p = {}".format(rho_BS, p_BS))
print("NS rho = {}, p = {}".format(rho_NS, p_NS))

##### remove axis boxes, axis limits, label adding #####
for ax in [ax_BS_VGS_Lab, ax_NS_VGS_Lab]:
    ax = remove_axis_box(ax, ["right", "top", "left", "bottom"])

for ax in [ax_q_off, ax_q_coeff, ax_q_laten, ax_q_AI, ax_q_tau]:
    ax.set_xlabel("")

for ax in [ax_q_off, ax_q_coeff, ax_q_laten, ax_q_AI, ax_q_tau, ax_BS_NS_sep, ax_q_A_C]:
    ax = remove_axis_box(ax, ["right", "top"])

for ax in [ax_BS_waveform, ax_NS_waveform]:
    ax.set_xlim(-plot_dt*1000, plot_dt_a*1000)
    ax.set_ylim(patch_ylim)
    ax = remove_axis_box(ax, ["right", "top"])
ax_NS_waveform = remove_axis_box(ax_NS_waveform, ["left"])
ax_BS_waveform.set_title("BS")
ax_NS_waveform.set_title("NS")


for ax in [ax_BS_VGS_Ex, ax_BS_VGS_P,  ax_BS_VGS_P_n, ax_NS_VGS_Ex, ax_NS_VGS_P, ax_NS_VGS_P_n]:
    ax.set_xlim(patch_xlim)
    ax = remove_axis_box(ax, ["right", "top"])

for ax in [ax_BS_VGS_P, ax_NS_VGS_P]:
    ax = remove_axis_box(ax, ["bottom"])

ax_BS_VGS_P_n.set_xlabel('Time (ms)')
ax_BS_VGS_P_n.set_xticks(AP_xticks)
ax_BS_VGS_P_n.set_xticklabels(AP_xtick_labels)
ax_BS_VGS_Ex = remove_axis_box(ax_BS_VGS_Ex, ["bottom"])
ax_BS_VGS_Ex.set_title("BS")
ax_NS_VGS_Ex = remove_axis_box(ax_NS_VGS_Ex, ["bottom"])
ax_NS_VGS_Ex.set_title("NS")
ax_NS_VGS_Ex.set_ylabel("V (mV)")
ax_NS_VGS_P_n.set_xlabel('Time (ms)')
ax_NS_VGS_P_n.set_xticks(AP_xticks)
ax_NS_VGS_P_n.set_xticklabels(AP_xtick_labels)

ax_NS_VGS_P_n.set_ylim(0, 1.1)
ax_BS_VGS_P_n.set_ylim(0, 1.1)

##### subplot letters #####
ax_BS_waveform.text(-0.025, 1.075, string.ascii_uppercase[0],
                    transform=ax_BS_waveform.transAxes, size=12, weight='bold')
ax_BS_NS_sep.text(-0.025, 1.075, string.ascii_uppercase[1],
                  transform=ax_BS_NS_sep.transAxes, size=12, weight='bold')
ax_BS_VGS_Ex.text(-0.025, 1.075, string.ascii_uppercase[2],
                  transform=ax_BS_VGS_Ex.transAxes, size=12, weight='bold')
ax_NS_VGS_Ex.text(-0.025, 1.075, string.ascii_uppercase[3],
                  transform=ax_NS_VGS_Ex.transAxes, size=12, weight='bold')
ax_q_coeff.text(-0.025, 1.075, string.ascii_uppercase[4],
                transform=ax_q_coeff.transAxes, size=12, weight='bold')
ax_q_off.text(-0.025, 1.075, string.ascii_uppercase[5],
              transform=ax_q_off.transAxes, size=12, weight='bold')
ax_q_A_C.text(-0.025, 1.075, string.ascii_uppercase[6],
              transform=ax_q_A_C.transAxes, size=12, weight='bold')
ax_q_AI.text(-0.025, 1.075, string.ascii_uppercase[7],
             transform=ax_q_AI.transAxes, size=12, weight='bold')
ax_q_tau.text(-0.025, 1.075, string.ascii_uppercase[8],
              transform=ax_q_tau.transAxes, size=12, weight='bold')
ax_q_laten.text(-0.025, 1.075, string.ascii_uppercase[9],
                transform=ax_q_laten.transAxes, size=12, weight='bold')

##### save and show #####
plt.savefig('./figures/Figure_4_in_vitro_adaptation.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Figure_4_in_vitro_adaptation.pdf',
            bbox_inches='tight', dpi=300)
plt.show()
