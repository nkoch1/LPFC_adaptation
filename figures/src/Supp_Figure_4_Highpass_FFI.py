#%%
import os
os.chdir('../../')
import string
from ast import literal_eval
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from statannotations.Annotator import Annotator
import seaborn as sns
import scikit_posthocs
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import lighten_color, remove_axis_box, circuit_dia_HP
plt.rcParams["font.family"] = "Arial"
# %% load data

I = 0.74
g = 5.6
gsynI = 2.45
VGS_I_psth = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_psth_0_1.csv".format(I, g, gsynI))
VGS_I_psth_t = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_psth_t_3.csv".format(I, g, gsynI))

VGS_I_psth_array = np.array([np.array(literal_eval(
    VGS_I_psth.iloc[i, 0])) for i in range(VGS_I_psth.shape[0])])
VGS_I_psth_t_array = np.array([np.array(literal_eval(
    VGS_I_psth_t.iloc[i, 0])) for i in range(VGS_I_psth_t.shape[0])])

VGS_sum = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_Inh_sim_I_{}_g_{}_gi_{}_10.csv".format(I, g, gsynI))
VGS_sum.rename(columns={"τ_step": "Step", "τ_VGS": "E",
               "τ_VGS_I": "E+I"}, inplace=True)

VGS_sum_AI = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_Inh_sim_I_{}_g_{}_gi_{}_10_AI.csv".format(I, g, gsynI))
VGS_sum_AI.rename(columns={"AI_step": "Step",
                  "AI_VGS": "E", "AI_VGS_I": "E+I"}, inplace=True)

tau_VGS_I = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_tau.csv".format(I, g, gsynI), header=None)
coeff_VGS_I = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_coeff.csv".format(I, g, gsynI), header=None)
offset_VGS_I = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_offset.csv".format(I, g, gsynI), header=None)
AI_VGS_I = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_AI.csv".format(I, g, gsynI), header=None)
highpass_cut = pd.read_csv(
    "./data/sims/Pospischil_highpass_filter_Inh_sim_I_{}_g_{}_gi_{}_10_highpass_cut.csv".format(I, g, gsynI), header=None)


order = [-1, 0, 1, 2, 3, 4, 5]
tau_VGS_I[-1] = 10**VGS_sum['E+I']
tau_VGS_I = tau_VGS_I[order]

order = [-1, 0, 1, 2, 3, 4, 5]
AI_VGS_I[-1] = VGS_sum_AI['E+I']
AI_VGS_I = AI_VGS_I[order]


highpass_cut.loc[-1, 0] = 'Lowpass\nonly'
highpass_cut.sort_index(inplace=True)


plot_col_i = [0, 1, 2, 4, 6]
tau_VGS_I = tau_VGS_I.iloc[:, plot_col_i]
AI_VGS_I = AI_VGS_I.iloc[:, plot_col_i]

plot_col_i_all = [-1, 0, 1, 3, 5]
highpass_cut = highpass_cut.loc[plot_col_i_all, :]


df_0_1 = pd.DataFrame(columns=['decay $\tau$ (ms)', 'condition'])
df_0_1['decay $\tau$'] = np.log10(tau_VGS_I[0])
df_0_1['condition'] = "0.1"

df_3 = pd.DataFrame(columns=['Decay $\tau$ (ms)', 'condition'])
df_3['decay $\tau$'] = np.log10(tau_VGS_I[3])
df_3['condition'] = "3"

df_5 = pd.DataFrame(columns=['Decay $\tau$ (ms)', 'condition'])
df_5['decay $\tau$'] = np.log10(tau_VGS_I[5])
df_5['condition'] = "5"

df_vgs = pd.read_csv('./data/exp/Summary_Decay_fit_250_psth_10.csv')
df_vgs_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_vgs_n['decay $\tau$'] = np.log10(
    -1 / (df_vgs.loc[df_vgs.index[~df_vgs["VGS NS"].isnull()], "VGS NS"] / 1000))
df_vgs_n['condition'] = "NS"

df_tau_pop = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_decay_PSTH_n_70_fit_1000_psth_50.csv')
df_pop_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_pop_n['decay $\tau$'] = np.log10(-1 / (df_tau_pop['VGS NS'] / 1000))
df_pop_n['condition'] = "Pop NS"

df_sum_data = pd.concat(
    [df_0_1, df_3, df_5, df_vgs_n, df_pop_n], ignore_index=True)

s, p = stats.kruskal(df_0_1["decay $\tau$"], df_3['decay $\tau$'], df_5['decay $\tau$'],
                     df_vgs_n["decay $\tau$"], df_pop_n["decay $\tau$"], nan_policy="omit")
posthoc_data = scikit_posthocs.posthoc_dunn(
    df_sum_data, val_col='decay $\tau$', group_col='condition')


s, p = stats.kruskal(VGS_sum["Step"], df_0_1["decay $\tau$"],
                     df_3["decay $\tau$"], df_5['decay $\tau$'], nan_policy="omit")

df_Step = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_Step['decay $\tau$'] = VGS_sum['Step']
df_Step['condition'] = "Step"

df_sum_mod = pd.concat([df_Step, df_0_1, df_3, df_5], ignore_index=True)
posthoc_mod = scikit_posthocs.posthoc_dunn(
    df_sum_mod, val_col='decay $\tau$', group_col='condition')

# AI #################################

df_0_1_AI = pd.DataFrame(columns=['AI', 'condition'])
df_0_1_AI['AI'] = AI_VGS_I[0]
df_0_1_AI['condition'] = "0.1"

df_3_AI = pd.DataFrame(columns=['AI', 'condition'])
df_3_AI['AI'] = AI_VGS_I[3]
df_3_AI['condition'] = "3"

df_5_AI = pd.DataFrame(columns=['AI', 'condition'])
df_5_AI['AI'] = AI_VGS_I[5]
df_5_AI['condition'] = "5"

df_offset_vivo = pd.read_csv(
    './data/exp/extracell_offset_only_fit_250_psth_10.csv')
df_coeff_vivo = pd.read_csv(
    './data/exp/extracell_coefficient_only_fit_250_psth_10.csv')
df_AI_vivo = df_offset_vivo / (df_offset_vivo + df_coeff_vivo)
df_vgs_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_vgs_n_AI['AI'] = df_AI_vivo["VGS NS"]
df_vgs_n_AI['condition'] = "NS"

df_pop_offset = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_offset_PSTH_n_70_fit_1000_psth_50.csv')
df_pop_coeff = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_coeff_PSTH_n_70_fit_1000_psth_50.csv')
df_pop_AI = df_pop_offset / (df_pop_offset + df_pop_coeff)
df_pop_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_pop_n_AI['AI'] = df_pop_AI['VGS NS']
df_pop_n_AI['condition'] = "Pop NS"

df_sum_data_AI = pd.concat(
    [df_0_1_AI, df_3_AI, df_5_AI, df_vgs_n_AI, df_pop_n_AI], ignore_index=True)

s, p = stats.kruskal(df_0_1_AI["AI"], df_3_AI['AI'], df_5_AI['AI'],
                     df_vgs_n_AI["AI"], df_pop_n_AI["AI"], nan_policy="omit")
posthoc_data_AI = scikit_posthocs.posthoc_dunn(
    df_sum_data_AI, val_col='AI', group_col='condition')


s, p = stats.kruskal(VGS_sum_AI["Step"], df_0_1_AI["AI"],
                     df_3_AI['AI'], df_5_AI['AI'], nan_policy="omit")
df_Step_AI = pd.DataFrame(columns=['AI', 'condition'])
df_Step_AI['AI'] = VGS_sum_AI['Step']
df_Step_AI['condition'] = "Step"

df_sum_mod_AI = pd.concat(
    [df_Step_AI, df_0_1_AI, df_3_AI, df_5_AI], ignore_index=True)
posthoc_mod_AI = scikit_posthocs.posthoc_dunn(
    df_sum_mod_AI, val_col='AI', group_col='condition')


# %% Create Figure
np.random.seed(0)  # set seed for stripplot jitter
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choices #####
c_step = lighten_color('#6D398B', amount=1.25)
c_E = 'tab:grey'
c_EI = '#C02717'
alpha = 0.5
lw = 2
stim_start = 750.
ms = 4
psth_xlim = (-250, 1000)

##### Layout #####
fig = plt.figure(figsize=(6.5, 6))
gs = fig.add_gridspec(3, 2, height_ratios=[
                      0.75, 0.75, 1], hspace=0.75,  wspace=0.35)
gs_l = gridspec.GridSpecFromSubplotSpec(
    1, 2, width_ratios=[0.75, 1.5], wspace=0.6, hspace=0.5, subplot_spec=gs[0, :])
ax_dia2 = fig.add_subplot(gs_l[0])
ax_VGS_BS_I = fig.add_subplot(gs_l[1])
ax_VGS_sum = fig.add_subplot(gs[1, 0])
ax_VGS_AI = fig.add_subplot(gs[2, 0])
ax_data_comp = fig.add_subplot(gs[1, 1])
ax_data_AI_comp = fig.add_subplot(gs[2, 1])

##### circuit diagram #####
ax_dia2 = circuit_dia_HP(ax_dia2, inhib=True, lw_c=4, lw_con=3,  c_E='#7DA54B',
                         c_M=c_EI, c_I="tab:brown", lw_sdf=0.5, lw_f=2, fs_labels=10, fs_eqn=12, fs_axis=8)

##### PSTH #####
sat = np.linspace(0.5, 1.25, VGS_I_psth_array.shape[0])
for i in range(VGS_I_psth_array.shape[0]):
    ax_VGS_BS_I.plot(VGS_I_psth_t_array[i], VGS_I_psth_array[i],
                     alpha=alpha, color=lighten_color(c_EI, sat[i]))
ax_VGS_BS_I.set_ylim(0, ax_VGS_BS_I.get_ylim()[1])
ax_VGS_BS_I.set_xticks([0, 500, 1000])
ax_VGS_BS_I.set_xticklabels(
    ['Stim. Onset',  '500',  '1000'])

##### decay tau boxplots #####
cond = VGS_sum.columns.tolist()
order_e = ['Step', "0.1", "3", "5"]
sns.boxplot(data=df_sum_mod.pivot(columns='condition', values='decay $\tau$'), ax=ax_VGS_sum, saturation=0.5,
            order=order_e, palette=[c_step, c_EI, c_EI, c_EI],  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_mod.pivot(columns='condition', values='decay $\tau$'),
              ax=ax_VGS_sum, palette=[c_step, c_EI, c_EI, c_EI],  order=order_e, s=ms)
ax_VGS_sum.set_xticks(ax_VGS_sum.get_xticks())
ax_VGS_sum.set_xticklabels(
    ax_VGS_sum.get_xticklabels(), rotation=0, ha='center')
ax_VGS_sum.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
ax_VGS_sum.set_ylabel("Decay $\\tau$ (ms)")
ax_VGS_sum.set_xlabel("Highpass Cutoff (Hz)", ha='center', x=0.625, labelpad=3)
ymin, ymax = ax_VGS_sum.get_ylim()
ax_VGS_sum.yaxis.set_ticks([1,2,3,4,5])
ax_VGS_sum.yaxis.set_ticks([np.log10(x) for p in [1,2,3,4,5] for x in np.linspace(
    10 ** (p -1), 10 ** (p ), 10)], minor=True)
ax_VGS_sum.set_ylim(ymin, ymax)
pairs1 = [(order_e[0], order_e[1]),
          (order_e[0], order_e[2]),
          (order_e[0], order_e[3]),
          ]
pvalues = []
for pair in pairs1:
    p = posthoc_mod.loc[pair[0], pair[1]]
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_VGS_sum, pairs1, data=df_sum_mod.pivot(
    columns='condition', values='decay $\tau$')[order_e])
annotator.configure(test=None, test_short_name="", loc='inside', fontsize=8,
                    show_test_name=False, pvalue_thresholds=pval_thresh,  hide_non_significant=True)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()
ax_VGS_sum.set_xticklabels(['I-SFA', '0.1', '3', '5'], rotation=0, ha='center')


##### adaptation index boxplots #####
sns.boxplot(data=df_sum_mod_AI.pivot(columns='condition', values='AI'), ax=ax_VGS_AI, saturation=0.5,
            order=order_e, palette=[c_step, c_EI, c_EI, c_EI],  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_mod_AI.pivot(columns='condition', values='AI'),
              ax=ax_VGS_AI, palette=[c_step, c_EI, c_EI, c_EI],  order=order_e, s=ms)
ax_VGS_AI.set_ylabel("Adaptation Index")
ax_VGS_AI.set_xlabel("Highpass Cutoff (Hz)", ha='center', x=0.625, labelpad=3)
pairs1 = [(order_e[0], order_e[1]),
          (order_e[0], order_e[2]),
          (order_e[0], order_e[3]),
          ]
pvalues_AI = []
for pair in pairs1:
    p = posthoc_mod_AI.loc[pair[0], pair[1]]
    pvalues_AI.append(p)
print("pvalues:", pvalues_AI)
annotator = Annotator(ax_VGS_AI, pairs1, data=df_sum_mod_AI.pivot(
    columns='condition', values='AI')[order_e])
annotator.configure(test=None, test_short_name="", loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh,  hide_non_significant=True)
annotator.set_pvalues(pvalues=pvalues_AI)
annotator.annotate()
ax_VGS_AI.set_ylim(0, ax_VGS_AI.get_ylim()[1])
ax_VGS_AI.set_xticklabels(['I-SFA', '0.1', '3', '5'], rotation=0, ha='center')

##### Tau vs Data #####
order_NS = ["NS", "Pop NS",  "0.1", "3", "5"]
palette_NS = ['#2060A7', "#007030", c_EI, c_EI, c_EI]
sns.boxplot(data=df_sum_data.pivot(columns='condition', values='decay $\tau$'), ax=ax_data_comp,
            saturation=0.5, order=order_NS, palette=palette_NS,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_data.pivot(columns='condition', values='decay $\tau$'),
              ax=ax_data_comp, palette=palette_NS,  order=order_NS, s=ms)
ax_data_comp.set_ylabel("E-SFA $\\tau$ (ms)")
ax_data_comp.set_xlabel("")
ax_data_comp.set_xticks(ax_data_comp.get_xticks())
ax_data_comp.set_xticklabels(
    ax_data_comp.get_xticklabels(), rotation=0, ha='center')
ax_data_comp.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
ymin, ymax = ax_data_comp.get_ylim()
# tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_data_comp.yaxis.set_ticks([1,2,3,4])
# tick_range_ticks = np.append([0.5], tick_range[:-1])
ax_data_comp.yaxis.set_ticks([np.log10(x) for p in [1,2,3,4] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_data_comp.set_ylim(ymin, ymax)
ax_data_comp.set_xlabel("Highpass Cutoff (Hz)",
                        ha='center', x=0.71, labelpad=3)
pairs1 = [(order_NS[0], order_NS[2]),
          (order_NS[1], order_NS[2]),
          (order_NS[0], order_NS[3]),
          (order_NS[1], order_NS[3]),
          (order_NS[0], order_NS[4]),
          (order_NS[1], order_NS[4]),
          ]
pvalues = []
for pair in pairs1:
    p = posthoc_data.loc[pair[0], pair[1]]
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_data_comp, pairs1, data=df_sum_data.pivot(
    columns='condition', values='decay $\tau$')[order_NS])
annotator.configure(test=None, test_short_name="", loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh,  hide_non_significant=True)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()

##### AI vs data #####
sns.boxplot(data=df_sum_data_AI.pivot(columns='condition', values='AI'), ax=ax_data_AI_comp,
            saturation=0.5, order=order_NS, palette=palette_NS,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_data_AI.pivot(columns='condition', values='AI'),
              ax=ax_data_AI_comp, palette=palette_NS,  order=order_NS, s=ms)
ax_data_AI_comp.set_ylabel("Adaptation Index")
ax_data_AI_comp.set_xlabel("Highpass Cutoff (Hz)",
                           ha='center', x=0.71, labelpad=3)
pairs1 = [(order_NS[0], order_NS[2]),
          (order_NS[1], order_NS[2]),
          (order_NS[0], order_NS[3]),
          (order_NS[1], order_NS[3]),
          (order_NS[0], order_NS[4]),
          (order_NS[1], order_NS[4]),
          ]
pvalues = []
for pair in pairs1:
    p = posthoc_data_AI.loc[pair[0], pair[1]]
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_data_AI_comp, pairs1, data=df_sum_data_AI.pivot(
    columns='condition', values='AI')[order_NS])
annotator.configure(test=None, test_short_name="", loc='inside', fontsize=10,
                    show_test_name=False, pvalue_thresholds=pval_thresh,  hide_non_significant=True)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()


##### remove axis box sides #####
for ax in [ax_VGS_BS_I]:
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Avg Firing (Hz)")
    ax.set_xlim(psth_xlim)

for ax in [ax_dia2,  ax_VGS_BS_I, ax_VGS_sum, ax_VGS_AI,  ax_data_comp, ax_data_AI_comp]:
    ax = remove_axis_box(ax, ["right", "top"])

##### subplot letters #####
ax_dia2.text(-0.05, 1.1, string.ascii_uppercase[0],
             transform=ax_dia2.transAxes, size=12, weight='bold')
ax_VGS_sum.text(-0.025, 1.075, string.ascii_uppercase[1],
                transform=ax_VGS_sum.transAxes, size=12, weight='bold')
ax_VGS_AI.text(-0.025, 1.075, string.ascii_uppercase[3],
               transform=ax_VGS_AI.transAxes, size=12, weight='bold')
ax_data_comp.text(-0.025, 1.075, string.ascii_uppercase[2],
                  transform=ax_data_comp.transAxes, size=12, weight='bold')
ax_data_AI_comp.text(-0.025, 1.075, string.ascii_uppercase[4],
                     transform=ax_data_AI_comp.transAxes, size=12, weight='bold')

##### save and show #####
plt.savefig('./figures/Supp_Figure_4_Highpass_FFI.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Supp_Figure_4_Highpass_FFI.pdf',
            bbox_inches='tight', dpi=300)
plt.show()
