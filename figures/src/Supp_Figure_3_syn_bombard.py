#%%
import os
os.chdir('../../')
import matplotlib.patches as patches
import string
from ast import literal_eval
from matplotlib import ticker
import scipy.io as sio
import matplotlib.gridspec as gridspecLatency
from statannotations.Annotator import Annotator
import seaborn as sns
import scikit_posthocs
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import lighten_color, circuit_dia_OU
plt.rcParams["font.family"] = "Arial"

# %% load data
step_tau = pd.read_csv("./data/sims/Pospischil_step_sim_tau.csv", header=None)
step_F = pd.read_csv("./data/sims/Pospischil_step_sim_F.csv")
step_t = pd.read_csv("./data/sims/Pospischil_step_sim_t.csv")

VGS_tau = pd.read_csv(
    "./data/sims/Pospischil_filter_OU_sim_tau.csv", header=None)
VGS_psth = pd.read_csv(
    "./data/sims/Pospischil_filter_OU_sim_psth.csv")
VGS_psth_t = pd.read_csv(
    "./data/sims/Pospischil_filter_OU_sim_psth_t.csv")
VGS_psth_array = np.array(
    [np.array(literal_eval(VGS_psth.iloc[i, 0])) for i in range(VGS_psth.shape[0])])
VGS_psth_t_array = np.array([np.array(literal_eval(
    VGS_psth_t.iloc[i, 0])) for i in range(VGS_psth_t.shape[0])])


VGS_sum = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_OU_sim.csv")
VGS_sum.rename(columns={"τ_step": "Step", "τ_VGS": "BS"}, inplace=True)

VGS_sum_AI = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_OU_sim_AI.csv")
VGS_sum_AI.rename(columns={"AI_step": "Step", "AI_VGS": "BS"}, inplace=True)

t = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_T.csv', header=None)
VGS_narrow = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_VGS_NS.csv', header=None)


df_Step = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_Step['decay $\tau$'] = VGS_sum['Step']
df_Step['condition'] = "Step"

df_BS = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_BS['decay $\tau$'] = VGS_sum['BS']
df_BS['condition'] = "BS"


df_vgs = pd.read_csv('./data/exp/Summary_Decay.csv')
df_vgs_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_vgs_n['decay $\tau$'] = np.log10(
    -1 / (df_vgs.loc[df_vgs.index[~df_vgs["VGS NS"].isnull()], "VGS NS"] / 1000))
df_vgs_n['condition'] = "NS"

df_tau_pop = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_decay.csv')
df_pop_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_pop_n['decay $\tau$'] = np.log10(-1 / (df_tau_pop['VGS NS'] / 1000))
df_pop_n['condition'] = "Pop NS"

df_sum_data = pd.concat([df_BS,  df_vgs_n, df_pop_n], ignore_index=True)

s, p = stats.kruskal(df_BS["decay $\tau$"], df_vgs_n["decay $\tau$"],
                     df_pop_n["decay $\tau$"], nan_policy="omit")
print(s)
print(p)
posthoc_data = scikit_posthocs.posthoc_dunn(
    df_sum_data, val_col='decay $\tau$', group_col='condition')


# %% Create Figure
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choices #####
c_step = lighten_color('#6D398B', amount=1.25)
c_E = 'tab:grey'
c_EI = '#C02717'
alpha = 0.5
lw = 2
stim_start = 750.
psth_xlim = (-250, 1000)

##### Layout #####
fig = plt.figure(figsize=(6.5, 5.25))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.75], height_ratios=[
                      0.8, 1], hspace=0.6,  wspace=0.5)
ax_dia = fig.add_subplot(gs[0, 0])
ax_VGS_BS = fig.add_subplot(gs[0, 1])
ax_tau_comp = fig.add_subplot(gs[1, 0])
ax_data_comp = fig.add_subplot(gs[1, 1])

ax_dia = circuit_dia_OU(ax_dia, lw_c=4, lw_con=3,  c_E="#7DA54B", c_M="tab:grey",
                        c_gE="peru", c_gI="maroon", lw_sdf=0.5, lw_f=2, fs_labels=12, fs_eqn=12, fs_axis=8)


##### PSTH #####
sat = np.linspace(0.5, 1.5, VGS_psth_array.shape[0])
for i in range(VGS_psth_array.shape[0]):
    ax_VGS_BS.plot(VGS_psth_t_array[i], VGS_psth_array[i],
                   alpha=alpha, color=lighten_color(c_E, sat[i]))

ax_VGS_BS.set_xlabel("Time (ms)")
ax_VGS_BS.set_ylabel("Avg Firing (Hz)")
ax_VGS_BS.set_xlim(psth_xlim)
ax_VGS_BS.set_xticks([0, 500, 1000])
ax_VGS_BS.set_xticklabels(['Stim. Onset',  '500', '1000'])

##### decay tau boxplots #####
cond = VGS_sum.columns.tolist()
order_e = cond
sns.boxplot(data=VGS_sum[['Step', 'BS']], ax=ax_tau_comp, saturation=0.5, order=order_e[0:2], palette=[
            c_step, c_E],  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=VGS_sum[['Step', 'BS']], ax=ax_tau_comp, palette=[
              c_step, c_E],  order=order_e[0:2])
ax_tau_comp.set_xticks(ax_tau_comp.get_xticks())
ax_tau_comp.set_xticklabels(['I-SFA', 'Exc.+Syn.'], rotation=0, ha='center')
ax_tau_comp.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
ymin, ymax = ax_tau_comp.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_tau_comp.yaxis.set_ticks(tick_range)
tick_range_ticks = np.append([0.], tick_range[:-1])
ax_tau_comp.yaxis.set_ticks([np.log10(x) for p in tick_range_ticks for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_tau_comp.set_ylabel("Decay $\\tau$ (ms)")
pairs1 = [(cond[0], cond[1]),]
pvalues = []
for pair in pairs1:
    U, p = stats.mannwhitneyu(VGS_sum['Step'], VGS_sum['BS'])
    print("U={} \n p = {}".format(U, p))
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_tau_comp, pairs1, data=VGS_sum[['Step', 'BS']])
annotator.configure(test=None, test_short_name="", pvalue_thresholds=pval_thresh,
                    loc='inside', fontsize=12, show_test_name=False)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()
ax_tau_comp.set_ylim(ymin, ymax)

##### Tau vs data #####
order_NS = ["NS", "Pop NS", "BS"]
palette_NS = ['#2060A7', "#007030", c_E]
sns.boxplot(data=df_sum_data.pivot(columns='condition', values='decay $\tau$')[
            order_NS], ax=ax_data_comp, saturation=0.5, order=order_NS, palette=palette_NS,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_data.pivot(columns='condition', values='decay $\tau$')[
              order_NS], ax=ax_data_comp, palette=palette_NS,  order=order_NS)
ax_data_comp.set_ylabel("E-SFA $\\tau$ (ms)")
ax_data_comp.set_xlabel(r"")
ax_data_comp.set_xticks(ax_data_comp.get_xticks())
ax_data_comp.set_xticklabels(
    ['NS', 'Pop NS', 'Exc.+Syn.'], rotation=0, ha='center')
ax_data_comp.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
ymin, ymax = ax_data_comp.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_data_comp.yaxis.set_ticks(tick_range)
tick_range_ticks = np.append([0.], tick_range[:-1])
ax_data_comp.yaxis.set_ticks([np.log10(x) for p in [1,2,3,4,] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
pairs1 = [(order_NS[0], order_NS[2]), (order_NS[1], order_NS[2]),]
pvalues = []
for pair in pairs1:
    p = posthoc_data.loc[pair[0], pair[1]]
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_data_comp, pairs1, data=df_sum_data.pivot(
    columns='condition', values='decay $\tau$')[order_NS])
annotator.configure(test=None, test_short_name="", pvalue_thresholds=pval_thresh,
                    loc='inside', fontsize=12, show_test_name=False)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()
ax_data_comp.set_ylim(ymin, 4.1)


##### remove axis box sides #####
for ax in [ax_dia, ax_VGS_BS, ax_tau_comp, ax_data_comp]:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

##### subplot letters #####
ax_dia.text(-0.05, 1.075, string.ascii_uppercase[0],
            transform=ax_dia.transAxes, size=12, weight='bold')
ax_tau_comp.text(-0.025, 1.075, string.ascii_uppercase[1],
                 transform=ax_tau_comp.transAxes, size=12, weight='bold')
ax_data_comp.text(-0.025, 1.075, string.ascii_uppercase[2],
                  transform=ax_data_comp.transAxes, size=12, weight='bold')

##### save and show #####
plt.savefig('./figures/Supp_Figure_3_syn_bombard.png',
            bbox_inches='tight', dpi=300)
# plt.savefig('./figures/Supp_Figure_3_syn_bombard.pdf',
#             bbox_inches='tight', dpi=300)
plt.show()