# %%
import os
os.chdir('../../')
import string
from matplotlib.colors import LogNorm
from matplotlib import ticker
import scipy.io as sio
import matplotlib.gridspec as gridspec
import seaborn as sns
import scikit_posthocs
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import remove_axis_box
plt.rcParams["font.family"] = "Arial"

# %% load data
sdf_t_df = pd.read_csv('./data/exp/PSTH_t_array.csv', header=None)
sdf_t = sdf_t_df[0].values
VGS_broad = pd.read_csv('./data/exp/VGS_broad_psth.csv')
VGS_narrow = pd.read_csv('./data/exp/VGS_narrow_psth.csv')
df_patch = pd.read_csv('./data/exp/Patch_exp_decay_fit_NS_BS.csv')

df = pd.read_csv('./data/exp/Summary_Decay.csv')

df_vgs_b = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_vgs_b['decay $\tau$'] = np.log10(
    -1 / (df.loc[df.index[~df["VGS BS"].isnull()], "VGS BS"] / 1000))
df_vgs_b['condition'] = "BS"

df_patch_b = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_patch_b['decay $\tau$'] = np.log10(
    -1 / (df.loc[df.index[~df["Patch BS"].isnull()], "Patch BS"] / 1000))
df_patch_b['condition'] = "Patch BS"
df_patch_b = df_patch_b[df_patch_b['decay $\tau$'] <= 5]


df_vgs_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_vgs_n['decay $\tau$'] = np.log10(
    -1 / (df.loc[df.index[~df["VGS NS"].isnull()], "VGS NS"] / 1000))
df_vgs_n['condition'] = "NS"

df_patch_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_patch_n['decay $\tau$'] = np.log10(
    -1 / (df.loc[df.index[~df["Patch NS"].isnull()], "Patch NS"] / 1000))
df_patch_n['condition'] = "Patch NS"

df_tau_pop = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_decay.csv')
df_pop_n = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_pop_n['decay $\tau$'] = np.log10(-1 / (df_tau_pop['VGS NS'] / 1000))
df_pop_n['condition'] = "Pop NS"

df_pop_b = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_pop_b['decay $\tau$'] = np.log10(-1 / (df_tau_pop['VGS BS'] / 1000))
df_pop_b['condition'] = "Pop BS"

df_sum = pd.concat([df_vgs_b, df_pop_b, df_patch_b, df_vgs_n,
                   df_pop_n, df_patch_n], ignore_index=True)

s, p = stats.kruskal(df["VGS BS"], df_tau_pop['VGS BS'], df["Patch BS"],
                     df["VGS NS"], df_tau_pop['VGS NS'], df["Patch NS"], nan_policy="omit")
posthoc_tau = scikit_posthocs.posthoc_dunn(
    df_sum, val_col='decay $\tau$', group_col='condition')

df_offset_vivo = pd.read_csv(
    './data/exp/extracell_offset.csv')
df_coeff_vivo = pd.read_csv(
    './data/exp/extracell_coefficient.csv')
df_AI_vivo = df_offset_vivo / (df_offset_vivo + df_coeff_vivo)

df_vgs_b_AI = pd.DataFrame(columns=['AI', 'condition'])
df_vgs_b_AI['AI'] = df_AI_vivo["VGS BS"]
df_vgs_b_AI['condition'] = "BS"

df_patch_b_AI = pd.DataFrame(columns=['AI', 'condition'])
df_patch_b_AI['AI'] = df_patch["Patch BS AI"]
df_patch_b_AI['condition'] = "Patch BS"

df_pop_offset = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_offset.csv')
df_pop_coeff = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_coeff.csv')
df_pop_AI = df_pop_offset / (df_pop_offset + df_pop_coeff)


df_pop_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_pop_n_AI['AI'] = df_pop_AI['VGS NS']
df_pop_n_AI['condition'] = "Pop NS"

df_pop_b_AI = pd.DataFrame(columns=['AI', 'condition'])
df_pop_b_AI['AI'] = df_pop_AI['VGS BS']
df_pop_b_AI['condition'] = "Pop BS"


df_vgs_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_vgs_n_AI['AI'] = df_AI_vivo["VGS NS"]
df_vgs_n_AI['condition'] = "NS"

df_patch_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_patch_n_AI['AI'] = df_patch["Patch NS AI"]
df_patch_n_AI['condition'] = "Patch NS"
df_sum = pd.concat([df_vgs_b_AI, df_pop_b_AI, df_patch_b_AI,
                   df_vgs_n_AI, df_pop_n_AI, df_patch_n_AI], ignore_index=True)


s2, p2 = stats.kruskal(df_patch["Patch BS AI"], df_patch["Patch NS AI"],
                       df_AI_vivo["VGS BS"], df_AI_vivo["VGS NS"], nan_policy="omit")
posthoc_AI = scikit_posthocs.posthoc_dunn(
    df_sum, val_col='AI', group_col='condition')


# %% Create Figure

np.random.seed(0)  # set seed for stripplot jitter


##### plot style choices #####
order_e = ['VGS BS', 'VGS NS']
order_p_AI = ['Patch BS AI', 'Patch NS AI']
order_p = ['Patch BS', 'Patch NS']
order = ['BS', 'NS', 'Pop BS', 'Pop NS', 'Patch BS', 'Patch NS']
palette_e = ['#40A787', '#2060A7']
palette_e_pop = ['#7DA54B', '#007030']
palette_p = ['#D03050', '#6D398B']
s = 3
lw = 2
decay_ylim = (0.5, 5)
AI_ylim = (-0.02, 1.2)
sdf_xlim = (-200, 800)
alpha = 0.3


#####  layout #####
fig = plt.figure(figsize=(6.5, 5.25))
gs_sum = fig.add_gridspec(2, 5, hspace=0.85, width_ratios=(
    0.15, 0.15, 0.15, 0.1, 0.4), wspace=0.35)
ax_e_AI = fig.add_subplot(gs_sum[0, 0])
ax_e_pop_AI = fig.add_subplot(gs_sum[0, 1])
ax_patch_AI = fig.add_subplot(gs_sum[0, 2])
ax_p_val_AI = fig.add_subplot(gs_sum[0, 4])
ax_e_sum = fig.add_subplot(gs_sum[1, 0])
ax_e_pop_sum = fig.add_subplot(gs_sum[1, 1])
ax_patch_sum = fig.add_subplot(gs_sum[1, 2])
ax_p_val_sum = fig.add_subplot(gs_sum[1, 4])


###### AI plots ##########################################################
ax_e_AI.set_ylim(AI_ylim)
ax_e_pop_AI.set_ylim(AI_ylim)
ax_patch_AI.set_ylim(AI_ylim)

#####  ind  boxplots #####
sns.boxplot(data=df_AI_vivo[['VGS BS', 'VGS NS']], ax=ax_e_AI, order=order_e,
            palette=palette_e, saturation=0.5, boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_AI_vivo[['VGS BS', 'VGS NS']],
              ax=ax_e_AI, order=order_e, palette=palette_e, size=s)
ax_e_AI.set_ylabel("Adaptation Index")
ax_e_AI = remove_axis_box(ax_e_AI, ["right", "top", ])
ax_e_AI.set_xticks(ax_e_AI.get_xticks())
ax_e_AI.set_xticklabels(['BS', 'NS'], rotation=90, ha='center')

##### pop boxplots #####
sns.boxplot(data=df_pop_AI, ax=ax_e_pop_AI, saturation=0.5, order=order_e,
            palette=palette_e_pop,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_pop_AI, ax=ax_e_pop_AI,
              palette=palette_e_pop,  order=order_e, size=s)
ax_e_pop_AI = remove_axis_box(ax_e_pop_AI, ["right", "top", "left"])
ax_e_pop_AI.set_xticks(ax_e_pop_AI.get_xticks())
ax_e_pop_AI.set_xticklabels(['Pop BS', 'Pop NS'], rotation=90, ha='center')


##### patch boxplots #####
sns.boxplot(data=df_patch[['Patch BS AI', 'Patch NS AI']], ax=ax_patch_AI, order=order_p_AI,
            palette=palette_p, saturation=0.75, boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_patch[['Patch BS AI', 'Patch NS AI']],
              ax=ax_patch_AI, order=order_p_AI, palette=palette_p, size=s)
ax_patch_AI = remove_axis_box(ax_patch_AI, ["right", "top", "left"])
ax_patch_AI.set_xticks(ax_patch_AI.get_xticks())
ax_patch_AI.set_xticklabels(['Patch BS', 'Patch NS'], rotation=90, ha='center')


#####  heatmap #####
posthocAI_2 = posthoc_AI.loc[order, order]
mask = np.triu(np.ones_like(posthocAI_2))
np.fill_diagonal(mask, True)
norm = LogNorm(vmin=0.001, vmax=0.75)
sns.heatmap(posthocAI_2, mask=mask,  ax=ax_p_val_AI, square=True,
            norm=norm, cbar_kws={'label': "p-value (Dunn's Test)", 'shrink': 0.9},)
ax_p_val_AI.tick_params(left=False, bottom=False)
ax_p_val_AI.set_xticks(ax_p_val_AI.get_xticks())
xlab = ax_p_val_AI.get_xticklabels()
xlab[-1] = ""
ax_p_val_AI.set_xticklabels(xlab, rotation=90, ha='center')
ylab = ax_p_val_AI.get_yticklabels()
ylab[0] = ""
ax_p_val_AI.set_yticklabels(ylab, rotation=0, ha='right')


###### Tau plots ##########################################################
##### ind boxplots #####
ax_e_sum.set_ylim(decay_ylim)
ax_e_pop_sum.set_ylim(decay_ylim)
ax_patch_sum.set_ylim(decay_ylim)
sns.boxplot(data=np.log10(-1 / (df[['VGS BS', 'VGS NS']]/1000)), ax=ax_e_sum, order=order_e,
            palette=palette_e, saturation=0.5, boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=np.log10(-1 / (df[['VGS BS', 'VGS NS']]/1000)),
              ax=ax_e_sum, order=order_e, palette=palette_e, size=s)
ax_e_sum.set_ylabel("SFA $\\tau$ (ms)")
ax_e_sum = remove_axis_box(ax_e_sum, ["right", "top"])
ax_e_sum.set_xticks(ax_e_sum.get_xticks())
ax_e_sum.set_xticklabels(['BS', 'NS'], rotation=90, ha='center')
ymin, ymax = ax_e_sum.get_ylim()

tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_e_sum.yaxis.set_ticks(tick_range)
tick_range_ticks = np.append([0.], tick_range[:-1])
ax_e_sum.yaxis.set_ticks([np.log10(x) for p in tick_range_ticks for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_e_sum.yaxis.set_ticklabels(['$10^{%s}$' % (round(i)) for i in tick_range])


##### pop boxplots #####
sns.boxplot(data=np.log10(-1 / (df_tau_pop[['VGS BS', 'VGS NS']]/1000)), ax=ax_e_pop_sum,
            saturation=0.5, order=order_e, palette=palette_e_pop,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=np.log10(-1 / (df_tau_pop[['VGS BS', 'VGS NS']]/1000)),
              ax=ax_e_pop_sum, palette=palette_e_pop,  order=order_e, size=s)
ax_e_pop_sum = remove_axis_box(ax_e_pop_sum, ["right", "top", "left"])
ax_e_pop_sum.set_xticks(ax_e_pop_sum.get_xticks())
ax_e_pop_sum.set_xticklabels(['Pop BS', 'Pop NS'], rotation=90, ha='center')

ax_e_pop_sum.set_ylim(0.5, 5)
#####  patch boxplots #####
sns.boxplot(data=np.log10(-1 / (df[['Patch BS', 'Patch NS']]/1000)), ax=ax_patch_sum,
            order=order_p, palette=palette_p, saturation=0.75, boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=np.log10(-1 / (df[['Patch BS', 'Patch NS']]/1000)),
              ax=ax_patch_sum, order=order_p, palette=palette_p, size=s)
ax_patch_sum = remove_axis_box(ax_patch_sum, ["right", "top", "left"])
ax_patch_sum.set_xticks(ax_patch_sum.get_xticks())
ax_patch_sum.set_xticklabels(
    ax_patch_sum.get_xticklabels(), rotation=90, ha='center')
ax_patch_sum.set_ylim(0.5, 5)


#####  heatmap #####
posthoc_tau_2 = posthoc_tau.loc[order, order]
mask = np.triu(np.ones_like(posthoc_tau_2))
np.fill_diagonal(mask, True)
norm = LogNorm(vmin=0.001, vmax=0.75)
sns.heatmap(posthoc_tau_2, mask=mask,  ax=ax_p_val_sum, square=True,
            norm=norm, cbar_kws={'label': "p-value (Dunn's Test)", 'shrink': 0.9},)
ax_p_val_sum.tick_params(left=False, bottom=False)
ax_p_val_sum.set_xticks(ax_p_val_sum.get_xticks())
xlab = ax_p_val_sum.get_xticklabels()
xlab[-1] = ""
ax_p_val_sum.set_xticklabels(xlab, rotation=90, ha='center')
ylab = ax_p_val_sum.get_yticklabels()
ylab[0] = ""
ax_p_val_sum.set_yticklabels(ylab, rotation=0, ha='right')


##### subplot labels #####
ax_e_AI.text(-0.025, 1.075, string.ascii_uppercase[0],
             transform=ax_e_AI.transAxes, size=12, weight='bold')
ax_e_pop_AI.text(-0.025, 1.075, string.ascii_uppercase[1],
                 transform=ax_e_pop_AI.transAxes, size=12, weight='bold')
ax_patch_AI.text(-0.025, 1.075, string.ascii_uppercase[2],
                 transform=ax_patch_AI.transAxes, size=12, weight='bold')
ax_p_val_AI.text(-0.35, 1.115, string.ascii_uppercase[3],
                 transform=ax_p_val_AI.transAxes, size=12, weight='bold')
ax_e_sum.text(-0.025, 1.075, string.ascii_uppercase[4],
              transform=ax_e_sum.transAxes, size=12, weight='bold')
ax_e_pop_sum.text(-0.025, 1.075, string.ascii_uppercase[5],
                  transform=ax_e_pop_sum.transAxes, size=12, weight='bold')
ax_patch_sum.text(-0.025, 1.075, string.ascii_uppercase[6],
                  transform=ax_patch_sum.transAxes, size=12, weight='bold')
ax_p_val_sum.text(-0.35, 1.115, string.ascii_uppercase[7],
                  transform=ax_p_val_sum.transAxes, size=12, weight='bold')

##### save and show #####
plt.savefig('./figures/Figure_5_in_vivo_in_vitro_comp.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Figure_5_in_vivo_in_vitro_comp.pdf',
            bbox_inches='tight', dpi=300)
plt.show()

# %%
