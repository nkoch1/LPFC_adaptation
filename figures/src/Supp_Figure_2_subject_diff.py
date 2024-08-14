#%%
import os
os.chdir('../../')
import string
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import remove_axis_box
plt.rcParams["font.family"] = "Arial"

# %% load data generated in simlu_recorded_sdf.m
df_VGS = pd.read_csv('./data/exp/VGS_simul_recorded_lat.csv')
df_VGS['Exponent (c)'] = np.log10(-1/(df_VGS['Exponent (c)'] / 1000))
di = {1: "T", 2: "B"}
df_VGS = df_VGS.replace({"anim": di})

df_VGS['Latency'] = df_VGS['Latency']/1000

# %% Create Figure
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
anim_palettte = ['tab:purple', 'tab:green'],
session_palette = ['olivedrab', 'forestgreen', 'darkgreen',
                   'mediumorchid', 'mediumvioletred', 'darkmagenta']

##### layout #####
fig = plt.figure(figsize=(6.5, 5))
gs = fig.add_gridspec(1, 3,  hspace=0.5,  wspace=0.5)
gs0 = gridspec.GridSpecFromSubplotSpec(
    2, 1, wspace=0.45, hspace=0.75, subplot_spec=gs[0])
gs1 = gridspec.GridSpecFromSubplotSpec(
    2, 1, wspace=0.45, hspace=0.75, subplot_spec=gs[1])
gs2 = gridspec.GridSpecFromSubplotSpec(
    2, 1, wspace=0.45, hspace=0.75, subplot_spec=gs[2])
ax_max_VGS = fig.add_subplot(gs0[0, 0])
ax_A_VGS = fig.add_subplot(gs0[1, 0])
ax_lat_VGS = fig.add_subplot(gs2[0, 0])
ax_tau_VGS = fig.add_subplot(gs1[0, 0])
ax_B_VGS = fig.add_subplot(gs1[1, 0])


##### average frequency #####
sns.boxplot(data=df_VGS, x='anim', y='PSTH', ax=ax_max_VGS,
            boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_VGS, x='anim', y='PSTH', ax=ax_max_VGS)
ax_max_VGS.set_ylabel('Max Frequency (Hz)')
pairs1 = [('B', 'T')]
annotator = Annotator(ax_max_VGS, pairs1, data=df_VGS, x='anim',  y='PSTH')
annotator.configure(test='Mann-Whitney', loc='outside',
                    pvalue_thresholds=pval_thresh, fontsize=10, show_test_name=False)
annotator.apply_test()
ax_max_VGS, test_results1 = annotator.annotate()
ax_max_VGS.set_ylim(0, ax_max_VGS.get_ylim()[1])

##### coefficient #####
sns.boxplot(data=df_VGS, x='anim', y='Coefficient (b)', ax=ax_A_VGS,
            boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_VGS, x='anim', y='Coefficient (b)', ax=ax_A_VGS)
ax_A_VGS.set_ylabel('Coefficient (Hz)')
pairs1 = [('B', 'T')]
annotator = Annotator(ax_A_VGS, pairs1, data=df_VGS,
                      x='anim',  y='Coefficient (b)')
annotator.configure(test='Mann-Whitney', loc='outside',
                    pvalue_thresholds=pval_thresh, fontsize=10, show_test_name=False)
annotator.apply_test()
ax_A_VGS, test_results1 = annotator.annotate()
ax_A_VGS.set_ylim(0, ax_A_VGS.get_ylim()[1])

##### Baseline #####
sns.boxplot(data=df_VGS, x='anim', y='Offset (a)', ax=ax_B_VGS,
            boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_VGS, x='anim', y='Offset (a)', ax=ax_B_VGS)
ax_B_VGS.set_ylabel('Baseline (Hz)')
pairs1 = [('B', 'T')]
annotator = Annotator(ax_B_VGS, pairs1, data=df_VGS, x='anim',  y='Offset (a)')
annotator.configure(test='Mann-Whitney', loc='outside',
                    pvalue_thresholds=pval_thresh, fontsize=10, show_test_name=False)
annotator.apply_test()
ax_B_VGS, test_results1 = annotator.annotate()

##### Decay tau #####
sns.boxplot(data=df_VGS, x='anim', y='Exponent (c)', ax=ax_tau_VGS,
            boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_VGS, x='anim', y='Exponent (c)', ax=ax_tau_VGS)
ax_tau_VGS.set_ylabel('E-SFA $\\tau$ (ms)')
pairs1 = [('B', 'T')]
annotator = Annotator(ax_tau_VGS, pairs1, data=df_VGS,
                      x='anim',  y='Exponent (c)')
annotator.configure(test='Mann-Whitney', loc='outside',
                    pvalue_thresholds=pval_thresh, fontsize=10, show_test_name=False)
annotator.apply_test()
ax_tau_VGS, test_results1 = annotator.annotate()
ymin, ymax = ax_tau_VGS.get_ylim()
tick_range = np.arange(1, ymax)
ax_tau_VGS.yaxis.set_ticks(tick_range)
ax_tau_VGS.yaxis.set_ticks([np.log10(x) for p in [0,1,2,3] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_tau_VGS.yaxis.set_major_formatter(ticker.FormatStrFormatter(r"10$^{%d}$"))
ax_tau_VGS.set_ylim(ymin, ymax)

##### Latency #####
sns.boxplot(data=df_VGS, x='anim', y='Latency', ax=ax_lat_VGS,
            boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_VGS, x='anim', y='Latency', ax=ax_lat_VGS)
ax_lat_VGS.set_ylabel('T$_{\mathrm{max}}$ (s)')
pairs1 = [('B', 'T')]
annotator = Annotator(ax_lat_VGS, pairs1, data=df_VGS, x='anim',  y='Latency')
annotator.configure(test='Mann-Whitney',  loc='outside',
                    pvalue_thresholds=pval_thresh, fontsize=10, show_test_name=False)
annotator.apply_test()
ax_lat_VGS, test_results1 = annotator.annotate()
ax_lat_VGS.set_ylim(0, ax_lat_VGS.get_ylim()[1])

###### remove axis box sides #####
for ax in [ax_max_VGS, ax_A_VGS,  ax_tau_VGS, ax_B_VGS, ax_lat_VGS]:
    ax = remove_axis_box(ax, ["right", "top"])
    ax.set_xlabel('Macaque')

##### subplot letters #####
let_x = -0.05
let_y = 1.15
ax_max_VGS.text(
    let_x, let_y, string.ascii_uppercase[0], transform=ax_max_VGS.transAxes, size=12, weight='bold')
ax_A_VGS.text(let_x, let_y, string.ascii_uppercase[2],
              transform=ax_A_VGS.transAxes, size=12, weight='bold')
ax_tau_VGS.text(
    let_x, let_y, string.ascii_uppercase[1], transform=ax_tau_VGS.transAxes, size=12, weight='bold')
ax_B_VGS.text(let_x, let_y, string.ascii_uppercase[3],
              transform=ax_B_VGS.transAxes, size=12, weight='bold')
ax_lat_VGS.text(
    let_x, let_y, string.ascii_uppercase[4], transform=ax_lat_VGS.transAxes, size=12, weight='bold')

##### save and show #####
plt.savefig('./figures/Supp_Figure_2_subject_diff.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Supp_Figure_2_subject_diff.pdf',
            bbox_inches='tight', dpi=300)
plt.show()
