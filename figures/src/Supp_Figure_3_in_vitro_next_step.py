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
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import lighten_color, remove_axis_box, circuit_dia_HP
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


### STEP 2


df2 = pd.read_csv('./data/exp/Patch_exp_decay_fit_NS_BS_2step.csv')

# tau df (log10)
df_tau_BS2 = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_tau_BS2['decay $\tau$'] = np.log10(-1/(df2['Patch BS Exponent'] / 1000))
df_tau_BS2['condition'] = ['BS2' for i in range(
    0, len(df2['Patch BS Exponent']-1))]
df_tau_BS2 = df_tau_BS2[df_tau_BS2['decay $\tau$'] <= 5]
df_tau_NS2 = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_tau_NS2['decay $\tau$'] = np.log10(-1/(df2['Patch NS Exponent'] / 1000))
df_tau_NS2['condition'] = ['NS2' for i in range(
    0, len(df2['Patch NS Exponent']-1))]
df_tau2 = pd.concat([df_tau_BS2, df_tau_NS2])

# AI df
df_AI_BS2 = pd.DataFrame(columns=['AI', 'condition'])
df_AI_BS2['Adaptation Index'] = df2['Patch BS AI']
df_AI_BS2['condition'] = ['BS2' for i in range(0, len(df2['Patch BS AI']-1))]
df_AI_NS2 = pd.DataFrame(columns=['AI', 'condition'])
df_AI_NS2['Adaptation Index'] = df2['Patch NS AI']
df_AI_NS2['condition'] = ['NS2' for i in range(0, len(df2['Patch NS AI']-1))]
df_AI2 = pd.concat([df_AI_BS2, df_AI_NS2])

# Coefficient df
df_coeff_BS2 = pd.DataFrame(columns=['Coefficient', 'condition'])
df_coeff_BS2['Coefficient'] = df2['Patch BS Coefficient']
df_coeff_BS2['condition'] = ['BS2' for i in range(
    0, len(df2['Patch BS Exponent']-1))]
df_coeff_NS2 = pd.DataFrame(columns=['Coefficient', 'condition'])
df_coeff_NS2['Coefficient'] = df2['Patch NS Coefficient']
df_coeff_NS2['condition'] = ['NS2' for i in range(
    0, len(df2['Patch NS Exponent']-1))]
df_coeff2 = pd.concat([df_coeff_BS2, df_coeff_NS2])


# Offset df
df_off_BS2 = pd.DataFrame(columns=['Offset', 'condition'])
df_off_BS2['Offset'] = df2['Patch BS Offset']
df_off_BS2['condition'] = ['BS2' for i in range(
    0, len(df2['Patch BS Exponent']-1))]
df_off_NS2 = pd.DataFrame(columns=['Offset', 'condition'])
df_off_NS2['Offset'] = df2['Patch NS Offset']
df_off_NS2['condition'] = ['NS2' for i in range(
    0, len(df2['Patch NS Exponent']-1))]
df_off2= pd.concat([df_off_BS2, df_off_NS2])



df_tau_BS = pd.concat([df_tau_BS, df_tau_BS2])
df_tau_NS = pd.concat([df_tau_NS, df_tau_NS2])


df_AI_BS = pd.concat([df_AI_BS, df_AI_BS2])
df_AI_NS = pd.concat([df_AI_NS, df_AI_NS2])

df_coeff_BS = pd.concat([df_coeff_BS, df_coeff_BS2])
df_coeff_NS = pd.concat([df_coeff_NS, df_coeff_NS2])

df_off_BS = pd.concat([df_off_BS, df_off_BS2])
df_off_NS = pd.concat([df_off_NS, df_off_NS2])
# %% Create Figure
np.random.seed(0)  # set seed for stripplot jitter
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
# test = 't-test_ind' # 
test = 'Mann-Whitney'
##### plot style choices #####
BS_col =['#D03050', lighten_color('#D03050', amount=1.5)]
NS_col = ['#6D398B', lighten_color('#6D398B', amount=1.3)]

alpha = 0.5
lw = 2
stim_start = 750.
ms = 4
tau_ylim = (0.5,6.25)
AI_ylim = (-0.2,4)
A_ylim = (-1,180)
B_ylim = (-1,155)

pairs1 = [('BS', 'BS2')]
pairs2 = [('NS', 'NS2')]

BS_lab = ['BS', 'BS\n+20pA']
NS_lab = ['NS', 'NS\n+20pA']


##### Layout #####
fig = plt.figure(figsize=(6.5, 5))
gs = fig.add_gridspec(1, 2, hspace=0.75,  wspace=0.6)
gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.4, hspace=0.75, subplot_spec=gs[0])
gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.4, hspace=0.75, subplot_spec=gs[1])
ax_tau_BS = fig.add_subplot(gs1[0, 0])
ax_tau_NS = fig.add_subplot(gs1[0, 1])
ax_AI_BS = fig.add_subplot(gs2[0, 0])
ax_AI_NS = fig.add_subplot(gs2[0, 1])
ax_A_BS = fig.add_subplot(gs1[1, 0])
ax_A_NS = fig.add_subplot(gs1[1, 1])
ax_B_BS = fig.add_subplot(gs2[1, 0])
ax_B_NS = fig.add_subplot(gs2[1, 1])


#####  tau BS #####
sns.boxplot(data=df_tau_BS, x='condition', y='decay $\tau$', ax=ax_tau_BS,
            palette=BS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_tau_BS, x='condition', y='decay $\tau$',
              ax=ax_tau_BS, 
              palette=BS_col,
                s=ms)
ymin, ymax = ax_tau_BS.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(7)+1, 1)
ax_tau_BS.yaxis.set_ticks(tick_range)
tick_range_ticks = np.append([0.5], tick_range[:-1])
ax_tau_BS.yaxis.set_ticks([np.log10(x) for p in [0,1,2,3,4,5,6,7] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_tau_BS.yaxis.set_ticklabels(['$10^{%s}$' % (round(i)) for i in tick_range])
ax_tau_BS.set_ylabel('I-SFA $\\tau$ (ms)')
annotator = Annotator(ax_tau_BS, pairs1, data=df_tau_BS,
                      x='condition',  y='decay $\tau$')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_tau_BS, test_results1 = annotator.annotate()
ax_tau_BS.set_ylim(tau_ylim)

#####  tau NS #####
sns.boxplot(data=df_tau_NS, x='condition', y='decay $\tau$', ax=ax_tau_NS,
            palette=NS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_tau_NS, x='condition', y='decay $\tau$',
              ax=ax_tau_NS, 
              palette=NS_col,
                s=ms)
ymin, ymax = ax_tau_NS.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_tau_NS.yaxis.set_ticklabels(['$10^{%s}$' % (round(i)) for i in tick_range])
annotator = Annotator(ax_tau_NS, pairs2, data=df_tau_NS,
                      x='condition',  y='decay $\tau$', line_height=-5)
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_tau_NS, test_results1 = annotator.annotate()
ax_tau_NS.set_ylim(tau_ylim)


#####  AI BS #####
sns.boxplot(data=df_AI_BS, x='condition', y='Adaptation Index', ax=ax_AI_BS,
            palette=BS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_AI_BS, x='condition', y='Adaptation Index',
              ax=ax_AI_BS, 
              palette=BS_col,
                s=ms)
ax_AI_BS.set_ylabel('Adaptation Index')
annotator = Annotator(ax_AI_BS, pairs1, data=df_AI_BS,
                      x='condition',  y='Adaptation Index')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_AI_BS, test_results1 = annotator.annotate()
ax_AI_BS.set_ylim(AI_ylim)

#####  AI NS #####
sns.boxplot(data=df_AI_NS, x='condition', y='Adaptation Index', ax=ax_AI_NS,
            palette=NS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_AI_NS, x='condition', y='Adaptation Index',
              ax=ax_AI_NS, 
              palette=NS_col,
                s=ms)
annotator = Annotator(ax_AI_NS, pairs2, data=df_AI_NS,
                      x='condition',  y='Adaptation Index')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_AI_NS, test_results1 = annotator.annotate()
ax_AI_NS.set_ylim(AI_ylim)






#####  coeff BS #####
sns.boxplot(data=df_coeff_BS, x='condition', y='Coefficient', ax=ax_A_BS,
            palette=BS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_coeff_BS, x='condition', y='Coefficient',
              ax=ax_A_BS, 
              palette=BS_col,
                s=ms)
ax_A_BS.set_ylabel('Coefficient')
annotator = Annotator(ax_A_BS, pairs1, data=df_coeff_BS,
                      x='condition',  y='Coefficient')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_A_BS, test_results1 = annotator.annotate()
ax_A_BS.set_ylim(A_ylim)

#####  coeff NS #####
sns.boxplot(data=df_coeff_NS, x='condition', y='Coefficient', ax=ax_A_NS,
            palette=NS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_coeff_NS, x='condition', y='Coefficient',
              ax=ax_A_NS, 
              palette=NS_col,
                s=ms)
annotator = Annotator(ax_A_NS, pairs2, data=df_coeff_NS,
                      x='condition',  y='Coefficient')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_A_NS, test_results1 = annotator.annotate()
ax_A_NS.set_ylim(A_ylim)




#####  baseline BS #####
sns.boxplot(data=df_off_BS, x='condition', y='Offset', ax=ax_B_BS,
            palette=BS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_off_BS, x='condition', y='Offset',
              ax=ax_B_BS, 
              palette=BS_col,
                s=ms)
ax_B_BS.set_ylabel('Offset')
annotator = Annotator(ax_B_BS, pairs1, data=df_off_BS,
                      x='condition',  y='Offset')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_B_BS, test_results1 = annotator.annotate()
ax_B_BS.set_ylim(B_ylim)

#####  coeff NS #####
sns.boxplot(data=df_off_NS, x='condition', y='Offset', ax=ax_B_NS,
            palette=NS_col,
              boxprops=dict(alpha=.5), saturation=0.5, showfliers=False)
sns.stripplot(data=df_off_NS, x='condition', y='Offset',
              ax=ax_B_NS, 
              palette=NS_col,
                s=ms)
annotator = Annotator(ax_B_NS, pairs2, data=df_off_NS,
                      x='condition',  y='Offset')
annotator.configure(test=test, loc='inside', fontsize=9,
                    show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.apply_test()
ax_B_NS, test_results1 = annotator.annotate()
ax_B_NS.set_ylim(B_ylim)

ax_A_BS.set_ylabel('Coefficient (Hz)')
ax_B_BS.set_ylabel('Baseline (Hz)')

##### remove axis boxes, axis limits, label adding #####

i = 0
for ax in [ax_tau_BS, ax_AI_BS, ax_A_BS, ax_B_BS]:
    ax = remove_axis_box(ax, ["right", "top"])
    ax.set_xlabel("")
    ax.set_xticklabels(BS_lab)
    ax.text(-0.025, 1.075, string.ascii_uppercase[i],
                    transform=ax.transAxes, size=12, weight='bold')
    i += 1
for ax in [ax_tau_NS, ax_AI_NS, ax_A_NS, ax_B_NS]:
    ax = remove_axis_box(ax, ["right", "top", "left"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(NS_lab)


plt.savefig('./figures/Supp_Figure_3_next_step.png',
            bbox_inches='tight', dpi=300)
# plt.savefig('./figures/Supp_Figure_3_next_step.pdf',
#             bbox_inches='tight', dpi=300)
plt.show()

# %%
