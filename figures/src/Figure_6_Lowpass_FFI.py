
#%%
import os
os.chdir('../../')
import string
from ast import literal_eval
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from statannotations.Annotator import Annotator
import seaborn as sns
import scikit_posthocs
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.transforms
from src.plt_fxns import lighten_color, remove_axis_box, circuit_dia_LP, circuit_dia_OU
plt.rcParams["font.family"] = "Arial"


# %% load data
step_tau = pd.read_csv("./data/sims/Pospischil_step_sim_tau.csv", header=None)
step_F = pd.read_csv("./data/sims/Pospischil_step_sim_F.csv")
step_t = pd.read_csv("./data/sims/Pospischil_step_sim_t.csv")


VGS_tau = pd.read_csv(
    "./data/sims/Pospischil_filter_sim_tau.csv", header=None)
VGS_psth = pd.read_csv(
    "./data/sims/Pospischil_filter_sim_psth.csv")
VGS_psth_t = pd.read_csv(
    "./data/sims/Pospischil_filter_sim_psth_t.csv")
VGS_psth_array = np.array(
    [np.array(literal_eval(VGS_psth.iloc[i, 0])) for i in range(VGS_psth.shape[0])])
VGS_psth_t_array = np.array([np.array(literal_eval(
    VGS_psth_t.iloc[i, 0])) for i in range(VGS_psth_t.shape[0])])



VGS_tau_syn = pd.read_csv(
    "./data/sims/Pospischil_filter_OU_sim_tau.csv", header=None)
VGS_psth_syn = pd.read_csv(
    "./data/sims/Pospischil_filter_OU_sim_psth.csv")
VGS_psth_t_syn = pd.read_csv(
    "./data/sims/Pospischil_filter_OU_sim_psth_t.csv")
VGS_psth_array_syn = np.array(
    [np.array(literal_eval(VGS_psth_syn.iloc[i, 0])) for i in range(VGS_psth_syn.shape[0])])
VGS_psth_t_array_syn = np.array([np.array(literal_eval(
    VGS_psth_t_syn.iloc[i, 0])) for i in range(VGS_psth_t_syn.shape[0])])


VGS_sum_syn = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_OU_sim.csv")
VGS_sum_syn.rename(columns={"τ_step": "Step", "τ_VGS": "BS"}, inplace=True)

VGS_sum_AI_syn = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_OU_sim_AI.csv")
VGS_sum_AI_syn.rename(columns={"AI_step": "Step", "AI_VGS": "BS"}, inplace=True)





VGS_I_tau = pd.read_csv(
    "./data/sims/Pospischil_filter_Inh_sim_tau.csv", header=None)
VGS_I_psth = pd.read_csv(
    "./data/sims/Pospischil_filter_Inh_sim_psth.csv")
VGS_I_psth_t = pd.read_csv(
    "./data/sims/Pospischil_filter_Inh_sim_psth_t.csv")
VGS_I_psth_array = np.array([np.array(literal_eval(
    VGS_I_psth.iloc[i, 0])) for i in range(VGS_I_psth.shape[0])])
VGS_I_psth_t_array = np.array([np.array(literal_eval(
    VGS_I_psth_t.iloc[i, 0])) for i in range(VGS_I_psth_t.shape[0])])

VGS_sum = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_Inh_sim.csv")
VGS_sum.rename(columns={"τ_step": "Step", "τ_VGS": "E",
               "τ_VGS_I": "E+I"}, inplace=True)
VGS_sum['E+Syn'] = VGS_sum_syn['BS']
VGS_sum_AI = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_Inh_sim_AI.csv")
VGS_sum_AI.rename(columns={"AI_step": "Step",
                  "AI_VGS": "E", "AI_VGS_I": "E+I"}, inplace=True)

VGS_sum_AI['E+Syn'] = VGS_sum_AI_syn['BS']
VGS_sum = VGS_sum[["Step", 'E',  'E+Syn','E+I']]
VGS_sum_AI = VGS_sum_AI[["Step", 'E', 'E+Syn','E+I']]


t = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_T.csv', header=None)
VGS_narrow = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_VGS_NS.csv', header=None)


s, p = stats.kruskal(VGS_sum["Step"], VGS_sum['E'],
                     VGS_sum['E+I'], VGS_sum['E+Syn'], nan_policy="omit")

df_Step = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_Step['decay $\tau$'] = VGS_sum['Step']
df_Step['condition'] = "Step"

df_BS = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_BS['decay $\tau$'] = VGS_sum['E']
df_BS['condition'] = "E"


df_BS_syn = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_BS_syn['decay $\tau$'] = VGS_sum_syn['BS'] *2
df_BS_syn['condition'] = "E+Syn"

df_BS_I = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_BS_I['decay $\tau$'] = VGS_sum['E+I']
df_BS_I['condition'] = "E+I"

df_sum_mod = pd.concat([df_Step, df_BS,df_BS_I, df_BS_syn], ignore_index=True) #
posthoc_mod = scikit_posthocs.posthoc_dunn(
    df_sum_mod, val_col='decay $\tau$', group_col='condition')



df_Step_AI = pd.DataFrame(columns=['AI', 'condition'])
df_Step_AI['AI'] = VGS_sum_AI['Step']
df_Step_AI['condition'] = "Step"

df_BS_AI = pd.DataFrame(columns=['AI', 'condition'])
df_BS_AI['AI'] = VGS_sum_AI['E']
df_BS_AI['condition'] = "E"


df_BS_AI_syn = pd.DataFrame(columns=['decay $\tau$', 'condition'])
df_BS_AI_syn['AI'] = VGS_sum_syn['BS']
df_BS_AI_syn['condition'] = "E+Syn"

df_BS_I_AI = pd.DataFrame(columns=['AI', 'condition'])
df_BS_I_AI['AI'] = VGS_sum_AI['E+I']
df_BS_I_AI['condition'] = "E+I"
s, p = stats.kruskal(
    VGS_sum_AI["Step"], VGS_sum_AI['E'], VGS_sum_AI['E+I'],VGS_sum_AI['E+Syn'],  nan_policy="omit")

df_sum_mod_AI = pd.concat(
    [df_Step_AI, df_BS_AI, df_BS_AI_syn, df_BS_I_AI], ignore_index=True)
posthoc_mod_AI = scikit_posthocs.posthoc_dunn(
    df_sum_mod_AI, val_col='AI', group_col='condition')


df_BS = pd.DataFrame(columns=['Decay $\tau$ (ms)', 'condition'])
df_BS['decay $\tau$'] = VGS_sum['E']
df_BS['condition'] = "E"

df_BS_I = pd.DataFrame(columns=['Decay $\tau$ (ms)', 'condition'])
df_BS_I['decay $\tau$'] = VGS_sum['E+I']
df_BS_I['condition'] = "E+I"


df_BS_syn = pd.DataFrame(columns=['Decay $\tau$ (ms)', 'condition'])
df_BS_syn['decay $\tau$'] = VGS_sum_syn['BS']
df_BS_syn['condition'] = "E+Syn"


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

df_sum_data = pd.concat(
    [df_BS, df_BS_I,df_BS_syn, df_vgs_n, df_pop_n], ignore_index=True)

s, p = stats.kruskal(df_BS["decay $\tau$"], df_BS_I['decay $\tau$'], df_BS_syn['decay $\tau$'],
                     df_vgs_n["decay $\tau$"], df_pop_n["decay $\tau$"], nan_policy="omit")
posthoc_data = scikit_posthocs.posthoc_dunn(
    df_sum_data, val_col='decay $\tau$', group_col='condition')


df_BS_AI = pd.DataFrame(columns=['AI', 'condition'])
df_BS_AI['AI'] = VGS_sum_AI['E']
df_BS_AI['condition'] = "E"

df_BS_syn_AI = pd.DataFrame(columns=['AI', 'condition'])
df_BS_syn_AI['AI'] = VGS_sum_AI_syn['BS']
df_BS_syn_AI['condition'] = "E+Syn"


df_BS_I_AI = pd.DataFrame(columns=['AI', 'condition'])
df_BS_I_AI['AI'] = VGS_sum_AI['E+I']
df_BS_I_AI['condition'] = "E+I"

df_offset_vivo = pd.read_csv(
    './data/exp/extracell_offset.csv')
df_coeff_vivo = pd.read_csv(
    './data/exp/extracell_coefficient.csv')
df_AI_vivo = df_offset_vivo / (df_offset_vivo + df_coeff_vivo)
df_vgs_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_vgs_n_AI['AI'] = df_AI_vivo["VGS NS"]
df_vgs_n_AI['condition'] = "NS"

df_pop_offset = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_offset.csv')
df_pop_coeff = pd.read_csv(
    './data/exp/Extracell_PSTH_pop_coeff.csv')
df_pop_AI = df_pop_offset / (df_pop_offset + df_pop_coeff)
df_pop_n_AI = pd.DataFrame(columns=['AI', 'condition'])
df_pop_n_AI['AI'] = df_pop_AI['VGS NS']
df_pop_n_AI['condition'] = "Pop NS"

df_sum_data_AI = pd.concat(
    [df_BS_AI, df_BS_I_AI,df_BS_syn_AI, df_vgs_n_AI, df_pop_n_AI], ignore_index=True)
s, p = stats.kruskal(df_BS_AI["AI"], df_BS_I_AI['AI'],df_BS_syn_AI['AI'],
                     df_vgs_n_AI["AI"], df_pop_n_AI["AI"], nan_policy="omit")
posthoc_data_AI = scikit_posthocs.posthoc_dunn(
    df_sum_data_AI, val_col='AI', group_col='condition')

psth_max_array = np.zeros(VGS_psth_array.shape[0])
for i in range(VGS_psth_array.shape[0]):
    psth_max_array[i] = np.max(VGS_I_psth_array[i])



# %% Create Figure
np.random.seed(0)  # set seed for stripplot jitter
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choices #####
c_step = lighten_color('#6D398B', amount=1.25)
c_E = 'tab:grey'
c_EI = '#C02717'
c_Esyn = 'tab:brown'
step_labels = ['I-SFA', 'Exc','Exc\n+Syn', 'Exc\n+Inh']
data_labels = ['NS', 'Pop NS', 'Exc','Exc\n+Syn','Exc\n+Inh']
alpha = 0.5
lw = 2
stim_start = 750.
ms = 3
psth_xlim = (-250, 1000)
psth_ylim = (0, 50)
tau_comp_lim = (-0.5, 4.5)
tau_comp_ylim = (1.75, 3.5)
size = 20
lw_m = 2
mut_scale_arrow = 10

##### Layout #####
fig = plt.figure(figsize=(6.5, 7.5))
gs = fig.add_gridspec(4, 3, left=0.025, right=0.975, top=0.975, bottom=0.075, width_ratios=[1, 1., 1.5],
                      height_ratios=[0.65, 0.45, 0.75, 1.8],
                      hspace=0.6,  wspace=0.75)
gs_l = gridspec.GridSpecFromSubplotSpec(
    3, 2, width_ratios=[0.75, 1.5], wspace=0.5, hspace=0.6, subplot_spec=gs[0:2, 0:2])
gs_r = gridspec.GridSpecFromSubplotSpec(
    2, 1, wspace=0.35, hspace=0.7, subplot_spec=gs[0:2, 2])
gs_b0 = gridspec.GridSpecFromSubplotSpec(
    1, 4, wspace=0.3,width_ratios=[0.09, 1., 0.075, 1.], hspace=0.75, subplot_spec=gs[2, :])
gs_b = gridspec.GridSpecFromSubplotSpec(2, 5, wspace=0.15, width_ratios=[
                                         0.15, 0.0375, 0.85, 0.05, 0.85], hspace=0.95, subplot_spec=gs[3, :])
# 0.0375, 0.85, 0.05, 0.85], hspace=0.95, subplot_spec=gs[3, :])

ax_dia = fig.add_subplot(gs_l[0, 0])
ax_VGS_BS = fig.add_subplot(gs_l[0, 1])
ax_dia1 = fig.add_subplot(gs_l[1, 0])
ax_VGS_BS_syn = fig.add_subplot(gs_l[1, 1])
ax_dia2 = fig.add_subplot(gs_l[2, 0])
ax_VGS_BS_I = fig.add_subplot(gs_l[2, 1])
ax_VGS_sum = fig.add_subplot(gs_r[0])
ax_VGS_AI = fig.add_subplot(gs_r[1])
ax_tau_comp = fig.add_subplot(gs_b[0, 1:3])
ax_amp_comp = fig.add_subplot(gs_b[1, 2])
ax_data_comp = fig.add_subplot(gs_b0[1])
ax_data_AI_comp = fig.add_subplot(gs_b0[3])

# schematic layout
gs_schem = gridspec.GridSpecFromSubplotSpec(2, 3,  hspace=0.5,  wspace=0.75, subplot_spec=gs_b[:, 4])
ax_I_fast = fig.add_subplot(gs_schem[1, 0])
ax_I_med = fig.add_subplot(gs_schem[1, 1])
ax_I_slow = fig.add_subplot(gs_schem[1, 2])
ax_I_fast_out = fig.add_subplot(gs_schem[0, 0])
ax_I_med_out = fig.add_subplot(gs_schem[0, 1])
ax_I_slow_out = fig.add_subplot(gs_schem[0, 2])


##### Circuit diagrams #####
ax_dia = circuit_dia_LP(ax_dia, inhib=False, ylim=(0.5, 3), lw_c=3, lw_con=2,  c_E='#7DA54B',
                        c_M=c_E, c_I="tab:grey", lw_sdf=0.5, lw_f=2, fs_labels=7, fs_eqn=7, fs_axis=6)

ax_dia1 = circuit_dia_OU(ax_dia1, lw_c=3, lw_con=2,  c_E="#7DA54B", c_M=c_Esyn,
                        c_gE="peru", c_gI="maroon", lw_sdf=0.5, lw_f=2, fs_labels=7,fs_labels2=6.5, fs_eqn=7, fs_axis=6)


ax_dia2 = circuit_dia_LP(ax_dia2,ylim=(0., 2.5), inhib=True, lw_c=3, lw_con=2,  c_E='#7DA54B',
                         c_M=c_EI, c_I="k", lw_sdf=0.5, lw_f=2, fs_labels=7, fs_eqn=7, fs_axis=6)

# ax_dia2 = circuit_dia_LP(ax_dia2, inhib=True, lw_c=3, lw_con=2,  c_E='#7DA54B',
#                          c_M=c_EI, c_I="k", lw_sdf=0.5, lw_f=2, fs_labels=8, fs_eqn=10, fs_axis=7)
#####  PSTH a #####
sat = np.linspace(0.5, 1.5, VGS_psth_array.shape[0])
for i in range(VGS_psth_array.shape[0]):
    ax_VGS_BS.plot(VGS_psth_t_array[i], VGS_psth_array[i],
                   alpha=alpha, color=lighten_color(c_E, sat[i]))
ax_VGS_BS.set_ylim(0, ax_VGS_BS.get_ylim()[1])
ax_VGS_BS.set_xticks([0, 500, 1000])
ax_VGS_BS.set_xticklabels(['Stim. Onset',  '500', '1000'])
ax_VGS_BS.set_title("Exc", y=0.7, x = 0.85, color=c_E)

#####  PSTH syn #####
sat = np.linspace(0.5, 1.25, VGS_psth_array_syn.shape[0])
for i in range(VGS_psth_array_syn.shape[0]):
    ax_VGS_BS_syn.plot(VGS_psth_t_array_syn[i], VGS_psth_array_syn[i],
                   alpha=alpha, color=lighten_color(c_Esyn, sat[i]))
ax_VGS_BS_syn.set_ylim(0, ax_VGS_BS_syn.get_ylim()[1])
ax_VGS_BS_syn.set_xticks([0, 500, 1000])
ax_VGS_BS_syn.set_xticklabels(['Stim. Onset',  '500', '1000'])
ax_VGS_BS_syn.set_title("Exc+Syn", y=0.7, x = 0.85, color=c_Esyn)

#####  PSTH b #####
sat = np.linspace(0.5, 1.25, VGS_I_psth_array.shape[0])
for i in range(VGS_psth_array.shape[0]):
    ax_VGS_BS_I.plot(VGS_I_psth_t_array[i], VGS_I_psth_array[i],
                     alpha=alpha, color=lighten_color(c_EI, sat[i]))
ax_VGS_BS_I.set_ylim(0, ax_VGS_BS_I.get_ylim()[1])
ax_VGS_BS_I.set_xticks([0, 500, 1000])
ax_VGS_BS_I.set_xticklabels(['Stim. Onset',  '500', '1000'])
ax_VGS_BS_I.set_title("Exc+Inh", y=0.7, x = 0.85, color=c_EI)

#### I-SFA vs E-SFA ####
x = VGS_sum['Step']
y = 10**VGS_sum['E+I']

# plot scatter
ax_tau_comp.scatter(x, y, c=c_EI, edgecolors=c_step, linewidths=lw_m, s=size,)
divider = make_axes_locatable(ax_tau_comp)
axHistx = divider.append_axes("bottom", 0.2, pad=0.1, sharex=ax_tau_comp)
axHisty = divider.append_axes("left", 0.2, pad=0.1, sharey=ax_tau_comp)

# make some labels invisible

ax_tau_comp.tick_params(labelbottom=False, bottom=False,
                        left=False, labelleft=False, right=False, labelright=False)
axHistx.tick_params(labelbottom=True, bottom=True,
                    left=False, labelleft=False, right=False, labelright=False)
axHisty.tick_params(labelleft=True, left=True,
                    bottom=False, labelbottom=True, top=False, labeltop=True)
# plot x and y axis boxplots
np.random.seed(0)  # set seed for stripplot jitter
sns.boxplot(x, ax=axHistx, color=c_step, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(x, ax=axHistx, color='w', edgecolor=c_step,
              size=np.sqrt(size), linewidth=lw_m, )
sns.boxplot(y=y, ax=axHisty, color=c_EI, saturation=0.5,
            boxprops=dict(alpha=.5), orient='x', showfliers=False)
sns.stripplot(y=y, ax=axHisty, color=c_EI, orient='v',
              size=np.sqrt(size)+0.5, alpha=1.)


# axes labels
axHistx.set_xlabel('I-SFA $\\tau$ (ms)')
axHisty.set_ylabel('E-SFA $\\tau$ (ms)')

# line horizontal
ind_min_y = np.argmin(y)
xy1 = (x[ind_min_y], y[ind_min_y])
xy2 = (0., y[ind_min_y])
print(xy1)
print(xy2)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                      axesA=ax_tau_comp, axesB=axHisty, color=c_EI, linestyle=":")
axHisty.add_artist(con)

# vertical line
xy1 = (x[ind_min_y], y[ind_min_y])
xy2 = (x[ind_min_y], 0.)
print(xy1)
print(xy2)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                      axesA=ax_tau_comp, axesB=axHistx, color=c_step, linestyle=":")
axHistx.add_artist(con)

# tick labels
axHistx.set_xticks([0, 1, 2, 3, 4])
axHistx.xaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
axHisty.set_yticks([50, 100])
axHistx.xaxis.set_ticks([np.log10(x) for p in [0,1,2,3,4] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_tau_comp.tick_params(axis="x", which="both", length=0.)

#### I-SFA vs Max response ####
x_max = VGS_sum['Step']
y_max = psth_max_array

rho, p = stats.spearmanr(x_max, y_max, nan_policy='omit')
print("rho = {}, p = {}".format(rho, p))

ax_amp_comp.scatter(x_max, y_max, c=c_EI, edgecolors=c_step,
                    linewidths=lw_m, s=size,)
divider = make_axes_locatable(ax_amp_comp)
axHistx_max = divider.append_axes("bottom", 0.2, pad=0.1, sharex=ax_amp_comp)

# make some labels invisible
ax_amp_comp.tick_params(labelbottom=False, bottom=False,
                        left=True, labelleft=True, right=False, labelright=False)
axHistx_max.tick_params(labelbottom=True, bottom=True,
                        left=False, labelleft=False, right=False, labelright=False)

np.random.seed(0)  # set seed for stripplot jitter
sns.boxplot(x_max, ax=axHistx_max, color=c_step, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(x_max, ax=axHistx_max, color='w', edgecolor=c_step,
              size=np.sqrt(size), linewidth=lw_m, )

# axes labels
axHistx_max.set_xlabel('I-SFA $\\tau$ (ms)')
ax_amp_comp.set_ylabel('Maximal \n Response (Hz)', y=0.25)
axHistx_max.set_xticks([0, 1, 2, 3, 4])
axHistx_max.xaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
axHistx_max.xaxis.set_ticks([np.log10(x) for p in [0,1,2,3,4] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_amp_comp.tick_params(axis="x", which="both", length=0.)

##### decay tau boxplots #####
cond = VGS_sum.columns.tolist()
order_e = cond
sns.boxplot(data=VGS_sum, ax=ax_VGS_sum, saturation=0.5, order=order_e, palette=[
            c_step, c_E,c_Esyn, c_EI],  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=VGS_sum, ax=ax_VGS_sum, palette=[
              c_step, c_E,c_Esyn, c_EI],  order=order_e, s=ms)
ax_VGS_sum.set_xticks(ax_VGS_sum.get_xticks())
ax_VGS_sum.set_xticklabels(
    ax_VGS_sum.get_xticklabels(), rotation=0, ha='center')
ax_VGS_sum.set_yticks([1, 2, 3, 4, 5])
ymin, ymax = ax_VGS_sum.get_ylim()
ax_VGS_sum.yaxis.set_ticks([np.log10(x) for p in [0,1,2,3,4,5] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)
ax_VGS_sum.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
ax_VGS_sum.set_ylabel("SFA $\\tau$ (ms)")
# pairs1 = [(cond[0], cond[1]), (cond[0], cond[2]), (cond[1], cond[2]),
#           (cond[1], cond[3]), (cond[2], cond[3]),(cond[0], cond[3]),]
pairs1 = [(cond[0], cond[1]), 
        #   (cond[0], cond[3]), #(cond[1], cond[2]),
        #   (cond[1], cond[3]), (cond[2], cond[3]),
          (cond[0], cond[2]),
        ]
pvalues = []
for pair in pairs1:
    p = posthoc_mod.loc[pair[0], pair[1]]
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_VGS_sum, pairs1, data=VGS_sum,  order=order_e)
annotator.configure(test=None, test_short_name="", loc='inside', 
                    fontsize=8, show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()
ax_VGS_sum.set_ylim(ymin, 6.5)
ax_VGS_sum.set_xticklabels(step_labels)


# hide y-axis for stats region
ystart = 5.05
yend = 6.150
ax_VGS_sum.add_patch(patches.Rectangle((-0.6, ystart), 0.2, yend, fill=True, color="white",
                                       transform=ax_VGS_sum.transData, clip_on=False, zorder=3))

##### adaptation index boxplots #####
cond = VGS_sum_AI.columns.tolist()
order_e = cond
sns.boxplot(data=VGS_sum_AI, ax=ax_VGS_AI, saturation=0.5, order=order_e, palette=[
            c_step, c_E,c_Esyn, c_EI],  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=VGS_sum_AI, ax=ax_VGS_AI, palette=[
              c_step, c_E, c_Esyn,c_EI],  order=order_e, s=ms)
ax_VGS_AI.set_ylabel("Adaptation \nIndex")
pvalues_AI = []
for pair in pairs1:
    p = posthoc_mod_AI.loc[pair[0], pair[1]]
    pvalues_AI.append(p)
print("pvalues:", pvalues_AI)
annotator = Annotator(ax_VGS_AI, pairs1, data=VGS_sum_AI)
annotator.configure(test=None, test_short_name="", loc='inside',
                    fontsize=8, show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.set_pvalues(pvalues=pvalues_AI)
annotator.annotate()
ax_VGS_AI.set_ylim(0, 1.25)
ax_VGS_AI.set_yticks([0, 0.5, 1])
ax_VGS_AI.set_xticklabels(step_labels)


# hide y-axis for stats region
ystart = 1.05
yend = 1.150
ax_VGS_AI.add_patch(patches.Rectangle((-0.6, ystart), 0.2, yend, fill=True, color="white",
                                      transform=ax_VGS_AI.transData, clip_on=False, zorder=3))

##### tau and data #####
order_NS = ["NS", "Pop NS", "E", "E+Syn", "E+I"]
palette_NS = ['#2060A7', "#007030", c_E,c_Esyn, c_EI,]
sns.boxplot(data=df_sum_data.pivot(columns='condition', values='decay $\tau$'), ax=ax_data_comp,
            saturation=0.5, order=order_NS, palette=palette_NS,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_data.pivot(columns='condition', values='decay $\tau$'),
              ax=ax_data_comp, palette=palette_NS,  order=order_NS, s=ms)
ax_data_comp.set_ylabel("E-SFA $\\tau$ (ms)")
ax_data_comp.set_xlabel("")
ax_data_comp.set_xticks(ax_data_comp.get_xticks())
ax_data_comp.set_xticklabels(
    ax_data_comp.get_xticklabels(), rotation=0, ha='center')

ax_data_comp.set_yticks([1, 2, 3])
ymin, ymax = ax_data_comp.get_ylim()
tick_range = np.arange(np.ceil(ymin), np.ceil(ymax)+1, 1)
ax_data_comp.yaxis.set_ticks([np.log10(x) for p in [0,1,2,3] for x in np.linspace(
    10 ** p, 10 ** (p + 1), 10)], minor=True)

ax_data_comp.yaxis.set_major_formatter(ticker.FormatStrFormatter("10$^{%d}$"))
# pairs2 = [(order_NS[0], order_NS[2]), (order_NS[0], order_NS[3]),
#           (order_NS[1], order_NS[2]), (order_NS[1], order_NS[3]),
#           (order_NS[0], order_NS[4]), (order_NS[1], order_NS[4]),]

pairs2 = [(order_NS[0], order_NS[2]), (order_NS[0], order_NS[3]),
          (order_NS[1], order_NS[2]), (order_NS[1], order_NS[3]),
        #   (order_NS[0], order_NS[4]), (order_NS[1], order_NS[4]),
          ]
pvalues = []
for pair in pairs2:
    p = posthoc_data.loc[pair[0], pair[1]]
    pvalues.append(p)
print("pvalues:", pvalues)
annotator = Annotator(ax_data_comp, pairs2, data=df_sum_data.pivot(
    columns='condition', values='decay $\tau$')[order_NS])
annotator.configure(test=None, test_short_name="", loc='inside',
                    fontsize=8, show_test_name=False, pvalue_thresholds=pval_thresh)
annotator.set_pvalues(pvalues=pvalues)
annotator.annotate()
ax_data_comp.set_ylim(ymin, 5.5)
ax_data_comp.set_xticklabels(data_labels)

# hide y-axis for stats region
ystart = 3.4
yend = 4 #5.150
ax_data_comp.add_patch(patches.Rectangle((-0.6, ystart), 0.2, yend, fill=True, color="white",
                                         transform=ax_data_comp.transData, clip_on=False, zorder=3))


##### AI and data #####
sns.boxplot(data=df_sum_data_AI.pivot(columns='condition', values='AI'), ax=ax_data_AI_comp,
            saturation=0.5, order=order_NS, palette=palette_NS,  boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(data=df_sum_data_AI.pivot(columns='condition', values='AI'),
              ax=ax_data_AI_comp, palette=palette_NS,  order=order_NS, s=ms)
ax_data_AI_comp.set_ylabel("Adaptation Index")
ax_data_AI_comp.set_xlabel("")
ax_data_AI_comp.set_xticklabels(data_labels)


######### SCHEMATIC #########
t = np.linspace(0, 500)
tau_fast = 10
tau_med = 100
tau_slow = 2000
tau1 = 20
tau2 = 120


# plot I-SFA
ax_I_fast.plot(t, np.exp(-t/tau_fast),
               color=lighten_color(c_step, amount=1.05))
ax_I_fast.text(100, 0.5, '  Fast', color=lighten_color(c_step, amount=1.05))
ax_I_fast.set_ylabel('I-SFA', color=c_step)
ax_I_med.plot(t, np.exp(-t/tau_med), color=lighten_color(c_step, amount=0.9))
ax_I_med.text(120, 0.4, '  Inter-\nmediate',
              color=lighten_color(c_step, amount=0.9))
ax_I_slow.plot(t,  np.exp(-t/tau_slow),
               color=lighten_color(c_step, amount=0.75))
ax_I_slow.text(50, 0.5, '  Slow', color=lighten_color(c_step, amount=0.75))


# plot E-SFA
tau1_fast = 20
tau2_fast = 100
y_fast = 0.55*(1 - np.exp(-(t) / tau1_fast)) * (np.exp(-(t) / tau2_fast))
y_fast_ind = np.argmax(y_fast)
y_fast_max = np.max(y_fast)
ax_I_fast_out.plot(t, y_fast, 
                   color=lighten_color(c_EI, amount=0.75))
ax_I_fast_out.set_ylabel('E-SFA', color=c_EI)
ax_I_fast_out.xaxis.set_label_position('top')

tau1_med = 20
tau2_med = 20
y_med = 2.625*(1 - np.exp(-(t) / tau1_med)) * (np.exp(-(t) / tau2_med))
y_med_ind = np.argmax(y_med)
y_med_max = np.max(y_med)
ax_I_med_out.plot(t, y_med, color=lighten_color(c_EI, amount=1))

tau1_slow = 20
tau2_slow = 125
y_slow = 1.3 * (1 - np.exp(-(t) / tau1_slow)) * (np.exp(-(t) / tau2_slow))
y_slow_ind = np.argmax(y_slow)
y_slow_max = np.max(y_slow)
ax_I_slow_out.plot(t, y_slow, color=lighten_color(c_EI, amount=1.25))


# add lines
xyAtop1 = [t[y_fast_ind], 0.]
xyBtop1 = [t[y_fast_ind], y_fast_max]
arrow_top1 = patches.ConnectionPatch(xyAtop1, xyBtop1, coordsA=ax_I_fast.transData, coordsB=ax_I_fast_out.transData,
                                     color=lighten_color(c_EI, amount=0.75), alpha=0.75,  linestyle=':', mutation_scale=0, linewidth=1)
fig.patches.append(arrow_top1)

xyAtop2 = [t[y_med_ind+1], 0.]
xyBtop2 = [t[y_med_ind+1], y_med_max]
arrow_top2 = patches.ConnectionPatch(xyAtop2, xyBtop2, coordsA=ax_I_med.transData, coordsB=ax_I_med_out.transData,
                                     color=lighten_color(c_EI, amount=1), alpha=0.75,  linestyle=':', mutation_scale=0, linewidth=1)
fig.patches.append(arrow_top2)

xyAtop3 = [t[y_slow_ind], 0.]
xyBtop3 = [t[y_slow_ind], y_slow_max]
arrow_top3 = patches.ConnectionPatch(xyAtop3, xyBtop3, coordsA=ax_I_slow.transData, coordsB=ax_I_slow_out.transData,
                                     color=lighten_color(c_EI, amount=1.25),  alpha=0.75, linestyle=':', mutation_scale=0, linewidth=1)
fig.patches.append(arrow_top3)


#####  remove axis side, add axis limits and labels #####
for ax in [ax_VGS_BS,ax_VGS_BS_syn, ax_VGS_BS_I]:
    # ax.set_xlabel("Time (ms)")
    # ax.set_ylabel("Avg Firing (Hz)")
    ax.set_xlim(psth_xlim)
    ax.set_ylim(psth_ylim)
ax_VGS_BS_syn.set_ylabel("Avg Firing (Hz)")
ax_VGS_BS_I.set_xlabel("Time (ms)")
for ax in [ax_VGS_BS,ax_VGS_BS_syn]:
    ax = remove_axis_box(ax, ["right", "top", "bottom"])
for ax in [ax_dia, ax_dia2, ax_VGS_BS_I, ax_VGS_sum, ax_tau_comp, ax_VGS_AI, ax_amp_comp, ax_data_comp, ax_data_AI_comp]:
    ax = remove_axis_box(ax, ["right", "top"])

for ax in [ax_I_fast, ax_I_med, ax_I_slow]:
    ax.set_xlim(-7.5, 400)
    ax.set_ylim(-0.05, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


for ax in [ax_I_fast_out, ax_I_med_out, ax_I_slow_out]:
    ax.set_xlim(-7.5, 400)
    ax.set_ylim(-0.05, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# remove spines
axHisty.spines["right"].set_visible(False)
axHisty.spines["top"].set_visible(False)
axHisty.spines["bottom"].set_visible(False)
axHistx.spines["right"].set_visible(False)
axHistx.spines["top"].set_visible(False)
axHistx.spines["left"].set_visible(False)
ax_tau_comp.spines["left"].set_visible(False)
ax_tau_comp.spines["top"].set_visible(False)
ax_tau_comp.spines["right"].set_visible(False)
ax_tau_comp.spines["bottom"].set_visible(False)
ax_tau_comp.set_ylim(45, 130)
ax_tau_comp.set_xlim(0.5, 5.)
axHistx_max.spines["right"].set_visible(False)
axHistx_max.spines["top"].set_visible(False)
axHistx_max.spines["left"].set_visible(False)
ax_amp_comp.spines["top"].set_visible(False)
ax_amp_comp.spines["right"].set_visible(False)
ax_amp_comp.spines["bottom"].set_visible(False)
ax_amp_comp.set_ylim(20, 60)
ax_amp_comp.set_xlim(0.5, 5.)
ax_VGS_BS.set_yticks([0, 20, 40])
ax_VGS_BS_I.set_yticks([0, 20, 40])
ax_VGS_BS_syn.set_yticks([0, 20, 40])


##### subplot letters #####
ax_dia.text(-0.05, 1.0, string.ascii_uppercase[0], transform=ax_dia.transAxes, size=12, weight='bold')
ax_dia1.text(-0.05, 1.1, string.ascii_uppercase[1],transform=ax_dia1.transAxes, size=12, weight='bold')
ax_dia2.text(-0.05, 1.1, string.ascii_uppercase[2],transform=ax_dia2.transAxes, size=12, weight='bold')
ax_VGS_sum.text(-0.05, 1.0, string.ascii_uppercase[3], transform=ax_VGS_sum.transAxes, size=12, weight='bold')
ax_VGS_AI.text(-0.05, 1.0, string.ascii_uppercase[4],transform=ax_VGS_AI.transAxes, size=12, weight='bold')
ax_data_comp.text(-0.025, 1.1, string.ascii_uppercase[5], transform=ax_data_comp.transAxes, size=12, weight='bold')
ax_data_AI_comp.text(-0.025, 1.1, string.ascii_uppercase[6], transform=ax_data_AI_comp.transAxes, size=12, weight='bold')
ax_tau_comp.text(-0.19, 1.175, string.ascii_uppercase[7], transform=ax_tau_comp.transAxes, size=12, weight='bold')
ax_amp_comp.text(-0.025, 1.2, string.ascii_uppercase[8], transform=ax_amp_comp.transAxes, size=12, weight='bold')
ax_I_fast_out.text(-0.25, 1.075, string.ascii_uppercase[9], transform=ax_I_fast_out.transAxes, size=12, weight='bold')

##### save and show #####

plt.savefig('./figures/Figure_6_Lowpass_syn_FFI.png', dpi=300)
plt.savefig('./figures/Figure_6_Lowpass_syn_FFI.pdf', dpi=300)
plt.show()

# %%

# %%

# %%
