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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plt_fxns import lighten_color, remove_axis_box, circuit_dia_LP
plt.rcParams["font.family"] = "Arial"
# %% load data
I = 0.74
g = 5.6
gi = 2.45
VGS_sum_AI = pd.read_csv(
    "./data/sims/Pospischil_TAU_filter_Inh_sim_AI.csv".format(I, g, gi))
VGS_sum_AI.rename(columns={"AI_step": "Step",
                  "AI_VGS": "E", "AI_VGS_I": "E+I"}, inplace=True)


# %% Create Figure

np.random.seed(0)  # set seed for stripplot jitter
pval_thresh = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

##### plot style choice #####
c_step = lighten_color('#6D398B', amount=1.25)
c_E = 'tab:grey'
c_EI = '#C02717'
step_labels = ['ISFA', 'E', 'E+I']
data_labels = ['NS', 'Pop NS', 'E', 'E+I']
alpha = 0.5
lw = 2
stim_start = 750.
ms = 3
size = 20
lw_m = 2

##### layout #####
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

##### Step AI vs E-SFA AI #####
x_AI = VGS_sum_AI['Step']
y_AI = VGS_sum_AI['E+I']

ax.scatter(x_AI, y_AI, c=c_EI, edgecolors=c_step, linewidths=lw_m, s=size,)
divider = make_axes_locatable(ax)
axHistx_AI = divider.append_axes("bottom", 0.2, pad=0.1, sharex=ax)
axHisty_AI = divider.append_axes("left", 0.2, pad=0.1, sharey=ax)


# make some labels invisible
ax.tick_params(labelbottom=False, bottom=False,
               left=False, labelleft=False, right=False, labelright=False)
axHistx_AI.tick_params(labelbottom=True, bottom=True,
                       left=False, labelleft=False, right=False, labelright=False)
axHisty_AI.tick_params(labelleft=True, left=True,
                       bottom=False, labelbottom=True, top=False, labeltop=True)

sns.boxplot(x_AI, ax=axHistx_AI, color=c_step, saturation=0.5,
            boxprops=dict(alpha=.5), showfliers=False)
sns.stripplot(x_AI, ax=axHistx_AI, color='w', edgecolor=c_step,
              size=np.sqrt(size), linewidth=lw_m, )
sns.boxplot(y=y_AI, ax=axHisty_AI, color=c_EI, saturation=0.5,
            boxprops=dict(alpha=.5), orient='x', showfliers=False)
sns.stripplot(y=y_AI, ax=axHisty_AI, color=c_EI,
              orient='v', size=np.sqrt(size)+0.5, alpha=1.)


# axes labels
axHistx_AI.set_xlabel('Intrinsic Adaptation Index')
axHisty_AI.set_ylabel('Embedded Adaptation Index')

# lines

# horizontal
ind_min_y = np.argmin(x_AI)
xy1 = (x_AI[ind_min_y], y_AI[ind_min_y])
xy2 = (0., y_AI[ind_min_y])
print(xy1)
print(xy2)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                      axesA=ax, axesB=axHisty_AI, color=c_EI, linestyle=":")
axHisty_AI.add_artist(con)

# vertical
xy1 = (x_AI[ind_min_y], y_AI[ind_min_y])
xy2 = (x_AI[ind_min_y], 0.)
print(xy1)
print(xy2)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                      axesA=ax, axesB=axHistx_AI, color=c_step, linestyle=":")
axHistx_AI.add_artist(con)


##### remove spines #####
axHisty_AI.spines["right"].set_visible(False)
axHisty_AI.spines["top"].set_visible(False)
axHisty_AI.spines["bottom"].set_visible(False)
axHistx_AI.spines["right"].set_visible(False)
axHistx_AI.spines["top"].set_visible(False)
axHistx_AI.spines["left"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_ylim(0, 0.4)
ax.set_xlim(0.4, 1.05)

##### save and show #####
plt.savefig('./figures/Supp_Figure_5_Lowpass_FFI_step_AI.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./figures/Supp_Figure_5_Lowpass_FFI_step_AI.pdf',
            bbox_inches='tight', dpi=300)
plt.show()
