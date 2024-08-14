# %%
import os
os.chdir('../../')
from src.plt_fxns import remove_axis_box
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import matplotlib.patches as patches
import matplotlib.image as mpimg
import scipy
plt.rcParams["font.family"] = "Arial"

# %% Create Figure

##### figure layout #####
fig = plt.figure(figsize=(6.5, 6))
gs = fig.add_gridspec(5, 2, width_ratios=[4, 1.25], height_ratios=[
                      1., 1.1, 0.65, 0.8, 0.7],  hspace=0.2,  wspace=0.3)
ax_p = fig.add_subplot(gs[:, 0])
ax_vivo = fig.add_subplot(gs[1, 1])
gs_vitro = gridspec.GridSpecFromSubplotSpec(
    2, 1, height_ratios=[4, 0.5], wspace=0.35, hspace=0.1, subplot_spec=gs[3, 1])
ax_vitro_NS = fig.add_subplot(gs_vitro[0])
ax_vitro_step = fig.add_subplot(gs_vitro[1])

###### add image #####
img = mpimg.imread('./figures/src/schematic.png')
ax_p.imshow(img)
ax_p = remove_axis_box(ax_p, ["top", "left", "right", "bottom"])
ax_p.text(950, -50, 'in vivo')
ax_p.text(950, 1075, 'in vitro')

###### add arrows #####
xyA = [285, 750]
xyB = [475, 500]
arrow0 = patches.ConnectionPatch(xyA, xyB, coordsA=ax_p.transData, coordsB=ax_p.transData,
                                 color='k', arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow0)

xyA2 = [285, 950]
xyB2 = [475, 1150]
arrow1 = patches.ConnectionPatch(xyA2, xyB2, coordsA=ax_p.transData, coordsB=ax_p.transData,
                                 color='k', arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow1)

xyAtop = [900, 450]
xyBtop = [1075, 450]
arrow_top = patches.ConnectionPatch(xyAtop, xyBtop, coordsA=ax_p.transData, coordsB=ax_p.transData,
                                    color='k', arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow_top)

xyAtop2 = [1775, 450]
xyBtop2 = [2000, 450]
arrow_top2 = patches.ConnectionPatch(xyAtop2, xyBtop2, coordsA=ax_p.transData, coordsB=ax_p.transData,
                                     color='k', arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow_top2)

xyAbottom = [900, 1325]
xyBbottom = [1075, 1325]
arrow_bottom = patches.ConnectionPatch(xyAbottom, xyBbottom, coordsA=ax_p.transData, coordsB=ax_p.transData,
                                       color='k', arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow_bottom)

xyAbottom2 = [1775, 1325]
xyBbottom2 = [2000, 1325
              ]
arrow_bottom2 = patches.ConnectionPatch(xyAbottom2, xyBbottom2, coordsA=ax_p.transData, coordsB=ax_p.transData,
                                        color='k', arrowstyle="-|>", mutation_scale=10, linewidth=2,)
fig.patches.append(arrow_bottom2)


###### in vivo example raster #####
container = sio.loadmat('./data/exp//VGS_NS_ex_spike_times.mat')
VGS_NS_spikes = []
for i in range(container['VGS_NS_struct'].shape[0]):
    VGS_NS_spikes.append(container['VGS_NS_struct'][i][0][:, 0] - 750)
ax_vivo.eventplot(VGS_NS_spikes, color='k', linewidths=1)
ax_vivo.set_xlabel('Time (ms)')
ax_vivo = remove_axis_box(ax_vivo, ["top", "left", "right"])
ax_vivo.set_xlim(-100, 1000)

###### example in vitro trace #####

# load data and find step start and end
plot_dt = 0.005
plot_dt_a = 0.01
min_spike_height = 10.  # mV
prominence = 10.  # mV
distance = 0.001  # s
FS_ex_cell = 'M10_JS_A1_C22'
df_patch = pd.read_csv('./data/exp/ICTable.csv')
FS_ex_df = pd.read_json(
    './data/exp_pro/dlPFC_FS/{}_AP_phase_analysis.json'.format(FS_ex_cell))
FS_stimVal = df_patch.loc[df_patch.index[df_patch.cells ==
                                         FS_ex_cell], 'stimVal'].values[0]
FS_ex_ind = FS_ex_df.index[FS_ex_df['I mag'] == FS_stimVal]
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

# plot in vitro trace
patch_xlim = (-25, 1020)
ax_vitro_NS.plot((np.array(FS_ex_df.loc[FS_ex_ind, 't'].values[0]) - FS_ex_df.loc[FS_ex_ind, 't'].values[0][0])*1000,
                 FS_ex_df.loc[FS_ex_ind, 'V'].values[0], color='k', linewidth=0.5)
ax_vitro_NS.set_xlim(patch_xlim)
ax_vitro_NS = remove_axis_box(ax_vitro_NS, ["top", "left", "right", "bottom"])

# add step current
x = [-100, 0, 1000, 1000, 1100]
y = [0., FS_stimVal,  FS_stimVal, 0., 0.]
ax_vitro_step.step(x, y, color='k', where='post')
ax_vitro_step.set_xlim(patch_xlim)
ax_vitro_step.set_ylim(-25, 100)

ax_vitro_step = remove_axis_box(ax_vitro_step, ["top", "left", "bottom", "right"])
ax_vitro_step.set_xlabel('Time (ms)')

##### save and show #####
plt.savefig('./figures/Figure_1_schematic.png', bbox_inches='tight', dpi=300)
plt.savefig('./figures/Figure_1_schematic.pdf', bbox_inches='tight', dpi=300)
plt.show()
