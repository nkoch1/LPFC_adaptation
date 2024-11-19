import numpy as np
import matplotlib.patches as patches
import pandas as pd


def lighten_color(color, amount=0.5):
    """
    from https://stackoverflow.com/a/49601444
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    Args:
        color: color to lighten
        amount: amount to lighten
    Returns:
        lightened color

    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def remove_axis_box(ax, s=["top", "right"]):
    """ remove the frame box sides around a plt subplot

    Args:
        ax: subplot axes
        s: sides of the axes to remove - "top", "right", "bottom", "left"

    Returns:
        ax: subplot axes with sides removed

    """
    for i in s:
        ax.spines[i].set_visible(False)
        if i == "bottom":
            ax.set_xticks([])
        elif i == "left":
            ax.set_yticks([])
    return ax


def generate_spike_train(numtrials, seed, F0, F1, tend, tstim, stim_num_spikes):
    """ generates simulated spike trains

    Args:
        numtrials: number of trials to simulate
        seed: seed for np.random
        F0: baseline firing frequency
        F1: firing frequency during stimulation
        tend: time at end of trial
        tstim: time at stimulation
        stim_num_spikes: number of spikes to simulate per trial

    Returns:
        simulated spike trains for n trials

    """
    cell_spikes = []
    np.random.seed(seed)
    for i in range(0, numtrials+1):
        j = 1
        isi = np.random.poisson(F0, 1)
        t = isi[0]
        while t <= tend:
            if t <= tstim or j >= stim_num_spikes:
                isi_i = np.random.poisson(F0, 1)
            else:
                isi_i = np.random.poisson(F1, 1)
                j += 1
            isi = np.concatenate([isi, isi_i])
            t = np.cumsum(isi)[-1]
        cell_spikes.append(np.cumsum(isi))
    return cell_spikes


def exp_curve(t, a, b, tau):
    """ Exponential decay curve

    Args:
        t: time series array
        a: scaling of exp decay
        b: offset of exp decay
        tau: time constant of exp decay 
    Returns:
        exponential decay curve 

    """
    return a*np.exp(t*tau) + b

def circuit_dia_LP(ax, inhib=True, ylim=(0, 2.5), lw_c=4, lw_con=3,  c_E="tab:green", c_M="tab:blue", c_I="tab:grey", lw_sdf=0.5, lw_f=2, fs_labels=10, fs_eqn=8, fs_axis=5):
    """ Low-Pass circuit diagram
    Args:
        ax: axes to plot diagram in 
        inhib: plot inhibition (Bool)
        ylim: axis ylim
        lw_c: FFI line width
        lw_con: connection line width
        c_E: color for exictatory input
        c_M: color for mixed input
        c_I: color for inhibition
        lw_sdf: SDF line width
        lw_f: filter line width
        fs_labels: label font size
        fs_eqn: equation font size
        fs_axis: aixe font size
    Returns:
        ax with diagram
    """
    E_rect = patches.Rectangle((0, 1.25), 1.5, 1, linewidth=lw_c, clip_on=False,
                               edgecolor=c_E, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(E_rect)

    M_rect = patches.Rectangle((2.75, 1.25), 1.5, 1, linewidth=lw_c, clip_on=False,
                               edgecolor=c_M, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(M_rect)

    # add excitatory connection
    ax.arrow(1.5, 1.75, 0.9, 0., linewidth=lw_con, color=c_E,
             linestyle="-", head_width=0.15, head_length=-0.2)

    # plot E SDF
    ax_E_in = ax.inset_axes([0.25, 1.325, 1, 1.0],
                            transform=ax.transData)  # x, y, width, height
    ax_E_in.set_title("SDF", fontsize=fs_axis, pad=0, color=c_E, weight="bold")
    VGS = pd.read_csv("./data/exp/VGS_BS_SDF_groups_sel_n_70.csv")
    sat_VGS = np.linspace(0.25, 1.35, VGS.shape[1])
    sdf_t = np.arange(VGS.shape[0])
    for i in range(0, VGS.shape[1]):
        ax_E_in.plot(sdf_t, VGS.iloc[:, i], linewidth=lw_sdf, alpha=0.25, color=lighten_color(
            c_E, sat_VGS[i]))  # color=c_E) #,
    ax_E_in.set_xlim(0, 250000)
    ax_E_in.spines["right"].set_visible(False)
    ax_E_in.spines["top"].set_visible(False)
    ax_E_in.spines["left"].set_visible(False)
    ax_E_in.spines["bottom"].set_visible(False)
    ax_E_in.set_xticks([])
    ax_E_in.set_yticks([])

    # add text
    ax.text(3.5, 2.64, "NS Model", fontsize=fs_labels,
            color=c_M, ha="center", va="center", weight="bold")
    ax.text(3.5, 1.75, "$\sum$ I", fontsize=fs_eqn, color=c_M,
            weight="bold", ha="center", va="center")

    if inhib:
        ax.arrow(2, 1.75, 0, -0.5, linewidth=lw_con, color=c_E,
                 linestyle="-", head_width=0.225, head_length=-0.15)

        I_rect = patches.Rectangle((1.25, 0), 1.5, 1, linewidth=lw_c, clip_on=False,
                                   edgecolor=c_I, facecolor='none', linestyle="-", zorder=10)
        ax.add_patch(I_rect)

        ax.arrow(2.75, 0.5, 0.75, 0, linewidth=lw_con, color=c_I,
                 linestyle="-", head_width=0., head_length=0)
        l1 = patches.FancyArrowPatch(posA=(3.5, 1.1), posB=(3.5, 0.45),
                                     arrowstyle=patches.ArrowStyle(
                                         ']-', widthA=2.0, lengthA=0., angleA=None),
                                     linewidth=lw_con,   color=c_I, linestyle="-", zorder=10, mutation_scale=3,)
        ax.add_patch(l1)
        # plot filter
        ax_I_in = ax.inset_axes([1.5, 0.25, 1.0, 0.5], transform=ax.transData)
        x_f = np.linspace(-10, 10, 1000)
        y_f = 1/(1 + np.exp(x_f))
        ax_I_in.plot(x_f, y_f, color=c_I, lw=lw_f)
        ax_I_in.set_xlim(-5.5, 1.2)
        ax_I_in.set_ylim(0.225, 1.1)
        ax_I_in.spines["right"].set_visible(False)
        ax_I_in.spines["top"].set_visible(False)
        ax_I_in.set_xticks([])
        ax_I_in.set_yticks([])

    # set limits and aspect ratio
    ax.set_xlim(0, 4.5)
    ax.set_ylim(ylim)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def circuit_dia_HP(ax, inhib=True, ylim=(0, 2.5), lw_c=4, lw_con=3,  c_E="tab:green", c_M="tab:blue", c_I="tab:grey", lw_sdf=0.5, lw_f=2, fs_labels=10, fs_eqn=8, fs_axis=5):
    """ High-Pass circuit diagram
    Args:
        ax: axes to plot diagram in 
        inhib: plot inhibition (Bool)
        ylim: axis ylim
        lw_c: FFI line width
        lw_con: connection line width
        c_E: color for exictatory input
        c_M: color for mixed input
        c_I: color for inhibition
        lw_sdf: SDF line width
        lw_f: filter line width
        fs_labels: label font size
        fs_eqn: equation font size
        fs_axis: aixe font size
    Returns:
        ax with diagram
    """
    E_rect = patches.Rectangle((0, 1.25), 1.5, 1, linewidth=lw_c, clip_on=False,
                               edgecolor=c_E, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(E_rect)

    M_rect = patches.Rectangle((2.75, 1.25), 1.5, 1, linewidth=lw_c, clip_on=False,
                               edgecolor=c_M, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(M_rect)
    ax.add_patch(M_rect)

    # add excitatory connection
    ax.arrow(1.5, 1.75, 0.9, 0., linewidth=lw_con, color=c_E,
             linestyle="-", head_width=0.15, head_length=-0.2)

    # plot E SDF
    ax_E_in = ax.inset_axes([0.25, 1.325, 1, 0.5],
                            transform=ax.transData)  # x, y, width, height
    ax_E_in.set_title("SDF", fontsize=fs_axis, pad=-
                      4, color=c_E, weight="bold")
    VGS = pd.read_csv("./data/exp/VGS_BS_SDF_groups_sel_n_70.csv")
    sat_VGS = np.linspace(0.25, 1.35, VGS.shape[1])
    sdf_t = np.arange(VGS.shape[0])
    for i in range(0, VGS.shape[1]):
        ax_E_in.plot(sdf_t, VGS.iloc[:, i], linewidth=lw_sdf, alpha=0.25, color=lighten_color(
            c_E, sat_VGS[i]))  # color=c_E) #,
    ax_E_in.set_xlim(0, 250000)
    ax_E_in.spines["right"].set_visible(False)
    ax_E_in.spines["top"].set_visible(False)
    ax_E_in.spines["left"].set_visible(False)
    ax_E_in.spines["bottom"].set_visible(False)
    ax_E_in.set_xticks([])
    ax_E_in.set_yticks([])

    # add text
    ax.text(3.5, 2.77, "NS Model", fontsize=fs_labels,
            color=c_M, ha="center", va="center", weight="bold")
    ax.text(3.5, 1.75, "$\sum$ I", fontsize=fs_eqn, color=c_M,
            weight="bold", ha="center", va="center")

    if inhib:
        ax.arrow(2, 1.75, 0, -0.575, linewidth=lw_con, color=c_E,
                 linestyle="-", head_width=0.225, head_length=-0.15)

        I_rect = patches.Rectangle((1.25, 0), 1.5, 1, linewidth=lw_c, clip_on=False,
                                   edgecolor=c_I, facecolor='none', linestyle="-", zorder=10)
        ax.add_patch(I_rect)

        ax.arrow(2.75, 0.5, 0.775, 0, linewidth=lw_con-1, color=c_I,
                 linestyle="-", head_width=0., head_length=0)

        l1 = patches.FancyArrowPatch(posA=(3.5, 1.1), posB=(3.5, 0.45),
                                     arrowstyle=patches.ArrowStyle(
                                         ']-', widthA=2.0, lengthA=0., angleA=None),
                                     linewidth=lw_con,   color=c_I, linestyle="-", zorder=10, mutation_scale=3,)
        ax.add_patch(l1)

        # plot filter
        ax_I_in = ax.inset_axes([1.5, 0.25, 1.0, 0.5], transform=ax.transData)
        x_f = np.linspace(-10, 10, 1000)
        y_f = (1/(1+np.exp(-(x_f - 2)/0.2))) * (1/(1+np.exp((x_f - 8)/0.5)))

        ax_I_in.plot(x_f, y_f, color=c_I, lw=lw_f)
        ax_I_in.set_xlim(0., 10)
        ax_I_in.set_ylim(0.225, 1.1)
        ax_I_in.spines["right"].set_visible(False)
        ax_I_in.spines["top"].set_visible(False)
        ax_I_in.set_xticks([])
        ax_I_in.set_yticks([])

    # set limits and aspect ratio
    ax.set_xlim(0, 4.5)
    ax.set_ylim(ylim)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def circuit_dia_OU(ax, ylim=(0, 2.5), lw_c=4, lw_con=3,  c_E="tab:green", c_M="tab:blue", c_gE="tab:grey", c_gI="tab:purple",  lw_sdf=0.5, lw_f=2, fs_labels=10, fs_labels2=10, fs_eqn=8, fs_axis=5):
    """ Synaptic Bombardment circuit diagram
    Args:
        ax: axes to plot diagram in 
        ylim: axis ylim
        lw_c: FFI line width
        lw_con: connection line width
        c_E: color for exictatory input
        c_M: color for mixed input
        c_I: color for inhibition
        lw_sdf: SDF line width
        lw_f: filter line width
        fs_labels: label font size
        fs_eqn: equation font size
        fs_axis: aixe font size
    Returns:
        ax with diagram
    """
    E_rect = patches.Rectangle((0, 1.25), 1.5, 1, linewidth=lw_c, clip_on=False,
                               edgecolor=c_E, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(E_rect)

    M_rect = patches.Rectangle((2.75, 1.25), 1.5, 1, linewidth=lw_c, clip_on=False,
                               edgecolor=c_M, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(M_rect)

    # add excitatory connection
    ax.arrow(1.5, 1.75, 0.9, 0., linewidth=lw_con, color=c_E,
             linestyle="-", head_width=0.15, head_length=-0.2)

    # plot E SDF
    ax_E_in = ax.inset_axes([0.25, 1.325, 1, 1.0],
                            transform=ax.transData)  # x, y, width, height
    ax_E_in.set_title("SDF", fontsize=fs_axis, pad=0, color=c_E, weight="bold")
    VGS = pd.read_csv("./data/exp/VGS_BS_SDF_groups_sel_n_70.csv")
    sat_VGS = np.linspace(0.25, 1.35, VGS.shape[1])
    sdf_t = np.arange(VGS.shape[0])
    for i in range(0, VGS.shape[1]):
        ax_E_in.plot(sdf_t, VGS.iloc[:, i], linewidth=lw_sdf, alpha=0.25, color=lighten_color(
            c_E, sat_VGS[i]))  # color=c_E) #,
    ax_E_in.set_xlim(0, 250000)
    ax_E_in.spines["right"].set_visible(False)
    ax_E_in.spines["top"].set_visible(False)
    ax_E_in.spines["left"].set_visible(False)
    ax_E_in.spines["bottom"].set_visible(False)
    ax_E_in.set_xticks([])
    ax_E_in.set_yticks([])

    # add text
    ax.text(3.5, 2.6, "NS Model", fontsize=fs_labels,
            color=c_M, ha="center", va="center", weight="bold")
    ax.text(3.5, 1.75, "$\sum$ I", fontsize=fs_eqn, color=c_M,
            weight="bold", ha="center", va="center")

    I_circ = patches.Ellipse((1.5, 0.25), 1.5,  0.8,  linewidth=lw_c, clip_on=False,
                             edgecolor=c_gE, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(I_circ)
    ax.arrow(1.9, 0.6, 0.5, 0.5, linewidth=lw_con, color=c_gE,
             linestyle="-", head_width=0.15, head_length=-0.2)

    I_circ2 = patches.Ellipse((3.5, 0.25), 1.5,  0.8,  linewidth=lw_c, clip_on=False,
                              edgecolor=c_gI, facecolor='none', linestyle="-", zorder=10)
    ax.add_patch(I_circ2)

    # add inhibitory connection
    l1_I_M = patches.Arrow(3.5, 0.675, 0., 0.35, linewidth=lw_con,
                           color=c_gI, linestyle="-", zorder=10, width=0)
    ax.add_patch(l1_I_M)
    l1_I_M_cap = patches.Arrow(
        3.3, 1.05, 0.4, 0., linewidth=lw_con, color=c_gI, linestyle="-", zorder=10, width=0)
    ax.add_patch(l1_I_M_cap)

    # text for ge and gi
    ax.annotate("g$_e$(t)", (1.05, 0.15), color=c_gE, fontsize=fs_labels2)
    ax.annotate("g$_i$(t)", (3.125, 0.15), color=c_gI, fontsize=fs_labels2)

    # set limits and aspect ratio
    ax.set_xlim(0, 4.5)
    ax.set_ylim(ylim)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax