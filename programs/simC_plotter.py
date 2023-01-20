# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["font.size"] = 13
plt.rcParams['axes.linewidth'] = 1.0

POPS = ["phi0"]
POP_LABELS = [r"$\phi_0$"]
POP_I = 0
pop_name = POPS[POP_I]
output_dir = os.path.join("outputs", "simC", pop_name)
os.makedirs(output_dir, exist_ok=True)
POP_VALS = np.round(np.array([.2, .5, .8]), 1)
N_MODELS = ['phi', 'DP', 'DFH', 'pARIs']
NW_MODELS = ['DFH', 'pARIs']
N_MODELS_LABEL = [r'$\phi\,(N)$', r'$\Delta P\,(N)$', r'$DFH\,(N)$', r'$pARIs\,(N)$']
NW_MODELS_LABEL = [r'$DFH\,(N_W)$', r'$pARIs\,(N_W)$']
SAMPLE_SIZE = 30
NW_FLAGS = [False, True]


def read_csv_files(pop, is_nw):
    PATH_DATA = os.path.expanduser(output_dir)
    mean_dir_path = os.path.join(PATH_DATA, 'data_mean')
    sd_dir_path = os.path.join(PATH_DATA, 'data_sd')
    nw_label = "NW" if is_nw else "N"
    MODELS = NW_MODELS if is_nw else N_MODELS

    # read csv
    df_mean = pd.read_csv(
        os.path.join(mean_dir_path, nw_label + '_mean_' + pop_name + "_" + str(round(pop, 1)) + '.csv'),
        nrows=SAMPLE_SIZE).astype(np.float64)
    df_sd = pd.read_csv(
        os.path.join(sd_dir_path, nw_label + '_sd_' + pop_name + "_" + str(round(pop, 1)) + '.csv'),
        nrows=SAMPLE_SIZE).astype(np.float64)

    # Trimming spaces
    df_mean.columns = df_mean.columns.map(str.strip)
    df_sd.columns = df_sd.columns.map(str.strip)
    # Extract only the target models
    df_mean = df_mean.loc[:, MODELS]
    df_sd = df_sd.loc[:, MODELS]

    return df_mean, df_sd, MODELS
# %%


markers = ["o", "^", "s", "D", "p"]
colors = ['orange', 'red', 'green', 'blue', 'magenta']
qty_n = len(N_MODELS)
qty_nw = len(NW_MODELS)
n_colors = colors[:qty_n]
nw_colors = colors[qty_n - qty_nw:qty_n]
n_styles = ["-"] * qty_n
nw_styles = ["--"] * qty_nw
indexer_x = np.arange(1, SAMPLE_SIZE + 0.01, 1)
indexer_y = np.arange(0, 1 + 0.01, .1)

fig = plt.figure(figsize=(7, 8))
axes = []
for i in range(2 * len(POP_VALS)):
    axes.append(fig.add_subplot(len(POP_VALS), 2, i + 1))
    axes[i].set_xlim(xmin=1, xmax=SAMPLE_SIZE)
    axes[i].set_ylim(ymin=0.0, ymax=1.0)

for nw_i, is_nw in enumerate(NW_FLAGS):
    for p_i, pop in enumerate(POP_VALS):
        df_mean, df_sd, _ = read_csv_files(pop, is_nw)
        df_mean.index = np.arange(1, SAMPLE_SIZE + 1, 1)
        df_sd.index = np.arange(1, SAMPLE_SIZE + 1, 1)
        mean_id = p_i * 2 + 0
        sd_id = p_i * 2 + 1
        df_mean.plot(ax=axes[mean_id], color=(nw_colors if is_nw else n_colors), style=(nw_styles if is_nw else n_styles), legend=False)
        df_sd.plot(ax=axes[sd_id], color=(nw_colors if is_nw else n_colors), style=(nw_styles if is_nw else n_styles), legend=False)
        if pop == POP_VALS[-1]:
            axes[mean_id].set_xlabel("Sample Size")
            axes[sd_id].set_xlabel("Sample Size")
        axes[mean_id].set_title(f"{POP_LABELS[POP_I]}$ = {pop}$")
        axes[sd_id].set_title(f"{POP_LABELS[POP_I]}$ = {pop}$")
        axes[mean_id].set_ylabel("Mean")
        axes[sd_id].set_ylabel("SD")
        axes[mean_id].grid()
        axes[sd_id].grid()

handles, labels = axes[-1].get_legend_handles_labels()
handles = [handles[i] for i in [4, 2, 5, 3, 0, 1]]
MODELS_LABEL_MERGE = [NW_MODELS_LABEL[0]] + [N_MODELS_LABEL[2]] + [NW_MODELS_LABEL[1]] + [N_MODELS_LABEL[3]] + N_MODELS_LABEL[0:2]
s = fig.subplotpars
bb = [s.left, s.top + 0.04, s.right - s.left, 0.05]
axes[0].legend(handles, MODELS_LABEL_MERGE, loc=8, bbox_to_anchor=bb, ncol=4, mode="expand", borderaxespad=0, bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "simC_timedev" + ".pdf"), bbox_inches="tight", pad_inches=0.05)
plt.savefig(os.path.join(output_dir, "simC_timedev" + ".png"), bbox_inches="tight", pad_inches=0.05)
plt.show()
plt.close()
# %%
