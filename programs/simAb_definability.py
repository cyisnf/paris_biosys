# encoding: utf-8
# %%
import numpy as np
import modules.ContingencyTable_pop_inc as ct
import modules.models as models
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["font.size"] = 13
plt.rcParams['axes.linewidth'] = 1.0

READ_PICKLE = False
NUM_TABLE = 100000
NUM_INS = 15
PROBS = np.round(np.array([.2, .8]), 1)
POP_VALUES = np.round(np.array([.2, .8]), 1)
POPS = [models.phi0]
POPS_LABEL = ["$\phi_0$"]
POP_I = 0
N_MODEL_LABEL = ['$\phi\,(N)$', '$\Delta P\,(N)$', '$DFH\,(N)$', '$pARIs\,(N)$']
NW_MODEL_LABEL = ['$DFH\,(N_W)$', '$pARIs\,(N_W)$']
pop_name = str(POPS[POP_I]).split(" ")[1]
output_dir = os.path.join("outputs", "simAb", pop_name)
os.makedirs(output_dir, exist_ok=True)

n_def_cnt = np.zeros([len(PROBS), len(POP_VALUES), len(N_MODEL_LABEL), NUM_INS])
n_def_rate = np.zeros([len(PROBS), len(POP_VALUES), len(N_MODEL_LABEL), NUM_INS])
nw_def_cnt = np.zeros([len(PROBS), len(POP_VALUES), len(N_MODEL_LABEL), NUM_INS])
nw_def_rate = np.zeros([len(PROBS), len(POP_VALUES), len(N_MODEL_LABEL), NUM_INS])
if not READ_PICKLE:
    for prob_i, prob_val in enumerate(PROBS):
        for pop_i, pop_val in tqdm(enumerate(POP_VALUES)):
            for tab_i in range(NUM_TABLE):
                table = ct.ContingencyTable(prob_val, prob_val, pop_name, pop_val, is_nw=False)
                for ins_i in range(NUM_INS):
                    table.sampling_inc()
                    if table.def_phi():
                        n_def_cnt[prob_i][pop_i][0][ins_i] += 1
                    if table.def_dp():
                        n_def_cnt[prob_i][pop_i][1][ins_i] += 1
                    if table.def_dfh():
                        n_def_cnt[prob_i][pop_i][2][ins_i] += 1
                    if table.def_prs():
                        n_def_cnt[prob_i][pop_i][3][ins_i] += 1

            for tab_i in range(NUM_TABLE):
                table = ct.ContingencyTable(prob_val, prob_val, pop_name, pop_val, is_nw=True)
                for ins_i in range(NUM_INS):
                    table.sampling_inc()
                    if table.def_dfh():
                        nw_def_cnt[prob_i][pop_i][0][ins_i] += 1
                    if table.def_prs():
                        nw_def_cnt[prob_i][pop_i][1][ins_i] += 1
        n_def_rate[prob_i] = n_def_cnt[prob_i] / NUM_TABLE
        nw_def_rate[prob_i] = nw_def_cnt[prob_i] / NUM_TABLE

    np.save(os.path.join(output_dir, f'simAb_n_def_rate_prob{prob_val}.npy'), n_def_rate)
    np.save(os.path.join(output_dir, f'simAb_nw_def_rate_prob{prob_val}.npy'), nw_def_rate)
else:
    for prob_i, prob_val in enumerate(PROBS):
        n_def_rate = np.load(os.path.join(output_dir, str(prob_val), f'simAb_n_def_rate_prob{prob_val}.npy'))
        nw_def_rate = np.load(os.path.join(output_dir, str(prob_val), f'simAb_nw_def_rate_prob{prob_val}.npy'))

# %%

# Plotting
markers = ["o", "^", "s", "D", "v"]
colors = ['orange', 'red', 'green', 'blue', 'magenta']

for prob_val_i, prob_val in enumerate(PROBS):
    fig = plt.figure(figsize=(3.14 * len(POP_VALUES), 3.64))
    for pop_val_i, phi_0 in enumerate(POP_VALUES):
        ax = fig.add_subplot(1, len(POP_VALUES), pop_val_i + 1)
        ax.grid()
        indexer_x = np.arange(0, NUM_INS + 1, 3)
        indexer_y = np.arange(0.0, 1.1, 0.2)
        for model_i in range(len(N_MODEL_LABEL)):
            plt.plot(
                np.arange(1, NUM_INS + 1),
                n_def_rate[prob_val_i][pop_val_i][model_i],
                label=N_MODEL_LABEL[model_i],
                marker=markers[model_i],
                markeredgewidth=1.5,
                mfc=colors[model_i],
                c=colors[model_i],
                mec=colors[model_i],
                alpha=.9
            )
        for model_i in range(len(NW_MODEL_LABEL)):
            plt.plot(
                np.arange(1, NUM_INS + 1),
                nw_def_rate[prob_val_i][pop_val_i][model_i],
                label=NW_MODEL_LABEL[model_i],
                marker=markers[model_i + (len(N_MODEL_LABEL) - len(NW_MODEL_LABEL))],
                markeredgewidth=1.5, mfc='white',
                c=colors[model_i + (len(N_MODEL_LABEL) - len(NW_MODEL_LABEL))],
                mec=colors[model_i + (len(N_MODEL_LABEL) - len(NW_MODEL_LABEL))],
                alpha=.8
            )
        ax.set_xlabel("Sample Size")
        if pop_val_i == 0:
            ax.set_ylabel("Definability")
        if pop_val_i == 0 and prob_val == .2:
            lab_order = [5, 4, 3, 2, 1, 0]
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[i] for i in lab_order]
            labels = [labels[i] for i in lab_order]
            s = fig.subplotpars
            bb = [s.left, s.top + 0.09, s.right - s.left, 0.05]
            ax.legend(handles, labels, loc=8, bbox_to_anchor=bb, ncol=3, mode="expand", borderaxespad=0, bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")
        ax.set_title(f"{POPS_LABEL[POP_I]}$ = {phi_0}$")
        ax.set_xlim(xmin=1 - .5, xmax=NUM_INS + .5)
        ax.set_ylim(ymin=-0.05, ymax=1.05)
    plt.savefig(os.path.join(output_dir, f"simAb_prob{prob_val}_def.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"simAb_prob{prob_val}_def.pdf"), bbox_inches="tight")
    plt.show()
    plt.close()

# %%
