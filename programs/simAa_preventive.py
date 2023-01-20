# this file encoding: utf-8
# %%
import pickle
import joblib
import os
import numpy as np
import modules.models as models
import modules.ContingencyTable as ct
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["font.size"] = 13
plt.rcParams['axes.linewidth'] = 1.0

TABLE_N = 100000
READ_PICKLE = False
PROBS = np.round(np.arange(0.1, 1., 0.1), 1)
MU_SAMPLES = [7, 21, 56]
FILTER = (False, True, True)
NW_SAMPLING = [False, True]
POPS = [models.phi]
POPS_NAME = ['$\phi_0$']
POP_I = 0
CONDITIONS = ["preventive", "non-generative"]
COND_I = 0
pop_name = str(POPS[POP_I]).split(" ")[1]
output_dir = os.path.join("outputs", "simAa", pop_name)
os.makedirs(output_dir, exist_ok=True)
PICKLE_NAME = f"{CONDITIONS[COND_I]}_{TABLE_N}"


def is_non_generative(prob_i_value, sample_mu, is_nw, fil0, fil1, fil2):
    table = ct.ContingencyTable(prob_i_value, prob_i_value, sample_mu, is_nw, fil0, fil1, fil2)
    return table.is_non_generative()


def is_preventive(prob_i_value, sample_mu, is_nw, fil0, fil1, fil2):
    table = ct.ContingencyTable(prob_i_value, prob_i_value, sample_mu, is_nw, fil0, fil1, fil2)
    return table.is_preventive()


pickle_path = os.path.join(output_dir, PICKLE_NAME)
if READ_PICKLE:
    with open(pickle_path + ".pkl", "rb") as f:
        cnt = pickle.load(f)
else:
    cnt = np.zeros([len(NW_SAMPLING), len(MU_SAMPLES), len(PROBS)])
    for is_nw_i, is_nw in enumerate(NW_SAMPLING):
        print('nw: ' + str(is_nw))
        for sample_i, sample_mu in enumerate(MU_SAMPLES):
            for prob_i, prob_i_value in tqdm(enumerate(PROBS)):
                if CONDITIONS[COND_I] == "preventive":
                    func = is_preventive
                elif CONDITIONS[COND_I] == "non-generative":
                    func = is_non_generative
                is_preventives = joblib.Parallel(n_jobs=-1)([joblib.delayed(func)(prob_i_value, sample_mu, is_nw, FILTER[0], FILTER[1], FILTER[2]) for _ in range(TABLE_N)])
                cnt[is_nw_i, sample_i, prob_i] = np.sum(is_preventives) / TABLE_N
    with open(pickle_path + ".pkl", "wb") as f:
        pickle.dump(cnt, f)

# %%

# Plot figures
markers = ["o", "^", "s", "D"]
linestyles = ['-', '--', ':', '-.']
colors = ["orange", "red", "green", "blue", "magenta"]
mecs = ["orange", "red", "green", "blue", "magenta"]
mfcs = ["orange", "red", "green", "blue", "magenta"]

for is_nw_i, is_nw in enumerate(NW_SAMPLING):
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    print('nw: ' + str(is_nw))
    for sample_i, sample_mu in enumerate(MU_SAMPLES):
        lab = f"$\mu = {sample_mu}$"
        ax.plot(PROBS, cnt[is_nw_i, sample_i], label=lab, marker=markers[sample_i])
    ax.set_ylabel(f"Proportion of {CONDITIONS[COND_I]} samples")
    ax.set_ylim(.0, .25)
    if is_nw_i == len(NW_SAMPLING) - 1:
        # ax.set_title("$N_W$-Sampling")
        ax.legend(frameon=True, loc="upper left", fancybox=False, edgecolor="k")
        if CONDITIONS[COND_I] == "preventive":
            ax.set_ylim(.0, .25)
        elif CONDITIONS[COND_I] == "non-generative":
            ax.set_ylim(.0, .25)
    ax.grid()
    ax.set_xlabel("$P(C)=P(E)$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"simAa_{CONDITIONS[COND_I]}_{'nw' if is_nw else 'n'}.pdf"), bbox_inches='tight')
    plt.show()
    # plt.close()

# %%
