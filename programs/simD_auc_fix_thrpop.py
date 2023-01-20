# %%
import pickle
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import os
import modules.models as models
import modules.ContingencyTable as ct
from tqdm import tqdm
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["font.size"] = 13
plt.rcParams['axes.linewidth'] = 1.0

READ_FROM_PICKLE = False
USE_ROC = True  # when False, use PR
extensions = ["pdf", "png"]
TABLE_N = 100000

THR_POPS = np.round(np.array([.5]), 1)
PROBS = np.round(np.array([.2, .5, .8]), 1)
SAMPLE_SIZE = [4, 8, 16, 32, 64, 128, 256, 512]
LIST_IS_NW = [False, True]
POPS = [models.phi0]
POP_I = 0
MODEL_LABELS = [r'$\phi_0$']
MODELS = [models.phi, models.deltap, models.dfh, models.paris]
MODEL_LABELS = [r'$\phi$', r'$\Delta P$', r'$DFH$', r'$pARIs$']
NW_MODELS = [models.dfh, models.paris]
NW_MODEL_LABELS = [r'$DFH$', r'$pARIs$']
pop_name = str(POPS[POP_I]).split(" ")[1]
output_dir = os.path.join("outputs", "simD", pop_name)
os.makedirs(output_dir, exist_ok=True)


def create_table(prob_c, prob_e, sample_N, is_nw):
    return ct.ContingencyTable(prob_c, prob_e, sample_N, is_nw, discard_preventive=False, discard_undef_phi=False, discard_undef_dfh=False)


def sim_f(n_sample, is_nw, thr_pop, prob):
    MODS = MODELS if is_nw else MODELS
    models_vals = np.zeros([len(MODS), TABLE_N])
    scores = np.zeros([len(MODS)])
    tables = joblib.Parallel(n_jobs=-1)([joblib.delayed(create_table)(prob, prob, n_sample, is_nw) for _ in range(TABLE_N)])
    pop_values = joblib.Parallel(n_jobs=-1)([joblib.delayed(POPS[POP_I])(table) for table in tables])
    is_relations = np.array(pop_values) >= thr_pop
    if np.all(is_relations) or np.all(~is_relations):
        nan_arr = np.zeros_like(scores)
        nan_arr[:] = np.nan
        return nan_arr
    for calc_mod_i, calc_model in enumerate(NW_MODELS if is_nw else MODELS):
        models_vals[calc_mod_i] = joblib.Parallel(n_jobs=-1)([joblib.delayed(calc_model)(table.abcd_array) for table in tables])

        # Randomly substituting {0,1} for nan
        idx_na = np.isnan(models_vals[calc_mod_i])
        cnt_na = np.count_nonzero(idx_na)
        rands = np.array([random.random() for _ in range(cnt_na)])
        rand_judges = rands >= .5
        models_vals[calc_mod_i][idx_na] = rand_judges.astype(int)
        _is_relations = is_relations
        _model_vals = models_vals[calc_mod_i]

        if USE_ROC:
            scores[calc_mod_i] = roc_auc_score(_is_relations, _model_vals)
        else:
            precision, recall, thresholds = precision_recall_curve(_is_relations, _model_vals)
            scores[calc_mod_i] = auc(recall, precision)
    return scores


# run simulation
PICKLE_PATH = os.path.join(output_dir, f"simD_auc_fix_thrpop{THR_POPS[0]}.pickle")
if READ_FROM_PICKLE:
    with open(PICKLE_PATH, 'rb') as f:
        all_scores = pickle.load(f)
else:
    all_scores = np.zeros([len(LIST_IS_NW), len(SAMPLE_SIZE), len(PROBS), len(THR_POPS), len(MODELS)])

    for is_nw_i, is_nw in enumerate(LIST_IS_NW):
        MODS = MODELS if is_nw else MODELS
        for n_sample_i, n_sample in tqdm(enumerate(SAMPLE_SIZE)):
            for prob_i, prob in enumerate(PROBS):
                for thr_pop_i, thr_pop in enumerate(THR_POPS):
                    all_scores[is_nw_i][n_sample_i][prob_i][thr_pop_i] = sim_f(
                        n_sample=n_sample,
                        is_nw=is_nw,
                        thr_pop=thr_pop,
                        prob=prob)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(all_scores, f)

# %%


def plotter(ext):
    tp_all_scores = all_scores.transpose(0, 2, 3, 4, 1)
    markers = ['o', '^', 's', 'D', 'h']
    colors = ['orange', 'red', 'green', 'blue']
    qty_n = len(MODELS)
    qty_nw = len(NW_MODELS)
    n_markers = markers[:qty_n]
    nw_markers = markers[qty_n - qty_nw:qty_n]
    n_colors = colors[:qty_n]
    nw_colors = colors[qty_n - qty_nw:qty_n]

    fig = plt.figure(figsize=(10, 4))
    axes = []
    for i in range(len(PROBS)):
        axes.append(fig.add_subplot(1, len(PROBS), i + 1))

    thr_pop_i = 0
    for is_nw_i, is_nw in enumerate(LIST_IS_NW):
        MODS = NW_MODELS if is_nw else MODELS
        LABELS = NW_MODEL_LABELS if is_nw else MODEL_LABELS
        for prob_i, prob in enumerate(PROBS):
            for calc_mod_i, calc_model in enumerate(MODS):
                ax_idx = prob_i
                _colors = nw_colors if is_nw else n_colors
                _markers = nw_markers if is_nw else n_markers
                axes[prob_i].plot(
                    SAMPLE_SIZE,
                    tp_all_scores[is_nw_i][prob_i][thr_pop_i][calc_mod_i],
                    label=f"{LABELS[calc_mod_i]}" + (" $(N_W)$" if is_nw else " $(N)$"),
                    marker=_markers[calc_mod_i],
                    color=_colors[calc_mod_i],
                    markeredgecolor=_colors[calc_mod_i],
                    markerfacecolor=("white" if is_nw else _colors[calc_mod_i]))
                axes[ax_idx].set_title(f"$P(C) = P(E) = {prob}$")
                axes[ax_idx].set_xscale('log', base=2)
                axes[ax_idx].set_ylim(.6, 1.)
                axes[ax_idx].set_xlabel("Mean sample size $\mu$")
                if ax_idx == 0:
                    axes[ax_idx].set_ylabel("ROC-AUC")
                axes[ax_idx].grid(True)
    lab_order = [5, 4, 3, 2, 0, 1]
    handles, labels = axes[-1].get_legend_handles_labels()
    labels = [labels[i] for i in lab_order]
    handles = [handles[i] for i in lab_order]
    axes[0].legend(handles, labels, loc="lower right", frameon=True, fancybox=False, edgecolor="k")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'simD_auc_fix_thrpop{THR_POPS[0]}.{ext}'), bbox_inches="tight", pad_inches=0.05)


for ext in extensions:
    plotter(ext)

# %%
