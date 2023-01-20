# this file encoding: utf-8
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import modules.models as models
import modules.ContingencyTable_pop as ct
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["font.size"] = 13
plt.rcParams['axes.linewidth'] = 1.0

READ_PICKLE = False
TABLE_N = 10000
PROB_ARRAY = np.round(np.array([.2]), 1)
POP_VALUES_ARRAY = np.round(np.arange(.0, 1.1, .1), 1)
MU_SAMPLES = [7, 21, 56]
NW_SAMPLING = [False]
FILTERS = [(False, False, False)]
extensions = ["pdf", "png"]
POPS = [models.phi0]
POPS_NAME = [r"$\phi_0$"]
POP_I = 0
pop_name = str(POPS[POP_I]).split(" ")[1]
MODELS_ARRAY = [models.phi, models.deltap, models.dfh, models.paris]
MODELS_NAME = [r"$\phi$", r"$\Delta P$", r"$DFH$", r"$pARIs$"]
output_dir = os.path.join("outputs", "simC", pop_name)
os.makedirs(output_dir, exist_ok=True)

result_all = np.zeros([
    len(FILTERS),
    len(NW_SAMPLING),
    len(MU_SAMPLES),
    len(PROB_ARRAY),
    len(POPS),
    len(POP_VALUES_ARRAY),
    len(MODELS_ARRAY),
    TABLE_N, ])
result_mean = np.zeros([
    len(FILTERS),
    len(NW_SAMPLING),
    len(MU_SAMPLES),
    len(PROB_ARRAY),
    len(POPS),
    len(MODELS_ARRAY),
    len(POP_VALUES_ARRAY)])


if READ_PICKLE:
    result_all = np.load(os.path.join(output_dir, 'result_all_simCb.npy'))
    result_all = np.load(os.path.join(output_dir, 'result_mean_simCb.npy'))
else:
    for fil_i, fil in enumerate(FILTERS):
        print(f"filter: {fil}")
        for is_nw_i, is_nw in enumerate(NW_SAMPLING):
            print(f"nw: {is_nw}")
            for sample_i, sample_mu in tqdm(enumerate(MU_SAMPLES)):
                for prob_vi, prob_v in enumerate(PROB_ARRAY):
                    for pop_mi, pop in enumerate(POPS):
                        for pop_vi, pop_v in enumerate(POP_VALUES_ARRAY):
                            for table_i in range(TABLE_N):
                                table = ct.ContingencyTable(
                                    prob_v, prob_v,
                                    str(pop).split(" ")[1],
                                    pop_v, sample_mu, is_nw,
                                    fil[0], fil[1], fil[2])
                                calc_abcd = table.abcd_array
                                for model_i, calc_model in enumerate(MODELS_ARRAY):
                                    result_all[fil_i][is_nw_i][sample_i][prob_vi][pop_mi][pop_vi][model_i][table_i] = calc_model(calc_abcd)
                            for model_i, calc_model in enumerate(MODELS_ARRAY):
                                result_mean[fil_i][is_nw_i][sample_i][prob_vi][pop_mi][model_i][pop_vi] = np.nanmean(result_all[fil_i][is_nw_i][sample_i][prob_vi][pop_mi][pop_vi][model_i])
    np.save(os.path.join(output_dir, 'result_all_simCb.npy'), result_all)
    np.save(os.path.join(output_dir, 'result_mean_simCb.npy'), result_all)


# %%


def plotter(path, pop_mi, result_mean, ext):
    x_val = POP_VALUES_ARRAY
    colors = ["orange", "red", "green", "blue", "magenta"]
    for fil_i, fil in enumerate(FILTERS):
        for is_nw_i, is_nw in enumerate(NW_SAMPLING):
            print(f"nw: {is_nw}")
            fig, axes = plt.subplots(len(MU_SAMPLES), 1, figsize=(3.5, 8))
            for samp_i in range(len(MU_SAMPLES)):
                for prob_vi in range(len(PROB_ARRAY)):
                    for model_i in range(len(MODELS_ARRAY)):
                        y_val = result_mean[fil_i][is_nw_i][samp_i][prob_vi][pop_mi][model_i]
                        ax = axes[samp_i]
                        ax.plot(
                            x_val, y_val,
                            label=MODELS_NAME[model_i],
                            color=colors[model_i],
                            alpha=0.6
                        )
                        ax.set_title(f"$N={MU_SAMPLES[samp_i]}$")
                        if samp_i == 2:
                            ax.set_xlabel(POPS_NAME[pop_mi])
                        ax.set_ylabel("Model value")
                    ax.plot(x_val, x_val, linestyle="--", color="black", alpha=0.8)
                    ax.grid()
            lab_order = [2, 3, 0, 1]
            handles, labels = axes[-1].get_legend_handles_labels()
            labels = [labels[i] for i in lab_order]
            handles = [handles[i] for i in lab_order]
            s = fig.subplotpars
            bb = [s.left, s.top + 0.04, s.right - s.left, 0.05]
            axes[0].legend(handles, labels, loc=8, bbox_to_anchor=bb, ncol=2, mode="expand", borderaxespad=0, bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"simCb_linear_raw.{ext}"), bbox_inches="tight", pad_inches=0.05)
            plt.show()
            plt.close()


for pop_mi in range(len(POPS)):
    for ext in extensions:
        plotter(output_dir, pop_mi, result_mean, ext)

# %%
