# this file encoding: utf-8
# %%
import os
import numpy as np
import modules.models as models
import modules.ContingencyTable as ct
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["font.size"] = 13
plt.rcParams['axes.linewidth'] = 1.0

READ_PICKLE = False
TRIAL_N = 100
TABLE_N = 1000
PROBS = np.round(np.arange(0.1, 1., 0.1), 1)
MU_SAMPLES = [7, 21, 56]
filters = [(True, True, True), (False, True, True)]
NW_SAMPLING = [False, True]

POPS = [models.phi0]
POPS_LABEL = ["$\phi_0$"]
POP_I = 0
pop_name = str(POPS[POP_I]).split(" ")[1]
output_dir = os.path.join("simBb", pop_name)
os.makedirs(output_dir, exist_ok=True)
MODELS = [models.phi, models.dfh, models.paris]
MODELS_LABEL = ['$\phi$', '$DFH$', '$pARIs$']

models_result = np.zeros([len(MODELS), TABLE_N])
pop_result = np.zeros(TABLE_N)
result = np.zeros([len(filters), len(NW_SAMPLING), len(MODELS), TRIAL_N])
result_final = np.zeros([len(filters), len(NW_SAMPLING), len(MODELS), len(MU_SAMPLES), len(PROBS)])

if not READ_PICKLE:
    for fil_i, fil in enumerate(filters):
        print('filter: ' + str(fil))
        for is_nw_i, is_nw in enumerate(NW_SAMPLING):
            print('nw: ' + str(is_nw))
            for sample_i, sample_mu in tqdm(enumerate(MU_SAMPLES)):
                for prob_i, prob_i_value in enumerate(PROBS):
                    result = np.zeros_like(result)
                    for trial_i in range(TRIAL_N):
                        for table_i in range(TABLE_N):
                            table = ct.ContingencyTable(prob_i_value, prob_i_value, sample_mu, is_nw, discard_preventive=fil[0], discard_undef_phi=fil[1], discard_undef_dfh=fil[2])
                            calc_abcd = table.abcd_array
                            pop_result[table_i] = POPS[POP_I](table)
                            for calc_index, calc_model in enumerate(MODELS):
                                models_result[calc_index][table_i] = calc_model(calc_abcd)
                        for mod_res_i, model_result in enumerate(models_result):
                            corr = np.corrcoef(model_result, pop_result)[0, 1]
                            result[fil_i][is_nw_i][mod_res_i][trial_i] = (corr * corr)
                        models_result = np.zeros_like(models_result)
                        pop_result = np.zeros_like(pop_result)
                    for model_i in range(len(MODELS)):
                        result_final[fil_i][is_nw_i][model_i][sample_i][prob_i] = np.nanmean(result[fil_i][is_nw_i][model_i])
    with open(os.path.join(output_dir, "result_final_simBb.pkl"), "wb") as f:
        pickle.dump(result_final, f)
else:
    with open(os.path.join(output_dir, "result_final_simBb.pkl"), "rb") as f:
        result_final = pickle.load(f)


# %%


def make_plot(result_final):
    for fil_i, fil in enumerate(filters):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        for is_nw_i, is_nw in enumerate(NW_SAMPLING):

            linestyles = [':', '--', '-', '-.']
            markers = ['o', 's', 'D', 'h']
            colors = ['orange', 'green', 'blue', 'magenta']
            mecs = ['orange', 'green', 'blue', 'magenta']
            mfcs = ['orange', 'green', 'blue', 'magenta']

            for model_i in range(len(MODELS)):
                model_name = MODELS_LABEL[model_i] + " (" + ('$N_W$' if is_nw is True else '$N$')
                for result_index, result_y_value in enumerate(result_final[fil_i][is_nw_i][model_i]):
                    if MODELS_LABEL[model_i] == '$\phi$' and is_nw is True:
                        continue
                    if MODELS_LABEL[model_i] != '$\phi$' and result_index != 0:
                        continue
                    lab = str(model_name) + r'$\approx$' + str(MU_SAMPLES[result_index]) + ')'
                    mark = markers[model_i]
                    ax.plot(PROBS, result_y_value, label=lab, linestyle=linestyles[result_index], marker=mark, markeredgewidth=2, markersize=6, color=colors[model_i], markeredgecolor=mecs[model_i], markerfacecolor=(mfcs[model_i] if not is_nw else "white"), alpha=0.7)
            ax.set_xlabel("$P(C) = P(E)$")
            ax.set_ylabel("$r^2$")
            ax.grid(True)
            ax.set_ylim(0.0, .9)

        if fil_i == 1:
            lab_order = [6, 5, 4, 3, 2, 1, 0]
            handles, labels = ax.get_legend_handles_labels()
            labels = [labels[i] for i in lab_order]
            handles = [handles[i] for i in lab_order]
            ax.legend(handles, labels, ncol=2, loc="best", frameon=True, fancybox=False, edgecolor="k")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'simBb_' + pop_name + "_fil" + str(fil_i) + ".pdf"))
        plt.show()
        plt.close()


make_plot(result_final)

# %%
