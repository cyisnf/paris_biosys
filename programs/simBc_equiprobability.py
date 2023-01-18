# this file encoding: utf-8
# %%
import csv
import os
import modules.ContingencyTable as ct
import modules.models as models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import importlib
importlib.reload(models)
importlib.reload(ct)

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
PROBS = np.round(np.arange(.1, .55, .1), 1)
MU_SAMPLES = [7]
POPS = [models.phi0]
POPS_LABEL = ["$\phi_0$"]
POP_I = 0
pop_name = str(POPS[POP_I]).split(" ")[1]
output_dir = os.path.join("simBc", pop_name)
os.makedirs(output_dir, exist_ok=True)

MODELS = [models.dfh, models.paris]
MODELS_NAME = ["$DFH$", "$pARIs$"]
USED_SAMPLING = ["nw", "nw"]


def main():
    if not READ_PICKLE:
        models_result = np.zeros([len(MODELS), TABLE_N])
        pop_result = np.zeros(TABLE_N)
        # result is correlation between phi and model for each trial.
        result = np.zeros([len(MODELS), TRIAL_N])
        # result_final for draw the plot.
        result_final = np.zeros([len(MODELS), len(MU_SAMPLES), len(PROBS), len(PROBS)])

        for mu_sample_i, mu_sample in enumerate(MU_SAMPLES):
            for prob_ci, prob_ci_value in tqdm(enumerate(PROBS)):
                for prob_ei, prob_ei_value in enumerate(PROBS):
                    result = np.zeros_like(result)
                    for trial_i in range(TRIAL_N):
                        for table_i in range(TABLE_N):
                            table = ct.ContingencyTable(prob_ci_value, prob_ei_value, mu_sample, is_nw=True, discard_preventive=False, discard_undef_phi=True, discard_undef_dfh=True)
                            calc_abcd = table.abcd_array
                            pop_result[table_i] = POPS[POP_I](table)
                            for calc_index, calc_model in enumerate(MODELS):
                                models_result[calc_index][table_i] = calc_model(calc_abcd)
                        for mod_res_i, model_result in enumerate(models_result):
                            corr = np.corrcoef(model_result, pop_result)[0, 1]
                            result[mod_res_i][trial_i] = (corr * corr)
                        models_result = np.zeros_like(models_result)
                        pop_result = np.zeros_like(pop_result)

                    for res_i, res in enumerate(result):
                        result_final[res_i][mu_sample_i][prob_ci][prob_ei] = np.mean(res)
        with open(os.path.join(output_dir, "result_final_simBc.pkl"), "wb") as f:
            pickle.dump(result_final, f)
    else:
        with open(os.path.join(output_dir, "result_final_simBc.pkl"), "rb") as f:
            result_final = pickle.load(f)

    # make modelname and savefile.
    for model_i, model in enumerate(MODELS):
        model_name = str(model).split(" ")[1] + "_" + USED_SAMPLING[model_i]
        edit_csv(os.path.join(output_dir, model_name), result_final[model_i])


def edit_csv(filename, result_final):
    """ this function output result to csv files. """
    file = open(filename + ".CSV", 'w', newline="")
    csvfile = csv.writer(file)
    header = ["P(C)", "P(E)", "P(C)-P(E)", "mean[P(C)+P(E)]", "sampleN", "r^2"]
    csvfile.writerow(header)
    for saindex, sam in enumerate(result_final):
        for xindex, row_result_final in enumerate(sam):
            for yindex, resultdata in enumerate(row_result_final):
                probc = PROBS[xindex]  # xindex*0.1 + 0.1
                probe = PROBS[yindex]  # yindex*0.1 + 0.1
                pcpe = np.round(probc - probe, 1)
                mean = (probc + probe) / 2.0
                sample = MU_SAMPLES[saindex]  # saindex
                res = [probc, probe, pcpe, mean, sample]
                res.append(resultdata)
                csvfile.writerow(res)
    file.close()

    readfilename = filename + ".CSV"
    file = open(readfilename, "r")
    csvfile = csv.reader(file)
    next(csvfile)  # skip header
    datafile = []
    for row in csvfile:
        datafile.append(row)
    file.close()

    row_names = ["-0.4", "-0.2", "0.0", "0.2", "0.4"]
    col_names = ["0.1", "0.2", "0.3", "0.4", "0.5"]

    table = np.zeros([len(row_names), len(col_names)])
    table[:] = np.nan  # NAN for dont draw unnecessary line.
    tables = []  # this variable will have result for draw for each MU_SAMPLES.
    for sample_n in MU_SAMPLES:
        for row_ind, row_name in enumerate(row_names):
            for col_ind, col_name in enumerate(col_names):
                for row in datafile:
                    if np.round(float(row[0]), 1) <= 0.5 and np.round(float(row[1]), 2) <= 0.5:  # P(C),P(E) <= 0.5 rule
                        if np.round(float(row[2]), 1) == np.round(float(row_name), 1) and np.round(float(row[3]), 2) == np.round(float(col_name), 2) and int(row[4]) == sample_n:
                            table[row_ind][col_ind] = row[5]
        tables.append(table)
        table = np.zeros([len(row_names), len(col_names)])

    for ind, data in enumerate(tables):
        file = open(filename + "_" + str(ind) + ".CSV", 'w', newline="")
        csvfile = csv.writer(file)

        header = ["P(C)-P(E)", "mean0.1", "mean0.2", "mean0.3", "mean0.4", "mean0.5"]
        csvfile.writerow(header)
        for rowindex, rowdata in enumerate(data):
            rowtmp = [row_names[rowindex]]
            rowtmp.extend(rowdata)
            csvfile.writerow(rowtmp)
        file.close()


def plotter():
    fig = plt.figure(figsize=(5, 5))
    y_ticks = np.round(np.arange(.0, .81, .1), 1)
    markers = ["s-", "o-", "^-", "v-", "D-"]
    labs = ["m = .1", "m = .2", "m = .3", "m = .4", "m = .5"]

    index = 0
    ax0 = fig.add_subplot(1, 2, 1)
    calc_model = MODELS[index]
    model_name = str(calc_model).split(" ")[1]
    readfilename = os.path.join(output_dir, model_name + "_" + USED_SAMPLING[index] + "_0.CSV")
    df0 = pd.read_csv(readfilename).set_index("P(C)-P(E)")
    df0 = df0.set_axis(labs, axis=1)
    df0.plot(ax=ax0, grid=True, style=markers)
    ax0.set_title(f"{MODELS_NAME[0]}")
    ax0.set_ylabel("$r^2$")
    ax0.set_ylim([0, 0.8])
    ax0.set_xlim([-0.5, 0.5])
    # ax0.set_xticks(x_ticks)
    ax0.set_yticks(y_ticks)
    ax0.set_xlabel("$P(C) - P(E)$")

    s = fig.subplotpars
    bb = [s.left, s.top + 0.08, s.right - s.left, 0.05]
    ax0.legend(loc=8, bbox_to_anchor=bb, ncol=3, mode="expand", borderaxespad=0, bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")

    index = 1
    ax1 = fig.add_subplot(1, 2, 2)
    calc_model = MODELS[index]
    model_name = str(calc_model).split(" ")[1]
    readfilename = os.path.join(output_dir, model_name + "_" + USED_SAMPLING[index] + "_0.CSV")
    df1 = pd.read_csv(readfilename).set_index("P(C)-P(E)")
    df1 = df1.set_axis(labs, axis=1)
    df1.plot(ax=ax1, grid=True, style=markers, legend=False)
    ax1.set_title(f"{MODELS_NAME[1]}")
    ax1.set_ylim([0, 0.8])
    ax1.set_xlim([-0.5, 0.5])
    # ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xlabel("$P(C) - P(E)$")

    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "SimBc_equiprobability.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "SimBc_equiprobability.png"), bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    plotter()

# %%
