# %%
import numpy as np
import modules.models as models
import modules.ContingencyTable_pop_inc as ct
import csv
import os

POP_VALUES = np.round(np.arange(0, 1.1, 0.1), 1)
TABLE_N = 10000
SAMPLE_SIZE = 30
PROB = 0.2
POPS = [models.phi0]
POP_I = 0
pop_name = str(POPS[POP_I]).split(" ")[1]
output_dir = os.path.join("outputs", "simC", pop_name)
os.makedirs(output_dir, exist_ok=True)
N_MODELS = [models.deltap, models.dfh, models.paris, models.phi]
NW_MODELS = [models.dfh, models.paris]
N_MODEL_NAMES = ['DP', 'DFH', 'pARIs', 'phi']
NW_MODEL_NAMES = ['DFH', 'pARIs']
NW_FLAGS = [False, True]

all_dir_path = os.path.join(output_dir, "data_all")
mean_dir_path = os.path.join(output_dir, "data_mean")
sd_dir_path = os.path.join(output_dir, "data_sd")
os.makedirs(all_dir_path, exist_ok=True)
os.makedirs(mean_dir_path, exist_ok=True)
os.makedirs(sd_dir_path, exist_ok=True)

for is_nw in NW_FLAGS:
    nw_label = "NW" if is_nw else "N"
    MODELS = NW_MODELS if is_nw else N_MODELS
    MODEL_NAMES = NW_MODEL_NAMES if is_nw else N_MODEL_NAMES
    model_values = np.zeros([len(MODELS), SAMPLE_SIZE, TABLE_N])
    raw_model_values = np.zeros([len(MODELS), SAMPLE_SIZE, TABLE_N])
    # mean_values[model][ins]
    mean_values = np.zeros([len(MODELS), SAMPLE_SIZE])
    sd_values = np.zeros([len(MODELS), SAMPLE_SIZE])

    for pop_val in POP_VALUES:
        for tab_i in range(TABLE_N):
            table = ct.ContingencyTable(PROB, PROB, pop_name, pop_val, is_nw)
            for ins_i in range(SAMPLE_SIZE):
                table.sampling_inc()
                for model_i, model in enumerate(MODELS):
                    try:
                        tmp = model(table.abcd_array)
                        model_values[model_i][ins_i][tab_i] = tmp
                    except Exception:
                        model_values[model_i][ins_i][tab_i] = np.nan
        # calculate mean and sd.
        for ins_i in range(SAMPLE_SIZE):
            for model_i, model in enumerate(MODELS):
                mean_values[model_i][ins_i] = np.nanmean(model_values[model_i][ins_i])
                sd_values[model_i][ins_i] = np.nanstd(model_values[model_i][ins_i])
        # output all model values.
        all_file_name = nw_label + "_all_" + pop_name + "_" + str(pop_val)
        for mod_name_i, mod_name in enumerate(MODEL_NAMES):
            with open(os.path.join(all_dir_path, mod_name + '_' + all_file_name + ".csv"), "w", newline="") as file:
                csvfile = csv.writer(file)
                for ins_i in range(SAMPLE_SIZE):
                    row = [model_values[mod_name_i][ins_i][tab_i] for tab_i in range(TABLE_N)]
                    csvfile.writerow(row)
        # output mean.
        mean_file_name = nw_label + "_mean_" + pop_name + "_" + str(pop_val)
        with open(os.path.join(mean_dir_path, mean_file_name + ".csv"), 'w', newline="") as file:
            csvfile = csv.writer(file)
            # writing
            csvfile.writerow(MODEL_NAMES)
            for ins_i in range(SAMPLE_SIZE):
                row = [mean_values[model_i][ins_i] for model_i in range(len(MODELS))]
                csvfile.writerow(row)
        # output sd.
        sd_file_name = nw_label + "_sd_" + pop_name + "_" + str(pop_val)
        with open(os.path.join(sd_dir_path, sd_file_name + ".csv"), 'w', newline="") as file:
            csvfile = csv.writer(file)
            # writing
            csvfile.writerow(MODEL_NAMES)
            for ins_i in range(SAMPLE_SIZE):
                row = [sd_values[model_i][ins_i] for model_i in range(len(MODELS))]
                csvfile.writerow(row)
        # init
        model_values = np.zeros_like(model_values)
        mean_values = np.zeros_like(mean_values)
        sd_values = np.zeros_like(sd_values)

# %%
