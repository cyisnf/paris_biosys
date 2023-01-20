# %%
import os
import numpy as np
import pandas as pd
import sys
sys.path.append("./modules")
import modules.models as models
import math

output_dir = os.path.join(".", "outputs")
os.makedirs(output_dir, exist_ok=True)
exp_data_dir = './experimental_data'
experiment_filenames = sorted(os.listdir(exp_data_dir))
exp_names = [fn.rstrip('.csv') for fn in experiment_filenames]


def fisher_z(r):
    return 0.5 * math.log((1 + r) / (1 - r))


def linear_trans(data):
    bef_conv_max = 100
    bef_conv_min = -100
    aft_conv_max = 100
    aft_conv_min = 0
    return (aft_conv_max - aft_conv_min) / (bef_conv_max - bef_conv_min) * (data - bef_conv_min) + aft_conv_min


significant_figure = 10
target_model_names = ['pARIs', 'DFH', 'phi', 'DP']
latex_model_names = ['pARIs', 'DFH', '\phi', '\delata P']
target_models = [models.paris, models.dfh, models.phi, models.deltap]
qty_stims = pd.Series([-1 for i in range(len(experiment_filenames))], name='qty_stims')
corr_df = pd.DataFrame(columns=target_model_names)

for exp_i, exp_fn in enumerate(experiment_filenames):
    # Read experimental data
    exp_df = pd.read_csv(os.path.join(exp_data_dir, str(exp_fn)))
    exp_df = exp_df.drop(exp_df.columns[0], axis=1)
    if (np.any(exp_df['rating'] < 0)):
        exp_df['rating'] = linear_trans(exp_df['rating'])

    # Calculate model values
    for stim_i in range(len(exp_df)):
        for model_i, model_v in enumerate(target_models):
            model_value = model_v(np.round(np.array(exp_df.loc[stim_i, 'a':'d']), significant_figure))
            exp_df.loc[stim_i, target_model_names[model_i]] = model_value

    # Calculate correlation coefficients between experimental data and each model's value
    corr_dict = {}
    rms_dict = {}
    qty_stims[exp_i] = len(exp_df)
    for model_i, model_v in enumerate(target_models):
        corr_dict[target_model_names[model_i]] = (np.corrcoef(exp_df['rating'], exp_df[target_model_names[model_i]])[0, 1])
        rms_dict[target_model_names[model_i]] = math.sqrt(np.mean((exp_df['rating'] - exp_df[target_model_names[model_i]] * 100)**2))
    corr_df = corr_df.append(corr_dict, ignore_index=True)

# Fisher z-transformation on the correlation coefficients
z_df = corr_df.applymap(fisher_z)  # z値
w_ser = qty_stims - 3
v_ser = 1 / w_ser

tidy_df = pd.DataFrame(corr_df.stack(), columns=["corr"])
tidy_df.loc[:, 'qty_stims'] = qty_stims.loc[tidy_df.index.get_level_values(0)].values
tidy_df.loc[:, 'z'] = pd.DataFrame(z_df.stack(), columns=["z"])
tidy_df.loc[:, 'v'] = v_ser.loc[tidy_df.index.get_level_values(0)].values
tidy_df.loc[:, 'w'] = w_ser.loc[tidy_df.index.get_level_values(0)].values
tidy_df.loc[:, 'rho_L'] = np.tanh(tidy_df.loc[:, 'z'] - 1.96 * np.sqrt(tidy_df.loc[:, 'v']))
tidy_df.loc[:, 'rho_U'] = np.tanh(tidy_df.loc[:, 'z'] + 1.96 * np.sqrt(tidy_df.loc[:, 'v']))
tidy_df.loc[:, 'w*z'] = tidy_df.loc[:, 'w'] * tidy_df.loc[:, 'z']

# Integrate correlation coefficients with fixed effects model
zeta_hat_dict = {}
for model_i, model_v in enumerate(target_models):
    zeta_hat_dict[target_model_names[model_i]] = np.sum(w_ser * z_df[target_model_names[model_i]]) / np.sum(w_ser)

zeta_hat_ser = pd.Series(zeta_hat_dict)
rho_ser = zeta_hat_ser.apply(math.tanh)
v_hat_zeta_hat = 1 / np.sum(w_ser)
se_hat_zeta_hat = np.sqrt(v_hat_zeta_hat)

zeta_hat_df = pd.DataFrame.from_dict(zeta_hat_dict, orient="index", columns=['zeta_hat'])
zeta_hat_df.loc[:, "zeta_L"] = zeta_hat_df.loc[:, "zeta_hat"] - 1.96 * se_hat_zeta_hat
zeta_hat_df.loc[:, "zeta_U"] = zeta_hat_df.loc[:, "zeta_hat"] + 1.96 * se_hat_zeta_hat
zeta_hat_df.loc[:, 'rho_hat'] = np.tanh(zeta_hat_df.loc[:, 'zeta_hat'])

zeta_hat_df.loc[:, 'rho_L'] = np.tanh(zeta_hat_df.loc[:, 'zeta_L'])
zeta_hat_df.loc[:, 'rho_U'] = np.tanh(zeta_hat_df.loc[:, 'zeta_U'])

output_df = np.round(pd.concat(
    [
        tidy_df.unstack()['corr'].T.set_axis(exp_names, axis=1),
        zeta_hat_df.loc[:, 'rho_hat':'rho_U']
    ],
    axis=1
), 3)
output_df.to_csv('outputs/meta_analysis.csv')
# %%
