#%%
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from pystan import StanModel
import pickle
from sklearn.linear_model import LinearRegression
import math
import arviz

def cal_EI_minimize_error(y_act, y_target, samples):
    # EI(Expected Improvement)
    error_pred = pd.DataFrame(abs(samples["y_new"] - y_target.values))
    error_act_min = abs(y_act - y_target.values).min()[0]

    kaizen = error_act_min - error_pred
    EI = (kaizen[error_pred < error_act_min]).sum() / len(error_pred)
    EI = pd.DataFrame(EI)
    return EI

def add_act_data_MLR(y_act, X_act, prediction_error, X_grid, y_grid):
    rowname = X_act.index[0:].tolist()
    prediction_error.loc[rowname,:] = 1000
    select_col = prediction_error[prediction_error[0] == prediction_error[0].min()].index[0]
    print(select_col)
    X_act = pd.concat([X_act,X_grid.loc[[select_col],:]])
    y_act = pd.concat([y_act,y_grid.loc[[select_col],:]])
    return X_act, y_act

def add_act_data_BO(y_act, X_act, EI, X_grid, y_grid):
    rowname = X_act.index[0:].tolist()
    EI.loc[rowname,:] = -10
    select_col = EI[EI[0] == EI[0].max()].index[0]
    print(select_col)
    X_act = pd.concat([X_act,X_grid.loc[[select_col],:]])
    y_act = pd.concat([y_act,y_grid.loc[[select_col],:]])
    return X_act, y_act

def cal_coef(X_act, y_act):
    model_lr = LinearRegression()
    model_lr.fit(X_act, y_act)
    coef = model_lr.coef_
    b = model_lr.intercept_
    return coef,b

def make_folder_BO_SP_GPR(seed_num, f_result):
    f_name = f_result/"seed_{}".format(seed_num)
    f_plot = f_name/"plot"
    f_hist = f_name/"hist"
    f_dens = f_name/"dens"
    f_dens2 = f_name/"dens2"
    f_trace = f_name/"trace"

    f_plot.mkdir(parents=True, exist_ok=True)
    f_hist.mkdir(parents=True, exist_ok=True)
    f_dens.mkdir(parents=True, exist_ok=True)
    f_dens2.mkdir(parents=True, exist_ok=True)
    f_trace.mkdir(parents=True, exist_ok=True)

    return f_plot, f_hist, f_dens, f_dens2, f_trace

def make_folder_BO_GPR(seed_num, f_result):
    f_name = f_result/"seed_{}".format(seed_num)
    f_plot = f_name/"plot"
    f_hist = f_name/"hist"
    f_dens = f_name/"dens"
    f_trace = f_name/"trace"

    f_plot.mkdir(parents=True, exist_ok=True)
    f_hist.mkdir(parents=True, exist_ok=True)
    f_dens.mkdir(parents=True, exist_ok=True)
    f_trace.mkdir(parents=True, exist_ok=True)

    return f_plot, f_hist, f_dens, f_trace

def plot_posterior_forest(fit,count,f_plot,para):
    arviz.plot_forest(fit,var_names=para)
    plt.savefig(f_plot/f"experiment_{count}.png", dpi=300)
    plt.show()

def plot_posterior_hist(fit,count,f_hist,para):
    arviz.rcParams["plot.density_kind"] = "hist"
    arviz.plot_density(fit,var_names=para)
    plt.savefig(f_hist/f"experiment_{count}.png", dpi=300)
    plt.show()

def plot_posterior_density(fit,count,f_dens,para):
    arviz.rcParams["plot.density_kind"] = "kde"
    arviz.plot_density(fit,var_names=para, shade=0.5, hdi_prob=0.997, textsize=30)
    plt.tight_layout()
    plt.savefig(f_dens/f"experiment_{count}.png", dpi=300)
    plt.show()

def plot_posterior_trace(fit,count,f_trace,para):
    arviz.plot_trace(fit,var_names=para)
    plt.tight_layout()
    plt.savefig(f_trace/f"experiment_{count}.png", dpi=300)
    plt.show()

def prepare_pca_data(data_raw):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_features = data_raw.iloc[:,1:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=data_raw.index)
    pca_df["Tb (K)"] = data_raw.iloc[:,0]

    return pca_df, scaler, pca

def save_pca_contribution_ratio(pca_model, output_folder):
    contribution_ratio = pca_model.explained_variance_ratio_
    cumulative_contribution_ratio = np.cumsum(contribution_ratio)

    df_contribution = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(contribution_ratio))],
        'Contribution ratio': contribution_ratio,
        'Cumulative contribution ratio': cumulative_contribution_ratio
    })

    file_path = output_folder / 'PCA_contribution_ratio.csv'
    df_contribution.to_csv(file_path, encoding='shift-jis', index=False)
    print(f"Saved PCA contribution ratios: {file_path}")

def create_pca_visualization_contour_only(pca_df, target_var, output_folder, pca_model):
    from scipy.interpolate import griddata

    contribution_ratio = pca_model.explained_variance_ratio_
    contribution_df = pd.DataFrame({
        'PC1': [contribution_ratio[0]],
        'PC2': [contribution_ratio[1]],
        'Cumulative': [contribution_ratio[0] + contribution_ratio[1]]
    })
    contribution_df.to_csv(output_folder / 'pca_contribution_ratio.csv', encoding='shift-jis', index=False)

    var_min = pca_df[target_var].min()
    var_max = pca_df[target_var].max()
    var_range = var_max - var_min
    extension = var_range * 0.3
    level = np.linspace(var_min - extension, var_max + extension, 15)

    pc1_min, pc1_max = pca_df["PC1"].min(), pca_df["PC1"].max()
    pc2_min, pc2_max = pca_df["PC2"].min(), pca_df["PC2"].max()

    margin = 0.1
    pc1_range = pc1_max - pc1_min
    pc2_range = pc2_max - pc2_min
    pc1_min -= margin * pc1_range
    pc1_max += margin * pc1_range
    pc2_min -= margin * pc2_range
    pc2_max += margin * pc2_range

    num_grid = 50
    pc1_grid = np.linspace(pc1_min, pc1_max, num_grid)
    pc2_grid = np.linspace(pc2_min, pc2_max, num_grid)
    PC1_mesh, PC2_mesh = np.meshgrid(pc1_grid, pc2_grid)

    points = pca_df[["PC1", "PC2"]].values
    values = pca_df[target_var].values
    Var_mesh = griddata(points, values, (PC1_mesh, PC2_mesh), method='linear')

    mask = np.isnan(Var_mesh)
    if mask.any():
        Var_mesh_nearest = griddata(points, values, (PC1_mesh, PC2_mesh), method='nearest')
        Var_mesh[mask] = Var_mesh_nearest[mask]

    title = f"{target_var} contour plot"
    file_pass = output_folder / f"{target_var}_contour.png"

    plt.figure(figsize=(6, 4.5))
    cont = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='gray', levels=level, linewidths=0.5, alpha=0.6)
    reduced_levels = level[::3]
    cont_reduced = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='black', levels=reduced_levels, linewidths=1.0)
    try:
        plt.clabel(cont_reduced, fmt='%.1f', fontsize=16, inline=True)
    except ValueError:
        pass
    plt.contourf(PC1_mesh, PC2_mesh, Var_mesh, cmap='coolwarm', alpha=0.5, levels=level)
    cbar = plt.colorbar()
    cbar.set_label(target_var, rotation=270, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('PC1', fontsize=18, labelpad=5)
    plt.ylabel('PC2', fontsize=18, labelpad=5)
    plt.title(title, fontsize=18, pad=15)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(file_pass, dpi=300)
    plt.close()

def create_pca_visualization_with_model(pca_df, scaler, pca_model, X_grid, model_lr,
                                        target_var, seed_num, count, N_first,
                                        target_num, initial_indices, current_indices,
                                        output_folder):
    from scipy.interpolate import griddata

    y_pred_all = model_lr.predict(X_grid)

    pca_df_pred = pca_df.copy()
    pca_df_pred[f"{target_var}_pred"] = y_pred_all

    var_min = pca_df[target_var].min()
    var_max = pca_df[target_var].max()
    var_range = var_max - var_min
    extension = var_range * 0.3
    level = np.linspace(var_min - extension, var_max + extension, 15)

    pc1_min, pc1_max = pca_df["PC1"].min(), pca_df["PC1"].max()
    pc2_min, pc2_max = pca_df["PC2"].min(), pca_df["PC2"].max()

    margin = 0.1
    pc1_range = pc1_max - pc1_min
    pc2_range = pc2_max - pc2_min
    pc1_min -= margin * pc1_range
    pc1_max += margin * pc1_range
    pc2_min -= margin * pc2_range
    pc2_max += margin * pc2_range

    num_grid = 50
    pc1_grid = np.linspace(pc1_min, pc1_max, num_grid)
    pc2_grid = np.linspace(pc2_min, pc2_max, num_grid)
    PC1_mesh, PC2_mesh = np.meshgrid(pc1_grid, pc2_grid)

    points = pca_df[["PC1", "PC2"]].values
    values = pca_df_pred[f"{target_var}_pred"].values
    Var_mesh = griddata(points, values, (PC1_mesh, PC2_mesh), method='linear')

    mask = np.isnan(Var_mesh)
    if mask.any():
        Var_mesh_nearest = griddata(points, values, (PC1_mesh, PC2_mesh), method='nearest')
        Var_mesh[mask] = Var_mesh_nearest[mask]

    target_pca = pca_df.loc[target_num, ["PC1", "PC2"]].values
    initial_pca = pca_df.loc[initial_indices, ["PC1", "PC2"]].values
    current_pca = pca_df.loc[current_indices, ["PC1", "PC2"]].values

    title = f"Seed {seed_num}: Experiment {count} ({target_var})"
    file_pass = output_folder / f"experiment_{count}.png"

    plt.figure(figsize=(6, 4.5))
    cont = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='gray', levels=level, linewidths=0.5, alpha=0.6)
    reduced_levels = level[::3]
    cont_reduced = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='black', levels=reduced_levels, linewidths=1.0)
    try:
        plt.clabel(cont_reduced, fmt='%.1f', fontsize=16, inline=True)
    except ValueError:
        pass
    plt.contourf(PC1_mesh, PC2_mesh, Var_mesh, cmap='coolwarm', alpha=0.5, levels=level)
    cbar = plt.colorbar()
    cbar.set_label(target_var, rotation=270, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('PC1', fontsize=18, labelpad=5)
    plt.ylabel('PC2', fontsize=18, labelpad=5)
    plt.title(title, fontsize=18, pad=15)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c="red", s=150, clip_on=False, zorder=20)

    if len(current_pca) > N_first:
        plt.scatter(current_pca[N_first:, 0], current_pca[N_first:, 1], c="blue", s=150, clip_on=False, zorder=20)

    plt.scatter(target_pca[0], target_pca[1], c="lime", s=150, clip_on=False,
                edgecolors='black', linewidths=2, marker='*', zorder=20)
    plt.tight_layout()
    plt.savefig(file_pass, dpi=300)
    plt.close()

def create_pca_visualization_with_pred_values(pca_df, scaler, pca_model, X_grid,
                                              pred_values, target_var, seed_num, count, N_first,
                                              target_num, initial_indices, current_indices,
                                              output_folder):
    from scipy.interpolate import griddata

    pca_df_pred = pca_df.copy()
    pca_df_pred[f"{target_var}_pred"] = pred_values

    var_min = pca_df[target_var].min()
    var_max = pca_df[target_var].max()
    var_range = var_max - var_min
    extension = var_range * 0.3
    level = np.linspace(var_min - extension, var_max + extension, 15)

    pc1_min, pc1_max = pca_df["PC1"].min(), pca_df["PC1"].max()
    pc2_min, pc2_max = pca_df["PC2"].min(), pca_df["PC2"].max()

    margin = 0.1
    pc1_range = pc1_max - pc1_min
    pc2_range = pc2_max - pc2_min
    pc1_min -= margin * pc1_range
    pc1_max += margin * pc1_range
    pc2_min -= margin * pc2_range
    pc2_max += margin * pc2_range

    num_grid = 50
    pc1_grid = np.linspace(pc1_min, pc1_max, num_grid)
    pc2_grid = np.linspace(pc2_min, pc2_max, num_grid)
    PC1_mesh, PC2_mesh = np.meshgrid(pc1_grid, pc2_grid)

    points = pca_df[["PC1", "PC2"]].values
    values = pca_df_pred[f"{target_var}_pred"].values
    Var_mesh = griddata(points, values, (PC1_mesh, PC2_mesh), method='linear')

    mask = np.isnan(Var_mesh)
    if mask.any():
        Var_mesh_nearest = griddata(points, values, (PC1_mesh, PC2_mesh), method='nearest')
        Var_mesh[mask] = Var_mesh_nearest[mask]

    target_pca = pca_df.loc[target_num, ["PC1", "PC2"]].values
    initial_pca = pca_df.loc[initial_indices, ["PC1", "PC2"]].values
    current_pca = pca_df.loc[current_indices, ["PC1", "PC2"]].values

    title = f"Seed {seed_num}: Experiment {count} ({target_var})"
    file_pass = output_folder / f"experiment_{count}.png"

    plt.figure(figsize=(6, 4.5))
    cont = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='gray', levels=level, linewidths=0.5, alpha=0.6)
    reduced_levels = level[::3]
    cont_reduced = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='black', levels=reduced_levels, linewidths=1.0)
    try:
        plt.clabel(cont_reduced, fmt='%.1f', fontsize=16, inline=True)
    except ValueError:
        pass
    plt.contourf(PC1_mesh, PC2_mesh, Var_mesh, cmap='coolwarm', alpha=0.5, levels=level)
    cbar = plt.colorbar()
    cbar.set_label(target_var, rotation=270, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('PC1', fontsize=18, labelpad=5)
    plt.ylabel('PC2', fontsize=18, labelpad=5)
    plt.title(title, fontsize=18, pad=15)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c="red", s=150, clip_on=False, zorder=20)

    if len(current_pca) > N_first:
        plt.scatter(current_pca[N_first:, 0], current_pca[N_first:, 1], c="blue", s=150, clip_on=False, zorder=20)

    plt.scatter(target_pca[0], target_pca[1], c="lime", s=150, clip_on=False,
                edgecolors='black', linewidths=2, marker='*', zorder=20)
    plt.tight_layout()
    plt.savefig(file_pass, dpi=300)
    plt.close()
