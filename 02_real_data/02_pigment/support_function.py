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

def cal_dE(pred, target):
    dE = (pred - target.values)**2
    dE = dE.sum(axis=1)
    dE = dE**0.5
    return dE

def cal_approach_degree(dE):
    for i in range(len(dE)-1):
        if dE.iloc[i] < dE.iloc[i+1]:
            dE.iloc[i+1] = dE.iloc[i]
    return dE

def calculate_dE_to_target(Y_end_all):
    Lab_initial = Y_end_all["initial"]
    Lab_experiment = Y_end_all["experiment"]
    Lab_target = Y_end_all["target"]
    seed_num_all = len(Lab_initial)

    dE_to_target_pd = pd.DataFrame()

    for i in range(seed_num_all):
        dE_initial = cal_dE(Lab_initial[i], Lab_target[i])
        dE_experiment = cal_dE(Lab_experiment[i], Lab_target[i])

        dE_ini_to_target = cal_approach_degree(dE_initial)
        dE_all = pd.concat([pd.Series(dE_ini_to_target.min()), dE_experiment])
        dE_to_target_pd[i] = cal_approach_degree(dE_all).values

    return dE_to_target_pd

def cal_EI_Lab(y_act, y_target, Y_pred):
    key = list(Y_pred.keys())[0]
    dE_pred = pd.DataFrame(index = range(0,len(Y_pred[key])))
    n_grid = Y_pred[key].shape[1]

    for i in range(n_grid):
        Lab_pred = pd.DataFrame(columns=["L","a","b"])
        for key in Y_pred.keys():
            Lab_pred[key] = Y_pred[key][:,i]
        dLab = Lab_pred - y_target
        dE_pred = pd.concat([dE_pred,(dLab**2).sum(axis=1)**0.5],axis=1)
    dE_pred.columns = range(0,len(dE_pred.T))

    dE_act_min = (((y_act - y_target)**2).sum(axis=1)**0.5).min()

    EI = dE_pred.copy()
    EI = (EI[EI < dE_act_min]).sum()/len(EI)
    EI = pd.DataFrame(EI)

    return EI

def add_act_data_MLR(y_act, X_act, dE, X_grid, y_grid):
    rowname = X_act.index[0:].tolist()
    dE.loc[rowname,:] = 1000
    select_col = dE[dE[0] == dE[0].min()].index[0]
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

def cal_coef_fixed_intercept(X_act, y_act, b_fixed):
    model_lr = LinearRegression(fit_intercept=False)
    y_adjusted = y_act - b_fixed
    model_lr.fit(X_act, y_adjusted)
    coef = model_lr.coef_
    b = b_fixed
    return coef, b

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

def plot_posterior_forest(fit, count, f_plot, para, key=None):
    arviz.plot_forest(fit,var_names=para)
    plt.savefig(f_plot/"experiment{}_{}.png".format(count, key), dpi=300)
    plt.show()

def plot_posterior_hist(fit, count, f_hist, para, key=None):
    arviz.rcParams["plot.density_kind"] = "hist"
    arviz.plot_density(fit,var_names=para)
    plt.savefig(f_hist/"experiment{}_{}.png".format(count, key), dpi=300)
    plt.show()

def plot_posterior_density(fit, count, f_dens, para, key=None):
    arviz.rcParams["plot.density_kind"] = "kde"
    arviz.plot_density(fit,var_names=para, shade=0.5, hdi_prob=0.997, textsize=30)
    plt.tight_layout()
    plt.savefig(f_dens/"experiment{}_{}.png".format(count, key), dpi=300)
    plt.show()

def plot_posterior_trace(fit, count, f_trace, para, key=None):
    arviz.plot_trace(fit,var_names=para)
    plt.tight_layout()
    plt.savefig(f_trace/"experiment{}_{}.png".format(count, key), dpi=300)
    plt.show()

def prepare_pca_data(y_grid, X_grid):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_grid)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=X_grid.index)
    for var in ["L", "a", "b"]:
        pca_df[var] = y_grid[var]

    return pca_df, scaler, pca


def create_pca_visualization_contour_only(pca_df, target_var, output_folder):
    from scipy.interpolate import griddata

    var_min = pca_df[target_var].min()
    var_max = pca_df[target_var].max()
    var_range = var_max - var_min
    var_min_extended = var_min - var_range * 0.6
    var_max_extended = var_max + var_range * 0.6
    level = np.linspace(var_min_extended, var_max_extended, 15)

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

    title = f"{target_var} Contour plot"
    file_pass = output_folder / f"{target_var}_contour.png"

    plt.figure(figsize=(6, 4.5))
    cont = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='gray', levels=level, linewidths=0.5, alpha=0.6)
    reduced_levels = level[::3]
    cont_reduced = plt.contour(PC1_mesh, PC2_mesh, Var_mesh, colors='black', levels=reduced_levels, linewidths=1.0)
    try:
        plt.clabel(cont_reduced, fmt='%.1f', fontsize=10, inline=True)
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
    plt.xlim(-2.5, 3)
    plt.ylim(-2, 3)
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
    var_min_extended = var_min - var_range * 0.6
    var_max_extended = var_max + var_range * 0.6
    level = np.linspace(var_min_extended, var_max_extended, 15)

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
        plt.clabel(cont_reduced, fmt='%.1f', fontsize=10, inline=True)
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
    plt.xlim(-2.5, 3)
    plt.ylim(-2, 3)

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

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    pca_df_pred = pca_df.copy()
    pca_df_pred[f"{target_var}_pred"] = pred_values

    var_min = pca_df[target_var].min()
    var_max = pca_df[target_var].max()
    var_range = var_max - var_min
    var_min_extended = var_min - var_range * 0.6
    var_max_extended = var_max + var_range * 0.6
    level = np.linspace(var_min_extended, var_max_extended, 15)

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
        plt.clabel(cont_reduced, fmt='%.1f', fontsize=10, inline=True)
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
    plt.xlim(-2.5, 3)
    plt.ylim(-2, 3)

    plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c="red", s=150, clip_on=False, zorder=20)

    if len(current_pca) > N_first:
        plt.scatter(current_pca[N_first:, 0], current_pca[N_first:, 1], c="blue", s=150, clip_on=False, zorder=20)

    plt.scatter(target_pca[0], target_pca[1], c="lime", s=150, clip_on=False,
                edgecolors='black', linewidths=2, marker='*', zorder=20)
    plt.tight_layout()
    plt.savefig(file_pass, dpi=300)
    plt.close()

def save_pca_contribution_ratio(pca_model, X_grid, output_folder):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    workspace_root = output_folder
    while workspace_root.name not in ['02_pigment', 'pigment']:
        workspace_root = workspace_root.parent
        if workspace_root == workspace_root.parent:
            workspace_root = Path.cwd()
            break

    pca_contribution_folder = workspace_root / 'result_PCA_contour_only'
    pca_contribution_folder.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_grid)
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    contribution_df = pd.DataFrame({
        'Principal component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Explained variance ratio': explained_variance_ratio,
        'Cumulative explained variance ratio': cumulative_variance_ratio
    })

    output_path = pca_contribution_folder / 'pca_contribution_ratio.csv'
    contribution_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_display = min(10, len(explained_variance_ratio))
    x_pos = np.arange(n_display)
    ax1.bar(x_pos, explained_variance_ratio[:n_display], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal component', fontsize=12)
    ax1.set_ylabel('Explained variance ratio', fontsize=12)
    ax1.set_title('Explained variance ratio by principal component', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'PC{i+1}' for i in range(n_display)])
    ax1.grid(axis='y', alpha=0.3)

    for i, v in enumerate(explained_variance_ratio[:n_display]):
        ax1.text(i, v + 0.01, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)

    ax2.plot(range(1, n_display+1), cumulative_variance_ratio[:n_display],
             marker='o', linewidth=2, markersize=6, color='darkred')
    ax2.set_xlabel('Number of principal components', fontsize=12)
    ax2.set_ylabel('Cumulative explained variance ratio', fontsize=12)
    ax2.set_title('Cumulative explained variance ratio', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(1, n_display+1))
    ax2.set_xticklabels([f'PC{i+1}' for i in range(n_display)])
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80%')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95%')
    ax2.legend(loc='lower right')

    for i, v in enumerate(cumulative_variance_ratio[:n_display]):
        ax2.text(i+1, v + 0.01, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    graph_path = pca_contribution_folder / 'pca_contribution_ratio.png'
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    n_components_to_save = min(5, len(explained_variance_ratio))
    loadings = pca_full.components_[:n_components_to_save].T * np.sqrt(pca_full.explained_variance_[:n_components_to_save])

    feature_names = X_grid.columns.tolist()
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(n_components_to_save)],
        index=feature_names
    )

    loadings_path = pca_contribution_folder / 'pca_loadings.csv'
    loadings_df.to_csv(loadings_path, encoding='utf-8-sig')

    print(f'\nSaved PCA contribution analysis results:')
    print(f'  - {output_path}')
    print(f'  - {graph_path}')
    print(f'  - {loadings_path}')
    print(f'\nPC1 explained variance ratio: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)')
    print(f'PC2 explained variance ratio: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)')
    print(f'PC1+PC2 cumulative explained variance ratio: {cumulative_variance_ratio[1]:.4f} ({cumulative_variance_ratio[1]*100:.2f}%)')
