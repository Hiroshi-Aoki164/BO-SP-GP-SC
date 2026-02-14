#%%
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import support_function as sp

f_data = Path("./data/")
y_grid = pd.read_csv(f_data/"Lab.csv", encoding="shift-jis", index_col=0)
X_grid = pd.read_csv(f_data/"recipe.csv", encoding="shift-jis", index_col=0)/100

y_grid = y_grid.reset_index(drop=True)
X_grid = X_grid.reset_index(drop=True)

pca_df, scaler, pca = sp.prepare_pca_data(y_grid, X_grid)

#%%
f_result = Path("./result/MLR")
f_result.mkdir(parents=True, exist_ok=True)

f_contour_only = Path("./result_PCA_contour_only")
f_contour_only.mkdir(parents=True, exist_ok=True)
sp.create_pca_visualization_contour_only(pca_df, "L", f_contour_only)
sp.create_pca_visualization_contour_only(pca_df, "a", f_contour_only)
sp.create_pca_visualization_contour_only(pca_df, "b", f_contour_only)

#%%
seed_num_all = 20
count_num = 20
N_first = 3
Y_end_all = {}
Y_end_all["experiment"] = {}
Y_end_all["initial"] = {}
Y_end_all["target"] = {}

for seed_num in range(0,seed_num_all):
    target_num = 42
    y_target = y_grid.iloc[[target_num],:]
    x_target = X_grid.iloc[[target_num],:]
    Y_end_all["target"][seed_num] = y_target.iloc[0]

    np.random.seed(seed_num)
    X_act = X_grid.sample(N_first)
    y_act = y_grid.loc[X_act.index,:]

    f_name = f_result/"seed_{}".format(seed_num)
    f_name.mkdir(parents=True, exist_ok=True)

    f_pca_base = f_name / "PCA"
    f_pca_L_seed = f_pca_base / "result_PCA_visualization_L"
    f_pca_a_seed = f_pca_base / "result_PCA_visualization_a"
    f_pca_b_seed = f_pca_base / "result_PCA_visualization_b"
    f_pca_L_seed.mkdir(parents=True, exist_ok=True)
    f_pca_a_seed.mkdir(parents=True, exist_ok=True)
    f_pca_b_seed.mkdir(parents=True, exist_ok=True)

    initial_indices = X_act.index.tolist()

    for count in range(0,count_num):
        model_lr_L = LinearRegression()
        model_lr_L.fit(X_act, y_act["L"])
        L_pred = pd.DataFrame(model_lr_L.predict(X_grid))

        model_lr_a = LinearRegression()
        model_lr_a.fit(X_act, y_act["a"])
        a_pred = pd.DataFrame(model_lr_a.predict(X_grid))

        model_lr_b = LinearRegression()
        model_lr_b.fit(X_act, y_act["b"])
        b_pred = pd.DataFrame(model_lr_b.predict(X_grid))

        Lab_pred = pd.concat([L_pred, a_pred, b_pred],axis=1)
        Lab_pred.index = y_grid.index
        Lab_pred.columns = y_grid.columns
        dE = pd.DataFrame(((Lab_pred - y_target.values)**2).sum(axis=1)**0.5)

        #%%
        X_act, y_act = sp.add_act_data_MLR(y_act, X_act, dE, X_grid, y_grid)

        current_indices = X_act.index.tolist()

        sp.create_pca_visualization_with_model(pca_df, scaler, pca, X_grid, model_lr_L,
                                               "L", seed_num, count, N_first,
                                               target_num, initial_indices, current_indices, f_pca_L_seed)
        sp.create_pca_visualization_with_model(pca_df, scaler, pca, X_grid, model_lr_a,
                                               "a", seed_num, count, N_first,
                                               target_num, initial_indices, current_indices, f_pca_a_seed)
        sp.create_pca_visualization_with_model(pca_df, scaler, pca, X_grid, model_lr_b,
                                               "b", seed_num, count, N_first,
                                               target_num, initial_indices, current_indices, f_pca_b_seed)

    Y_end_all["experiment"][seed_num] = y_act.iloc[N_first:,:]
    Y_end_all["initial"][seed_num] = y_act.iloc[:N_first,:]

    y_combined = pd.concat([y_target, y_act], axis=0)
    y_combined.to_csv(f_name/'act_data_y.csv', encoding='shift-jis')

    x_combined = pd.concat([x_target, X_act], axis=0)
    x_combined.to_csv(f_name/'act_data_x.csv', encoding='shift-jis')

import pickle
with open(f_result/'Lab_action.pickle', mode='wb') as fo:
    pickle.dump(Y_end_all, fo)

dE_to_target_pd = sp.calculate_dE_to_target(Y_end_all)
dE_to_target_pd.to_csv(f_result/"dE_to_target.csv")
