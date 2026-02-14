#%%
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import support_function as sp

f_data = Path("./data")
data_raw = pd.read_csv(f_data/"curated_crit_descriptors_5188.csv", encoding="shift-jis", index_col=0)

data_raw = data_raw.replace('#DIV/0!', np.nan)
data_raw = data_raw.dropna(how='any')
data_raw[data_raw.columns] = data_raw[data_raw.columns].apply(pd.to_numeric, errors='coerce')

data_raw['log(Weight)'] = np.log10(data_raw['Weight'])
data_raw = data_raw.drop('Weight', axis=1)

data_raw['combined'] = data_raw[["TPSA", "log(Weight)"]].astype(str).apply(lambda row: ''.join(row), axis=1)

data_raw = data_raw.groupby("combined").mean()
data_raw = data_raw.reset_index(drop=True)

y_grid = data_raw.loc[:,["Tb (K)"]]
X_grid = data_raw.iloc[:,1:]

y_grid_original = y_grid.copy()
y_grid = y_grid.reset_index(drop=True)
X_grid = X_grid.reset_index(drop=True)

pca_input_data = pd.concat([y_grid, X_grid], axis=1)
pca_df, scaler, pca_model = sp.prepare_pca_data(pca_input_data)

#%%
f_result = Path("./result/MLR")
f_result.mkdir(parents=True, exist_ok=True)

f_contour_only = Path("./result_PCA_contour_only")
f_contour_only.mkdir(parents=True, exist_ok=True)
sp.create_pca_visualization_contour_only(pca_df, "Tb (K)", f_contour_only, pca_model)

#%%
seed_num_all = 20
count_num = 20
N_first = 3
Y_end_all = {}
Y_end_all["target"] = {}
Y_end_all["initial"] = {}
Y_end_all["experiment"] = {}
Y_end_all["initial_error"] = {}
Y_end_all["experiment_error"] = {}

Y_end_all["coef"] = {}
Y_end_all["select_data"] = {}

seed_range = range(0,seed_num_all)

for seed_num in seed_range:

    f_name = f_result/"seed_{}".format(seed_num)
    f_name.mkdir(parents=True, exist_ok=True)

    f_pca_seed = f_name / "PCA"
    f_pca_seed.mkdir(parents=True, exist_ok=True)

    y_grid_original_reset = y_grid_original.reset_index(drop=True)
    candidates_y_target = y_grid_original_reset[y_grid_original_reset.iloc[:,0] >= 400]
    y_target_index = candidates_y_target.sample(1, random_state=seed_num+1).index[0]
    y_target = y_grid.loc[[y_target_index]]
    y_target_value = y_grid_original_reset.iloc[y_target_index, 0]

    np.random.seed(seed_num)

    candidates_X_initial = y_grid_original_reset[y_grid_original_reset.iloc[:,0] <= 200]

    first_point_index = candidates_X_initial.sample(1, random_state=seed_num).index[0]
    selected_indices = [first_point_index]

    first_point_x = X_grid.loc[first_point_index].values
    candidates_indices = candidates_X_initial.index.tolist()
    candidates_indices.remove(first_point_index)

    distances = pd.Series(index=candidates_indices, dtype=float)
    for idx in candidates_indices:
        distances[idx] = np.linalg.norm(X_grid.loc[idx].values - first_point_x)

    similar_indices = distances.nsmallest(N_first - 1).index.tolist()
    selected_indices.extend(similar_indices)

    X_act = X_grid.loc[selected_indices]
    y_act = y_grid.loc[X_act.index,:]
    coef = pd.DataFrame(columns = [*(X_act.columns.tolist()),"intercept"])

    initial_indices = X_act.index.tolist()

    target_num = y_target_index

    for count in range(0,count_num):
        model = LinearRegression()
        model.fit(X_act, y_act)
        y_pred = pd.DataFrame(model.predict(X_grid))
        coef.loc[count,:] = np.append(model.coef_ ,model.intercept_)
        y_pred.index = y_grid.index
        prediction_error = pd.DataFrame(abs(y_pred - y_target.values))
        #%%
        X_act, y_act = sp.add_act_data_MLR(y_act, X_act, prediction_error, X_grid, y_grid)

        current_indices = X_act.index.tolist()

        sp.create_pca_visualization_with_model(pca_df, scaler, pca_model, X_grid, model,
                                               "Tb (K)", seed_num, count, N_first,
                                               target_num, initial_indices, current_indices, f_pca_seed)

    Y_end_all["target"][seed_num] = y_target.values[0].tolist()
    Y_end_all["initial"][seed_num] = y_act.iloc[:N_first,:].iloc[:,0].tolist()
    Y_end_all["experiment"][seed_num] = y_act.iloc[N_first:,:].iloc[:,0].tolist()

    dE_act = pd.DataFrame(abs(y_act - y_target.values))
    Y_end_all["initial_error"][seed_num] = dE_act.iloc[:N_first,:].iloc[:,0].tolist()
    Y_end_all["experiment_error"][seed_num] = dE_act.iloc[N_first:,:].iloc[:,0].tolist()

    Y_end_all["coef"][seed_num] = coef

    coef.to_csv(f_name/"coef.csv", encoding="shift-jis")
    pd.concat([y_target, y_act]).to_csv(f_name/"act_data.csv", encoding="shift-jis")

import pickle
with open(f_result/'Lab_action.pickle', mode='wb') as fo:
    pickle.dump(Y_end_all, fo)

pd.DataFrame(Y_end_all["target"]).to_csv(f_result/"y_target.csv", encoding="shift-jis")
pd.DataFrame(Y_end_all["initial"]).to_csv(f_result/"y_initial.csv", encoding="shift-jis")
pd.DataFrame(Y_end_all["experiment"]).to_csv(f_result/"y_experiment.csv", encoding="shift-jis")
pd.DataFrame(Y_end_all["initial_error"]).to_csv(f_result/"initial_error.csv", encoding="shift-jis")
pd.DataFrame(Y_end_all["experiment_error"]).to_csv(f_result/"experiment_error.csv", encoding="shift-jis")