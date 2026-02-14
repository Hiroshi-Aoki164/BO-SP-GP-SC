#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel execution script for BO_SP_GPR_SC.py (for 2variable)
Multiprocessing support on Windows
"""

#%%
import matplotlib
matplotlib.use('Agg')

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
import support_function as sp
import multiprocessing as mp
import warnings
import os
warnings.filterwarnings("ignore")

#%%
def process_single_seed(seed_num, shared_data):
    try:
        print(f"Process {os.getpid()}: Seed {seed_num} started")

        X_grid = shared_data['X_grid']
        y_grid = shared_data['y_grid']
        N_grid = shared_data['N_grid']
        f_result = shared_data['f_result']
        count_num = shared_data['count_num']
        N_first = shared_data['N_first']
        iter_ = shared_data['iter_']
        warmup = shared_data['warmup']
        pca_df = shared_data['pca_df']
        scaler = shared_data['scaler']
        pca_model = shared_data['pca_model']

        f_model = Path("./model")
        with open(f_model/'model_SP_GPR_SC.pickle', mode='br') as fi:
            model = pickle.load(fi)

        y_grid_original = shared_data['y_grid_original']
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
        f_plot, f_hist, f_dens, f_dens2, f_trace = sp.make_folder_BO_SP_GPR(seed_num, f_result)

        initial_indices = X_act.index.tolist()

        target_num = y_target_index

        f_name = f_result/"seed_{}".format(seed_num)
        f_pca_seed = f_name / "PCA"
        f_pca_seed.mkdir(parents=True, exist_ok=True)

        D = len(X_act.T)

        a_mean, b_mean = sp.cal_coef(X_act, y_act.iloc[:,0])
        a_sd = np.full(D, 4)
        b_sd = 4

        for count in range(0,count_num):
            N = len(X_act)
            data = {
                    "N" : N,
                    "D" : D,
                    "X" : X_act,
                    "y" : y_act.iloc[:,0],
                    "N_grid" : N_grid,
                    "X_grid" : X_grid,
                    "a_mean" : a_mean,
                    "a_sd" : a_sd,
                    "b_mean" : b_mean,
                    "b_sd" : b_sd
              }

            fit = model.sampling(data = data,
                            iter = iter_,#2000
                            warmup = warmup,
                            seed = 123,
                            chains = 1,
                            n_jobs=1
            )

            para = ["rho","alpha","sd_e","a","b"]
            sp.plot_posterior_forest(fit,count,f_plot,para)
            sp.plot_posterior_hist(fit,count,f_hist,para)
            sp.plot_posterior_density(fit,count,f_dens,para)
            para = ["a"]
            sp.plot_posterior_density(fit,count,f_dens2,para)
            sp.plot_posterior_trace(fit,count,f_trace,para)

            summary = arviz.summary(fit)
            print(
                f"Process {os.getpid()}: Seed {seed_num}, Experiment {count}: count of r_hat<=1.1: "
                + str(sum(summary["r_hat"] <= 1.10))
            )

            samples = fit.extract()

            EI = sp.cal_EI_minimize_error(y_act, y_target, samples)

            y_ave = samples["y_new"].mean(0)

            X_act, y_act = sp.add_act_data_BO(y_act, X_act, EI, X_grid, y_grid)

            current_indices = X_act.index.tolist()

            sp.create_pca_visualization_with_pred_values(pca_df, scaler, pca_model, X_grid, y_ave,
                                                         "Tb (K)", seed_num, count, N_first,
                                                         target_num, initial_indices, current_indices, f_pca_seed)

            a_mean, b_mean = sp.cal_coef(X_act, y_act.iloc[:,0])

            if y_target.iloc[0,0] - y_act.iloc[-1,0] == 0:
                print(f"Process {os.getpid()}: Seed {seed_num}: finished (target reached)")
                brank_data = pd.DataFrame(data = y_target.iloc[0,0]*np.ones(count_num-count-1), columns=y_act.columns)
                y_act = pd.concat([y_act, brank_data])
                break

        pd.concat([y_target, y_act]).to_csv((f_hist.parent)/"act_data.csv", encoding="shift-jis")

        result_dict = {
            "y_target": y_target.values[0].tolist(),
            "y_initial": y_act.iloc[:N_first,:].iloc[:,0].tolist(),
            "y_experiment": y_act.iloc[N_first:,:].iloc[:,0].tolist()
        }

        error_act = pd.DataFrame(abs(y_act - y_target.values))
        result_dict["initial_error"] = error_act.iloc[:N_first,:].iloc[:,0].tolist()
        result_dict["experiment_error"] = error_act.iloc[N_first:,:].iloc[:,0].tolist()

        print(f"Process {os.getpid()}: Seed {seed_num} completed")

        return seed_num, result_dict

    except Exception as e:
        print(f"Process {os.getpid()}: Seed {seed_num} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return seed_num, None

def main():
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

    data_raw = data_raw.loc[:,["Tb (K)", "TPSA", "log(Weight)"]]

    y_grid = data_raw.loc[:,["Tb (K)"]]
    X_grid = data_raw.iloc[:,1:]

    N_grid = len(X_grid)

    y_grid_original = y_grid.copy()
    y_grid = y_grid.reset_index(drop=True)
    X_grid = X_grid.reset_index(drop=True)

    pca_input_data = pd.concat([y_grid, X_grid], axis=1)
    pca_df, scaler, pca_model = sp.prepare_pca_data(pca_input_data)

    f_result = Path("./result/BO(SP_GPR_SC)")
    f_result.mkdir(parents=True, exist_ok=True)

    f_contour_only = Path("./result_PCA_contour_only")
    f_contour_only.mkdir(parents=True, exist_ok=True)
    sp.create_pca_visualization_contour_only(pca_df, "Tb (K)", f_contour_only, pca_model)
    f_model = Path(("./model"))
    f_model.mkdir(parents=True, exist_ok=True)

    seed_num_all = 20
    count_num = 20
    N_first = 3
    iter_ = 1000
    warmup = 200

    try:
        with open(f_model/'model_SP_GPR_SC.pickle', mode='br') as fi:
            model = pickle.load(fi)
        print("Loaded existing model")
    except FileNotFoundError:
        print("Building model...")
        model = StanModel("stan/SP_GPR_SC.stan")
        with open(f_model/'model_SP_GPR_SC.pickle', mode='wb') as fo:
            pickle.dump(model, fo)
        print("Built and saved model")

    shared_data = {
        'X_grid': X_grid,
        'y_grid': y_grid,
        'N_grid': N_grid,
        'f_result': f_result,
        'count_num': count_num,
        'N_first': N_first,
        'iter_': iter_,
        'warmup': warmup,
        'y_grid_original': y_grid_original,
        'pca_df': pca_df,
        'scaler': scaler,
        'pca_model': pca_model
    }

    num_processes = max(1, min(mp.cpu_count() - 1, seed_num_all))
    print(f"Parallel workers: {num_processes} processes")
    print(f"Total CPU cores: {mp.cpu_count()}")

    Y_end_all = {
        "target": {},
        "initial": {},
        "experiment": {},
        "initial_error": {},
        "experiment_error": {}
    }

    print("Starting parallel processing...")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_seed,
                              [(seed_num, shared_data) for seed_num in range(seed_num_all)])

    print("Parallel processing completed. Merging results...")

    for seed_num, result in results:
        if result is not None:
            Y_end_all["target"][seed_num] = result["y_target"]
            Y_end_all["initial"][seed_num] = result["y_initial"]
            Y_end_all["experiment"][seed_num] = result["y_experiment"]
            Y_end_all["initial_error"][seed_num] = result["initial_error"]
            Y_end_all["experiment_error"][seed_num] = result["experiment_error"]
        else:
            print(f"Seed {seed_num} result is invalid")

    print("Saving results...")

    with open(f_result/'result_all.pickle', mode='wb') as fo:
        pickle.dump(Y_end_all, fo)

    pd.DataFrame(Y_end_all["target"]).to_csv(f_result/"y_target.csv", encoding="shift-jis")
    pd.DataFrame(Y_end_all["initial"]).to_csv(f_result/"y_initial.csv", encoding="shift-jis")
    pd.DataFrame(Y_end_all["experiment"]).to_csv(f_result/"y_experiment.csv", encoding="shift-jis")
    pd.DataFrame(Y_end_all["initial_error"]).to_csv(f_result/"initial_error.csv", encoding="shift-jis")
    pd.DataFrame(Y_end_all["experiment_error"]).to_csv(f_result/"experiment_error.csv", encoding="shift-jis")

    print(f"All runs completed. Results saved to {f_result}.")

if __name__ == "__main__":
    # Windows support
    mp.freeze_support()
    main()
