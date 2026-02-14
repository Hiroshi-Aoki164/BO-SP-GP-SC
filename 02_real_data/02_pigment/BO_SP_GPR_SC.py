#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel execution script for BO_SP_GPR_SC.py
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
        b_act = shared_data['b_act']
        pca_df = shared_data['pca_df']
        scaler = shared_data['scaler']
        pca_model = shared_data['pca_model']

        f_model = Path("./model")
        model_dict = {}
        for key in ["L", "a", "b"]:
            with open(f_model/f'model_SP_GPR_SC_{key}.pickle', mode='br') as fi:
                model_dict[key] = pickle.load(fi)

        target_num = 42
        y_target = y_grid.iloc[target_num]
        x_target = X_grid.iloc[target_num]

        np.random.seed(seed_num)
        X_act = X_grid.sample(N_first)
        y_act = y_grid.loc[X_act.index,:]

        f_plot, f_hist, f_dens, f_dens2, f_trace = sp.make_folder_BO_SP_GPR(seed_num, f_result)

        f_name = f_result/"seed_{}".format(seed_num)
        f_pca_base = f_name / "PCA"
        f_pca_L_seed = f_pca_base / "result_PCA_visualization_L"
        f_pca_a_seed = f_pca_base / "result_PCA_visualization_a"
        f_pca_b_seed = f_pca_base / "result_PCA_visualization_b"
        f_pca_L_seed.mkdir(parents=True, exist_ok=True)
        f_pca_a_seed.mkdir(parents=True, exist_ok=True)
        f_pca_b_seed.mkdir(parents=True, exist_ok=True)

        initial_indices = X_act.index.tolist()

        D = len(X_act.T)

        a_mean_dict = {}
        a_sd_dict = {}
        b_mean_dict = {}
        b_sd_dict = {}

        for key in ["L", "a", "b"]:
            a_mean_dict[key], b_mean_dict[key] = sp.cal_coef(X_act, y_act[key])
            a_sd_dict[key] = np.full(D, 4)
            b_sd_dict[key] = 4

        for count in range(0,count_num):
            N = len(X_act)
            samples = {}
            for i, key in enumerate(["L","a","b"]):
                print(f"Process {os.getpid()}: Seed {seed_num}, Experiment {count}, {key}")

                data = {
                    "N" : N,
                    "D" : D,
                    "X" : X_act,
                    "y" : y_act[key],
                    "N_grid" : N_grid,
                    "X_grid" : X_grid,
                    "a_mean" : a_mean_dict[key],
                    "a_sd" : a_sd_dict[key],
                    "b_mean" : b_act[key],
                    "b_sd" : b_sd_dict[key],
                    }

                fit = model_dict[key].sampling(data = data,
                            iter = iter_,#2000
                            warmup = warmup,
                            seed = 123,
                            chains = 1,
                            n_jobs=1
                            )

                samples[key] = fit.extract()

                para = ["rho","alpha","sd_e","a","b"]
                sp.plot_posterior_forest(fit, count, f_plot, para, key)
                sp.plot_posterior_hist(fit, count, f_hist, para, key)
                sp.plot_posterior_density(fit, count, f_dens, para, key)
                sp.plot_posterior_trace(fit, count, f_trace, para, key)
                para = ["a"]
                sp.plot_posterior_density(fit, count, f_dens2, para, key)

                summary = arviz.summary(fit)
                print(
                    f"Process {os.getpid()}: Seed {seed_num}, Experiment {count}, {key}: "
                    + "count(r_hat <= 1.1): "
                    + str(sum(summary["r_hat"] <= 1.10))
                )

            #%%
            Y_pred = {}
            for key in samples.keys():
                Y_pred[key] = samples[key]["y_new"]

            EI = sp.cal_EI_Lab(y_act, y_target, Y_pred)

            L_ave = Y_pred["L"].mean(0)
            a_ave = Y_pred["a"].mean(0)
            b_ave = Y_pred["b"].mean(0)
            Lab_ave = pd.DataFrame([L_ave, a_ave, b_ave], index = ["L","a","b"]).T

            X_act, y_act = sp.add_act_data_BO(y_act, X_act, EI, X_grid, y_grid)

            current_indices = X_act.index.tolist()

            sp.create_pca_visualization_with_pred_values(pca_df, scaler, pca_model, X_grid, L_ave,
                                                         "L", seed_num, count, N_first,
                                                         target_num, initial_indices, current_indices, f_pca_L_seed)
            sp.create_pca_visualization_with_pred_values(pca_df, scaler, pca_model, X_grid, a_ave,
                                                         "a", seed_num, count, N_first,
                                                         target_num, initial_indices, current_indices, f_pca_a_seed)
            sp.create_pca_visualization_with_pred_values(pca_df, scaler, pca_model, X_grid, b_ave,
                                                         "b", seed_num, count, N_first,
                                                         target_num, initial_indices, current_indices, f_pca_b_seed)

            for key in ["L", "a", "b"]:
                a_sd_dict[key] = np.full(D, 4)
                b_sd_dict[key] = 4
                a_mean_dict[key], b_mean_dict[key] = sp.cal_coef(X_act, y_act[key])

        f_name = f_result/"seed_{}".format(seed_num)
        f_name.mkdir(parents=True, exist_ok=True)

        y_combined = pd.concat([y_target.to_frame().T, y_act], axis=0)
        y_combined.to_csv(f_name/'act_data_y.csv', encoding='shift-jis')

        x_combined = pd.concat([x_target.to_frame().T, X_act], axis=0)
        x_combined.to_csv(f_name/'act_data_x.csv', encoding='shift-jis')

        print(f"Process {os.getpid()}: Seed {seed_num} completed")

        return seed_num, {
            "y_act": y_act,
            "y_target": y_target
        }

    except Exception as e:
        print(f"Process {os.getpid()}: Seed {seed_num} encountered an error: {str(e)}")
        import traceback
        traceback.print_exc()
        return seed_num, None

def main():
    f_data = Path("./data")
    y_grid = pd.read_csv(f_data/"Lab.csv", encoding="shift-jis", index_col=0)
    X_grid = pd.read_csv(f_data/"recipe.csv", encoding="shift-jis", index_col=0)/100
    N_grid = len(X_grid)

    y_grid = y_grid.reset_index(drop=True)
    X_grid = X_grid.reset_index(drop=True)

    pca_df, scaler, pca_model = sp.prepare_pca_data(y_grid, X_grid)

    sp.save_pca_contribution_ratio(pca_model, X_grid, Path("./result/BO(SP_GPR_SC)"))

    f_result = Path("./result/BO(SP_GPR_SC)")
    f_result.mkdir(parents=True, exist_ok=True)
    f_model = Path(("./model"))
    f_model.mkdir(parents=True, exist_ok=True)

    b_act = {"L": 96.46, "a": 2.21, "b": -0.97}

    seed_num_all = 20
    count_num = 20
    N_first = 3
    iter_ = 1000
    warmup = 200
    coef_num = 6

    model_dict = {}
    for key in ["L", "a", "b"]:
        try:
            with open(f_model/f'model_SP_GPR_SC_{key}.pickle', mode='br') as fi:
                model_dict[key] = pickle.load(fi)
            print(f"Loaded existing model (SP_GPR_SC_{key})")
        except FileNotFoundError:
            print(f"Building model (SP_GPR_SC_{key})...")
            model_dict[key] = StanModel(f"stan/SP_GPR_SC_{key}.stan")
            with open(f_model/f'model_SP_GPR_SC_{key}.pickle', mode='wb') as fo:
                pickle.dump(model_dict[key], fo)
            print(f"Built and saved model (SP_GPR_SC_{key})")

    shared_data = {
        'X_grid': X_grid,
        'y_grid': y_grid,
        'N_grid': N_grid,
        'f_result': f_result,
        'count_num': count_num,
        'N_first': N_first,
        'iter_': iter_,
        'warmup': warmup,
        'b_act': b_act,
        'pca_df': pca_df,
        'scaler': scaler,
        'pca_model': pca_model
    }

    num_processes = max(1, min(mp.cpu_count() - 1, seed_num_all))
    print(f"Parallel processes: {num_processes}")
    print(f"Total CPU count: {mp.cpu_count()}")

    Y_end_all = {}
    Y_end_all["experiment"] = {}
    Y_end_all["initial"] = {}
    Y_end_all["target"] = {}

    print("Starting parallel processing...")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_seed,
                              [(seed_num, shared_data) for seed_num in range(seed_num_all)])

    print("Parallel processing completed; merging results...")

    for seed_num, result in results:
        if result is not None:
            Y_end_all["experiment"][seed_num] = result["y_act"].iloc[N_first:,:]
            Y_end_all["initial"][seed_num] = result["y_act"].iloc[:N_first,:]
            Y_end_all["target"][seed_num] = result["y_target"]
        else:
            print(f"Seed {seed_num} result is invalid")

    print("Saving results...")
    with open(f_result/'Lab_action.pickle', mode='wb') as fo:
        pickle.dump(Y_end_all, fo)

    dE_to_target_pd = sp.calculate_dE_to_target(Y_end_all)
    dE_to_target_pd.to_csv(f_result/"dE_to_target.csv")

    print(f"All processing completed. Results saved to {f_result}.")

if __name__ == "__main__":
    # Windows support
    mp.freeze_support()
    main()




