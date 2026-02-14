#%%
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import support_function as sp

#%%
f_result = Path("./result/MLR")
f_result.mkdir(parents=True, exist_ok=True)

# Get grid data in mesh format
num = 11
X1, X2, Y = sp.plt_data(0, num-1, 0, num-1, num, eq=1)
X_grid, y_grid = sp.make_grid(X1, X2, Y)
N_grid = num*num

title = "actual y"
level = range(-35,41,5)
file_pass = "y_actual.png"
sp.plot_contour(X1, X2, Y, title, file_pass, level)

#%%
seed_num_all = 50
count_num = 20
N_first = 3
Y_end_all = pd.DataFrame()

for seed_num in range(0,seed_num_all):

    data_ = sp.generate_initial_data(N_first, seed_num, x1_range=[0, 4], x2_range=[0, 4])
    X_act = pd.DataFrame(data_, columns = X_grid.columns)
    y_act = sp.self_made(X_act["X1"],X_act["X2"])

    sigma_ave = pd.DataFrame()

    f_ypred = sp.make_folder_MLR(seed_num, f_result)

    for count in range(0,count_num):
        N = len(X_act)
        model_lr = LinearRegression(fit_intercept=False)
        model_lr.fit(X_act, y_act)
        y_pred = pd.DataFrame(model_lr.predict(X_grid))
        z_ypred = sp.make_z_y(y_pred)

        #%%
        X_act, y_act = sp.add_act_data(X_act, N_first, y_pred, X_grid, eq=1)

        title = "Experiment {}".format(count)
        level = range(-35,41,5)
        file_pass = f_ypred/"experiment_{}.png".format(count)
        sp.plot_contour(X1, X2, z_ypred, title, file_pass, level, X_act, N_first)

    y_act = y_act.reset_index(drop=True)
    Y_end_all = pd.concat([Y_end_all,y_act],axis=1)

Y_end_all.to_csv(f_result/"y_all.csv", encoding="shift-jis")
