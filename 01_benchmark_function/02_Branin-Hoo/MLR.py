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

# Build mesh grid data
start_x1 = -5
stop_x1 = 10
start_x2 = 0
stop_x2 = 15
num = 11
eq = 2
level = np.linspace(-100,540,15)

X1, X2, Y = sp.plt_data(start_x1,stop_x1,start_x2,stop_x2,num,eq=eq)
X_grid, y_grid = sp.make_grid(X1, X2, Y)

title = "actual y"
file_pass = "y_actual.png"
sp.plot_contour(X1, X2, Y, title, file_pass, level)

#%%
seed_num_all = 50
count_num = 20
Y_end_all = pd.DataFrame()

for seed_num in range(0,seed_num_all):
    N_first = 3

    data_ = sp.generate_initial_data(N_first, seed_num, x1_range=[5, 10], x2_range=[10, 15])

    X_act = pd.DataFrame(data_, columns = X_grid.columns)
    y_act = sp.brain_hoo(X_act["X1"],X_act["X2"])

    sigma_ave = pd.DataFrame()

    f_ypred = sp.make_folder_MLR(seed_num, f_result)

    for count in range(0,count_num):
        N = len(X_act)

        X_act_expanded = pd.concat([X_act,X_act**2,X_act["X1"]**3,X_act["X1"]**4],axis=1)  # polynomial terms only are assumed known
        X_act_expanded.columns = ["X1","X2","X1^2","X2^2","X1^3","X1^4"]
        X_grid_expanded = pd.concat([X_grid,X_grid**2,X_grid["X1"]**3,X_grid["X1"]**4],axis=1)
        X_grid_expanded.columns = ["X1","X2","X1^2","X2^2","X1^3","X1^4"]

        model_lr = LinearRegression()
        model_lr.fit(X_act_expanded, y_act)
        y_pred = pd.DataFrame(model_lr.predict(X_grid_expanded))
        z_ypred = sp.make_z_y(y_pred)

        #%%
        X_act, y_act = sp.add_act_data(X_act, N_first, y_pred, X_grid, eq=eq)

        title = "Experiment {}".format(count)
        file_pass = f_ypred/"experiment_{}.png".format(count)
        sp.plot_contour(X1, X2, z_ypred, title, file_pass, level, X_act, N_first)

    y_act = y_act.reset_index(drop=True)
    Y_end_all = pd.concat([Y_end_all,y_act],axis=1)

Y_end_all.to_csv(f_result/"y_all.csv", encoding="shift-jis")

