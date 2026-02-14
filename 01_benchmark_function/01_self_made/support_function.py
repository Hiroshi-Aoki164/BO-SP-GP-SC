#%%
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression
import math
import arviz

def cal_ave(samples):
    y_ave = pd.DataFrame(samples["y_new"].mean(axis=0))
    return y_ave

def cal_sd(samples):
    y_sd = pd.DataFrame(samples["y_new"].std(axis=0))
    return y_sd

def cal_EI(y_act, samples):
    # EI(Expected Improvement)
    Y_max = max(y_act)
    improvement = samples["y_new"] - Y_max
    improvement = np.maximum(improvement, 0)
    EI = pd.DataFrame(improvement.mean(axis=0))
    return EI

def cal_PI(y_act, samples):
    # PI(Probability of Improvement)
    Y_max = max(y_act)
    PI = pd.DataFrame((samples["y_new"] > Y_max).mean(axis=0))
    # PI = pd.DataFrame((samples["y_new"] < Y_max).mean(axis=0))
    return PI

def add_act_data(X_act, N_first, EI, X_grid, eq):
    rowname = X_act.index[0:].tolist()[(N_first):]
    EI.loc[rowname,:] = -10
    select_col = EI[EI[0] == EI[0].max()].index[0]
    print(select_col)
    X_act = pd.concat([X_act,X_grid.loc[[select_col],:]])
    if eq == 1:
        y_act = self_made(X_act["X1"],X_act["X2"])
    if eq == 2:
        y_act = brain_hoo(X_act["X1"],X_act["X2"])
    if eq == 3:
        y_act = cosines(X_act["X1"],X_act["X2"])

    return X_act, y_act

def make_z_y(y):
    nn = int(len(y)**0.5)
    z_y = pd.DataFrame()
    for i in range(0,nn):
        for j in range(0,nn):
            z_y.loc[i,j] = y.loc[i*nn+j,0]
    return z_y

def cal_coef(X_act, y_act):
    model_lr = LinearRegression(fit_intercept=False)
    model_lr.fit(X_act, y_act)
    coef = model_lr.coef_
    b = model_lr.intercept_
    return coef,b

def self_made(X1,X2):
    Y = 2*X1 + 2*X2 - abs((X1-2)*(X1-8)) - abs((X2-8)*(X2-2))
    return Y

def cosines(X1,X2):
    u = 1.6*X1 - 0.5
    v = 1.6*X2 - 0.5
    pi = np.pi
    Y = 1 - (u**2 + v**2 - 0.3*np.cos(3*pi*u) - 0.3*np.cos(3*pi*v)+0.7)
    return Y

def brain_hoo(X1,X2):
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)

    Y = a*(X2 - b*X1**2 + c*X1 - r)**2 + s*(1-t)*np.cos(X1) + s

    return Y

def plt_data(start_x1, stop_x1, start_x2, stop_x2, num,eq):
    x1 = np.linspace(start_x1, stop_x1, num)
    x2 = np.linspace(start_x2, stop_x2, num)
    X1, X2 = np.meshgrid(x1, x2)
    if eq == 1:
        Y = self_made(X1,X2)
    if eq == 2:
        Y = brain_hoo(X1,X2)
    if eq == 3:
        Y = cosines(X1,X2)
    return X1,X2,Y

def plot_contour(X1, X2, Y, title, file_pass, level=None, X_first=None, N_first=None):
    plt.figure(figsize=(6,4.5))
    cont = plt.contour(X1,X2,Y,colors='black', levels=level)
    plt.clabel(cont, fmt='%d', fontsize=12)
    plt.contourf(X1,X2,Y,cmap='coolwarm', alpha=0.5, levels=level)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('X$_{1}$', fontsize=18, labelpad=-10)
    plt.ylabel('X$_{2}$', fontsize=18, labelpad=-15)
    plt.title(title, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    if N_first != None:
        plt.scatter(X_first.iloc[0:N_first,0],X_first.iloc[0:N_first,1], c="red",clip_on=False)
        plt.scatter(X_first.iloc[N_first:-1,0],X_first.iloc[N_first:-1,1], c="b",s=60,clip_on=False)
    plt.tight_layout()
    plt.savefig(file_pass, dpi=300)
    plt.show()

def make_grid(X1, X2, Y):
    colum_n = ["X1","X2"]
    X_grid_ = np.concatenate([X1.reshape(-1,1), X2.reshape(-1,1)],axis=1)
    X_grid = pd.DataFrame(data = X_grid_, columns=colum_n)
    y_grid = pd.DataFrame(data = Y.reshape(-1,1))

    return X_grid,y_grid

def make_folder_BO_SP_GPR(seed_num, f_result):
    f_name = f_result/"seed_{}".format(seed_num)
    f_plot = f_name/"plot"
    f_hist = f_name/"hist"
    f_dens = f_name/"dens"
    f_dens2 = f_name/"dens2"
    f_trace = f_name/"trace"
    f_kakutoku = f_name/"kakutoku"
    f_yave = f_name/"yave"
    f_ysd = f_name/"ysd"

    f_plot.mkdir(parents=True, exist_ok=True)
    f_hist.mkdir(parents=True, exist_ok=True)
    f_dens.mkdir(parents=True, exist_ok=True)
    f_dens2.mkdir(parents=True, exist_ok=True)
    f_trace.mkdir(parents=True, exist_ok=True)
    f_kakutoku.mkdir(parents=True, exist_ok=True)
    f_yave.mkdir(parents=True, exist_ok=True)
    f_ysd.mkdir(parents=True, exist_ok=True)

    return f_plot, f_hist, f_dens, f_dens2, f_trace, f_kakutoku, f_yave, f_ysd

def make_folder_BO_GPR(seed_num, f_result):
    f_name = f_result/"seed_{}".format(seed_num)
    f_plot = f_name/"plot"
    f_hist = f_name/"hist"
    f_dens = f_name/"dens"
    f_trace = f_name/"trace"
    f_kakutoku = f_name/"kakutoku"
    f_yave = f_name/"yave"
    f_ysd = f_name/"ysd"

    f_plot.mkdir(parents=True, exist_ok=True)
    f_hist.mkdir(parents=True, exist_ok=True)
    f_dens.mkdir(parents=True, exist_ok=True)
    f_trace.mkdir(parents=True, exist_ok=True)
    f_kakutoku.mkdir(parents=True, exist_ok=True)
    f_yave.mkdir(parents=True, exist_ok=True)
    f_ysd.mkdir(parents=True, exist_ok=True)

    return f_plot, f_hist, f_dens, f_trace, f_kakutoku, f_yave, f_ysd

def make_folder_MLR(seed_num, f_result):
    f_name = f_result/"seed_{}".format(seed_num)
    f_ypred = f_name/"ypred"

    f_ypred.mkdir(parents=True, exist_ok=True)

    return f_ypred


def plot_posterior_forest(fit,count,f_plot,para):
    arviz.plot_forest(fit,var_names=para)
    plt.savefig(f_plot/"experiment_{}.png".format(count), dpi=300)
    plt.show()

def plot_posterior_hist(fit,count,f_hist,para):
    arviz.rcParams["plot.density_kind"] = "hist"
    arviz.plot_density(fit,var_names=para)
    plt.savefig(f_hist/"experiment_{}.png".format(count), dpi=300)
    plt.show()

def plot_posterior_density(fit,count,f_dens,para):
    arviz.rcParams["plot.density_kind"] = "kde"
    arviz.plot_density(fit,var_names=para, shade=0.5, hdi_prob=0.997, textsize=30)
    plt.tight_layout()
    plt.savefig(f_dens/"experiment_{}.png".format(count), dpi=300)
    plt.show()

def plot_posterior_trace(fit,count,f_trace,para):
    arviz.plot_trace(fit,var_names=para)
    plt.tight_layout()
    plt.savefig(f_trace/"experiment_{}.png".format(count), dpi=300)
    plt.show()

def generate_initial_data(N_first, seed, x1_range, x2_range):
    np.random.seed(seed)

    data_ = []

    for _ in range(N_first):
        x1 = np.random.uniform(x1_range[0], x1_range[1])
        x2 = np.random.uniform(x2_range[0], x2_range[1])
        point = np.array([x1, x2])
        data_.append(point)

    return np.array(data_)
