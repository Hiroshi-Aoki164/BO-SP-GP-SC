import pandas as pd
import numpy as np
import colour as cs
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def cal_R(Rg, Sm, X, R_inf_m):
    Rg_Rinf = (-R_inf_m).add(Rg.values, axis = 1)
    Rg_Rinf2 = (-1/R_inf_m).add(Rg.values, axis = 1)
    Rinf_Rinf = 1/R_inf_m -R_inf_m
    SX = Sm

    bunsi = 1/R_inf_m * Rg_Rinf - R_inf_m * Rg_Rinf2 * np.exp(SX * Rinf_Rinf)
    bunbo = Rg_Rinf - Rg_Rinf2 * np.exp(SX * Rinf_Rinf)
    Rkm = bunsi / bunbo

    return Rkm

def R_to_XYZ(R, cmfs, illuminant):
    XYZ = pd.DataFrame(columns = ["X", "Y", "Z"])
    for name in R.T.columns:
        sd = cs.SpectralDistribution(R.T.loc[:,name])
        XYZ.loc[name,:] = cs.sd_to_XYZ(sd, cmfs, illuminant)/100
    return XYZ


def make_grid_recipe(total_amount):
    x1 = np.arange(0, 100, 5)
    x2 = np.arange(0, 100, 5)
    x3 = np.arange(0, 100, 5)
    x4 = np.arange(0, 100, 5)
    x5 = np.arange(0, 100, 5)

    X1, X2, X3, X4, X5 = np.meshgrid(x1, x2, x3, x4, x5)

    recipe = np.concatenate([X1.reshape(-1,1),X2.reshape(-1,1),
                         X3.reshape(-1,1),X4.reshape(-1,1),
                         X5.reshape(-1,1)],axis=1)

    recipe = pd.DataFrame(recipe, columns =
                      ["Bengala","Inari yellow clay","Tahara white clay",
                       "Lascaux black","Italian green clay"])

    recipe["Blue gray powder"] = total_amount - recipe.sum(1)
    recipe = recipe[recipe["Blue gray powder"] >= 0]
    return recipe


illuminant = cs.SDS_ILLUMINANTS['D65']
cmfs = cs.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

K = pd.read_excel('K_S_values.xlsx', index_col=0, sheet_name = "K")
S = pd.read_excel('K_S_values.xlsx', index_col=0, sheet_name = "S")
R_blank = pd.read_excel('R_blank.xlsx', index_col=0)

R_inf = 1 + K/S - ((K/S)**2  + 2*K/S)**0.5

recipe = pd.DataFrame()
for i in range(10,101,10):
    recipe = pd.concat([recipe,make_grid_recipe(total_amount=i)])

recipe = recipe.reset_index(drop=True)

recipe = recipe.sample(n=400, random_state=42).reset_index(drop=True)

Rg = R_blank.loc[["white_blank"],:]
Km = recipe.dot(K)/100
Sm = recipe.dot(S)/100

r_sum = recipe.sum(axis=1)
R_inf_m = recipe.dot(R_inf).div(r_sum.values, axis=0)

X = 1

Rkm = cal_R(Rg, Sm, X, R_inf_m)

XYZ = R_to_XYZ(Rkm, cmfs, illuminant)
RGB = cs.XYZ_to_sRGB(XYZ)
RGB = pd.DataFrame(RGB, columns = ["R", "G", "B"])
Lab = cs.XYZ_to_Lab(XYZ)
Lab = pd.DataFrame(Lab, columns = ["L", "a", "b"])

output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)

recipe.to_csv(os.path.join(output_dir, "recipe.csv"), encoding="shift-jis")
RGB.to_csv(os.path.join(output_dir, "RGB.csv"), encoding="shift-jis")
Lab.to_csv(os.path.join(output_dir, "Lab.csv"), encoding="shift-jis")
