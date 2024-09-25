import joblib
import pandas as pd
import matplotlib.pyplot as plt
import ternary
from featuresselect import *

model_hardness = joblib.load("ML_model_H.pkl")
model_modulus = joblib.load("ML_model_M.pkl")

step_size = 0.01

df = pd.DataFrame(columns=["W", "Ta", "V", "Nb", "Cr"])

x_values = np.arange(0, 1/3 +step_size, step_size)

for x in x_values:

    remaining_sum = 1 - 3 * x

    if remaining_sum < 0:

        continue

    w_values = np.arange(0, remaining_sum, step_size)
    for W in w_values:
        Ta = remaining_sum - W
        if Ta < 0:
            continue

        df = df._append({"W": W, "Ta": Ta, "V": x, "Nb": x, "Cr": x}, ignore_index=True)

des = compute_descriptor(df)
des_H = des[['ave:num_f_valence', 'var:evaporation_heat', 'var:num_unfilled', 'var:vdw_radius_alvarez']]
des_M = des[['ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy']]

pressure = 0.281
bias = 0.811
flow = 0.015
df_pressure = pd.DataFrame(np.full((len(df),1), pressure), columns=['pressure'])
df_bias = pd.DataFrame(np.full((len(df),1), bias), columns=['bias'])
df_flow = pd.DataFrame(np.full((len(df),1), flow), columns=['flow'])
remaining_comp = pd.DataFrame(np.zeros((len(df),10)), columns=['Cu', 'Al', 'Fe', 'Zr', 'Co', 'Ni', 'Ti', 'Mo', 'Mn', 'Hf'])
df_comp = pd.concat([df_pressure, df_bias, df_flow, df, remaining_comp], axis=1)
df_comp_sorted = df_comp.reindex(columns=['pressure', 'bias', 'flow', 'Cu', 'Al', 'Fe', 'Zr', 'V', 'Co', 'Ni', 'Nb', 'Ti', 'Cr', 'Mo', 'Mn', 'W', 'Ta', 'Hf'])

df_all_H = pd.concat([df_comp_sorted, des_H], axis=1)
df_all_M = pd.concat([df_comp_sorted, des_M], axis=1)

H = model_hardness.predict(df_all_H)
M = model_modulus.predict(df_all_M)

result = {"H": H, "M": M, "H/E": H/M, "var:evaporation_heat": des_H['var:evaporation_heat']}

df_result = pd.DataFrame(result)

df_result = pd.concat([df_result, df_comp], axis=1)

#df_result.to_csv("analyse.csv", index=False)










