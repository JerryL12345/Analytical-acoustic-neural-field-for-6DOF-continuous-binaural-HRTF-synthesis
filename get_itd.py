# fit_itd.py
import os
import pandas as pd
import numpy as np
from pysr import PySRRegressor

def spherical_deg_to_cartesian_unit(angles_deg: np.ndarray) -> np.ndarray:
    azimuth_rad = np.deg2rad(angles_deg[:, 0])
    elevation_rad = np.deg2rad(angles_deg[:, 1])
    cos_ele = np.cos(elevation_rad)
    x_pos = np.cos(azimuth_rad) * cos_ele
    y_pos = np.sin(azimuth_rad) * cos_ele
    z_pos = np.sin(elevation_rad)
    return np.column_stack([x_pos, y_pos, z_pos])

print("启动独立 ITD 物理方程演化...")
df = pd.read_csv("itd_training_data.csv")
angles_deg = np.column_stack([df["azimuth_deg"].values, df["elevation_deg"].values])
X = spherical_deg_to_cartesian_unit(angles_deg)
Y = df["itd_ms"].values

model = PySRRegressor(
    niterations=1500,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos"], 
    model_selection="accuracy",
    parallelism="multithreading", # 纯多线程，绝不死锁
    populations=20,
    population_size=40,
    maxsize=15, # 强制找极简方程
    verbosity=2,
)

model.fit(X, Y, variable_names=["x_pos", "y_pos", "z_pos"])

print("演化完成！请看上面的 Hall of Fame 表格。")
print("请把最下面那个最精确的公式记下来！")