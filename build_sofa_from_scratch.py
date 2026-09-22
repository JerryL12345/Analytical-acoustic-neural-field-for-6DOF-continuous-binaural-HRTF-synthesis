#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从公式与 PCA 权重“凭空”构建 SOFA 文件。

输入文件（同目录）：
1) pysr_pc_summary.csv
2) pca_weights.npz
3) AANF-HRTF.py（提供基础函数）

输出文件：
- AANF_Absolute_Math_360.sofa
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sympy as sp


def load_aanf_module(module_path: Path):
    """动态加载 AANF-HRTF.py，提取需要的基础函数。"""
    spec = importlib.util.spec_from_file_location("aanf_hrtf_dynamic", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础库文件: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_woodworth_itd(y_pos, head_radius=0.0875, c=343.0, fs=48000):
    """
    几何声学 ITD模型。
    返回 (left_delay_samples, right_delay_samples)。
    """
    theta_mid = np.arcsin(np.clip(np.abs(y_pos), -1.0, 1.0))
    delay_sec = (head_radius / c) * (theta_mid + np.sin(theta_mid))
    delay_samples = int(np.round(delay_sec * fs))

    if y_pos > 0:
        return 0, delay_samples
    return delay_samples, 0


def shift_with_zero_fill(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """先 np.roll，再把头部空缺补 0.0。"""
    if delay_samples <= 0:
        return signal

    out = np.roll(signal, delay_samples)
    out[:delay_samples] = 0.0
    return out


def main():
    base_dir = Path(__file__).resolve().parent

    csv_path = base_dir / "pysr_pc_summary.csv"
    npz_path = base_dir / "pca_weights.npz"
    aanf_path = base_dir / "AANF-HRTF.py"
    target_sofa_path = base_dir / "AANF_Absolute_Math_360.sofa"

    if not csv_path.exists():
        raise FileNotFoundError(f"缺少公式文件: {csv_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"缺少 PCA 权重文件: {npz_path}")
    if not aanf_path.exists():
        raise FileNotFoundError(f"缺少基础工具库: {aanf_path}")

    # ----------------------------
    # 步骤 1/工具：加载基础函数
    # ----------------------------
    aanf = load_aanf_module(aanf_path)
    ensure_2d_angles = aanf.ensure_2d_angles
    spherical_deg_to_cartesian_unit = aanf.spherical_deg_to_cartesian_unit
    magnitude_to_minimum_phase_hrir = aanf.magnitude_to_minimum_phase_hrir

    # ----------------------------
    # 步骤 2：加载 PCA 权重库
    # ----------------------------
    pca_pack = np.load(npz_path)
    pca_components = np.asarray(pca_pack["components"], dtype=np.float64)
    pca_mean = np.asarray(pca_pack["mean"], dtype=np.float64)
    fs = float(np.asarray(pca_pack["fs"]).reshape(-1)[0])
    n_samples = 256

    # ----------------------------
    # 步骤 3：构造 360 度纯数学空间网格
    # ----------------------------
    # 👇 关键参数：在这里修改步长（0.1 代表每隔 0.1 度生成一个点，精度）
    step = 0.001  
    
    azimuths = np.arange(0.0, 360.0, step, dtype=np.float64)
    num_points = len(azimuths) # 让总点数自动跟随步长计算 (比如 360 / 0.1 = 3600个点)
    elevations = np.zeros(num_points, dtype=np.float64)

    angles_left = ensure_2d_angles(azimuths, elevations)
    X_left = spherical_deg_to_cartesian_unit(angles_left)
    X_right = X_left.copy()
    X_right[:, 1] *= -1.0

    # ----------------------------
    # 步骤 4：动态解析 CSV 数学公式
    # ----------------------------
    df = pd.read_csv(csv_path)
    if "equation" not in df.columns:
        raise ValueError("CSV 中缺少 'equation' 列。")

    x0, x1, x2 = sp.symbols("x0 x1 x2")

    funcs = []
    for _, row in df.iterrows():
        expr_text = str(row["equation"]).strip()
        if not expr_text or expr_text.lower() == "nan":
            continue

        expr = sp.sympify(expr_text)
        fn = sp.lambdify((x0, x1, x2), expr, modules=["numpy"])
        funcs.append(fn)

    if len(funcs) == 0:
        raise ValueError("CSV 中未解析到可用公式。")

    z_left_cols = [
        np.asarray(fn(X_left[:, 0], X_left[:, 1], X_left[:, 2]), dtype=np.float64).reshape(-1)
        for fn in funcs
    ]
    z_right_cols = [
        np.asarray(fn(X_right[:, 0], X_right[:, 1], X_right[:, 2]), dtype=np.float64).reshape(-1)
        for fn in funcs
    ]

    z_left = np.column_stack(z_left_cols)
    z_right = np.column_stack(z_right_cols)

    # ----------------------------
    # 步骤 5：频域恢复与截断护盾
    # ----------------------------
    if z_left.shape[1] != pca_components.shape[0]:
        raise ValueError(
            "公式数量与 PCA 主成分维度不一致: "
            f"z_cols={z_left.shape[1]}, pca_components_rows={pca_components.shape[0]}"
        )

    pred_left_db = np.dot(z_left, pca_components) + pca_mean
    pred_right_db = np.dot(z_right, pca_components) + pca_mean

    pred_left_db = np.clip(pred_left_db, -100.0, 30.0)
    pred_right_db = np.clip(pred_right_db, -100.0, 30.0)

    # ----------------------------
    # 步骤 6：最小相位重建与 ITD 注入
    # ----------------------------
    hrir_left = magnitude_to_minimum_phase_hrir(pred_left_db, n_samples)
    hrir_right = magnitude_to_minimum_phase_hrir(pred_right_db, n_samples)

    for i in range(num_points):
        d_left, d_right = calculate_woodworth_itd(
            y_pos=X_left[i, 1],
            head_radius=0.0875,
            c=343.0,
            fs=fs,
        )

        hrir_left[i] = shift_with_zero_fill(hrir_left[i], d_left)
        hrir_right[i] = shift_with_zero_fill(hrir_right[i], d_right)

    # ----------------------------
    # 步骤 7：从底层构建 AES69 SOFA 文件
    # ----------------------------
    with h5py.File(target_sofa_path, "w") as f:
        # 1) Global Attributes
        f.attrs["Conventions"] = "SOFA"
        f.attrs["Version"] = "1.0"
        f.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        f.attrs["SOFAConventionsVersion"] = "1.0"
        f.attrs["DataType"] = "FIR"
        f.attrs["RoomType"] = "free field"

        # 2) 坐标系 Attributes（全局 + 数据集双保险）
        f.attrs["EmitterPositionType"] = "cartesian"
        f.attrs["EmitterPositionUnits"] = "meter"
        f.attrs["ListenerPositionType"] = "cartesian"
        f.attrs["ListenerPositionUnits"] = "meter"
        f.attrs["ReceiverPositionType"] = "cartesian"
        f.attrs["ReceiverPositionUnits"] = "meter"
        f.attrs["SourcePositionType"] = "spherical"
        f.attrs["SourcePositionUnits"] = "degree, degree, meter"

        # 3) SourcePosition: [360, 3]
        source_position = np.column_stack([
            azimuths,
            elevations,
            np.ones(num_points, dtype=np.float64) * 1.2,
        ])
        ds_source = f.create_dataset("SourcePosition", data=source_position, dtype=np.float64)
        ds_source.attrs["Type"] = "spherical"
        ds_source.attrs["Units"] = "degree, degree, meter"

        # 4) Data.IR: [360, 2, 256], float64
        data_ir = np.zeros((num_points, 2, n_samples), dtype=np.float64)
        data_ir[:, 0, :] = hrir_left
        data_ir[:, 1, :] = hrir_right
        f.create_dataset("Data.IR", data=data_ir, dtype=np.float64)

        # 5) Data.SamplingRate: [fs]
        f.create_dataset("Data.SamplingRate", data=np.array([fs], dtype=np.float64), dtype=np.float64)

        # 6) Data.Delay: [1, 2]
        f.create_dataset("Data.Delay", data=np.zeros((1, 2), dtype=np.float64), dtype=np.float64)

        # 7) ListenerPosition: [1, 3]
        ds_listener = f.create_dataset("ListenerPosition", data=np.zeros((1, 3), dtype=np.float64), dtype=np.float64)
        ds_listener.attrs["Type"] = "cartesian"
        ds_listener.attrs["Units"] = "meter"

        # 8) ReceiverPosition: [2, 3, 1]
        receiver_position = np.zeros((2, 3, 1), dtype=np.float64)
        receiver_position[0, 1, 0] = 0.0875
        receiver_position[1, 1, 0] = -0.0875
        ds_receiver = f.create_dataset("ReceiverPosition", data=receiver_position, dtype=np.float64)
        ds_receiver.attrs["Type"] = "cartesian"
        ds_receiver.attrs["Units"] = "meter"

        # 9) EmitterPosition: [1, 3, 1]
        emitter_position = np.zeros((1, 3, 1), dtype=np.float64)
        ds_emitter = f.create_dataset("EmitterPosition", data=emitter_position, dtype=np.float64)
        ds_emitter.attrs["Type"] = "cartesian"
        ds_emitter.attrs["Units"] = "meter"

    print("[Done] 已完成从数学公式与 PCA 权重构建 SOFA:")
    print(target_sofa_path)


if __name__ == "__main__":
    main()
