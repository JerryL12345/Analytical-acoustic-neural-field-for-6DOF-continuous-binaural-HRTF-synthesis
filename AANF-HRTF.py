#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analytical Acoustic Neural Field Demo Pipeline
目标：
1) 读取 SimpleFreeFieldHRIR SOFA
2) 左耳 HRIR -> 频域 -> 对数幅度谱(dB)
3) PCA 降到 8 维潜变量
4) 用 PySR 做符号回归，并基于镜像对称生成双耳频谱
"""

import os
import csv
import shutil
import concurrent.futures
import warnings
from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Optional, cast

import h5py
import numpy as np
from sklearn.decomposition import PCA


# ----------------------------
# 工具函数
# ----------------------------

def ensure_2d_angles(azimuth, elevation) -> np.ndarray:
    """
    将输入角度整理成 Nx2，支持标量或数组。
    """
    az = np.asarray(azimuth, dtype=np.float64)
    el = np.asarray(elevation, dtype=np.float64)

    if az.shape != el.shape:
        raise ValueError("azimuth 和 elevation 的形状必须一致。")

    az = np.mod(az, 360.0)
    x = np.column_stack([az.reshape(-1), el.reshape(-1)])
    return x


def spherical_deg_to_cartesian_unit(angles_deg: np.ndarray) -> np.ndarray:
    """
    将 Nx2 的球坐标角度 [azimuth_deg, elevation_deg] 映射到单位球面的 Nx3 笛卡尔坐标。
    """
    if angles_deg.ndim != 2 or angles_deg.shape[1] != 2:
        raise ValueError("angles_deg 必须是形状 (M, 2) 的矩阵。")

    azimuth_rad = np.deg2rad(angles_deg[:, 0].astype(np.float64))
    elevation_rad = np.deg2rad(angles_deg[:, 1].astype(np.float64))

    cos_ele = np.cos(elevation_rad)
    x_pos = np.cos(azimuth_rad) * cos_ele
    y_pos = np.sin(azimuth_rad) * cos_ele
    z_pos = np.sin(elevation_rad)
    return np.column_stack([x_pos, y_pos, z_pos])


def decode_attr(attr_value) -> str:
    """
    安全解码 HDF5 属性到字符串。
    """
    if isinstance(attr_value, bytes):
        return attr_value.decode("utf-8", errors="ignore")
    if isinstance(attr_value, np.ndarray) and attr_value.dtype.type is np.bytes_:
        return b"".join(attr_value.tolist()).decode("utf-8", errors="ignore")
    return str(attr_value)


def cartesian_to_spherical_deg(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    笛卡尔坐标 -> 方位角/俯仰角（单位：度）
    azimuth: [0, 360)
    elevation: [-90, 90]
    """
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("笛卡尔坐标必须为 Nx3。")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    azimuth = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    r_xy = np.sqrt(x * x + y * y)
    elevation = np.degrees(np.arctan2(z, r_xy))
    return azimuth, elevation


# ----------------------------
# 步骤 1：数据加载与频域转换
# ----------------------------

@dataclass
class HRTFData:
    azimuth_deg: np.ndarray         # (M,)
    elevation_deg: np.ndarray       # (M,)
    left_logmag_db: np.ndarray      # (M, F)
    freqs_hz: np.ndarray            # (F,)
    sample_rate_hz: float


def load_sofa_left_logmag(
    sofa_path: str,
    eps: float = 1e-10
) -> HRTFData:
    """
    读取 SOFA(SimpleFreeFieldHRIR)并提取左耳对数幅度谱(dB)。
    """
    if not os.path.isfile(sofa_path):
        raise FileNotFoundError(f"找不到 SOFA 文件: {sofa_path}")

    with h5py.File(sofa_path, "r") as f:
        # 基本结构检查
        if "Data.IR" not in f:
            raise KeyError("SOFA 文件缺少 Data.IR 数据集。")
        if "SourcePosition" not in f:
            raise KeyError("SOFA 文件缺少 SourcePosition 数据集。")
        if "Data.SamplingRate" not in f:
            raise KeyError("SOFA 文件缺少 Data.SamplingRate 数据集。")

        ir = f["Data.IR"][:]  # 常见形状: (M, R, N)
        src_pos = f["SourcePosition"][:]  # (M, C)
        fs = float(np.asarray(f["Data.SamplingRate"][:]).reshape(-1)[0])

        if ir.ndim != 3:
            raise ValueError(f"Data.IR 维度异常，期望 3 维，得到 {ir.ndim}。")
        if ir.shape[1] < 1:
            raise ValueError("Data.IR 接收器维度异常，无法提取左耳。")
        if src_pos.ndim != 2 or src_pos.shape[0] != ir.shape[0]:
            raise ValueError("SourcePosition 与 Data.IR 的测点数量不一致。")

        # 判断 SourcePosition 的坐标类型
        src_type = decode_attr(f["SourcePosition"].attrs.get("Type", "spherical")).lower()
        src_units = decode_attr(f["SourcePosition"].attrs.get("Units", "degree, degree, meter")).lower()

        if "sph" in src_type:
            # 常见 SOFA：第 0 列方位角，第 1 列俯仰角（单位通常 degree）
            if src_pos.shape[1] < 2:
                raise ValueError("球坐标 SourcePosition 至少需要两列(azi, ele)。")
            azimuth = np.mod(src_pos[:, 0].astype(np.float64), 360.0)
            elevation = src_pos[:, 1].astype(np.float64)

            # 若单位可能是弧度，尝试自动转换（保守判断）
            if "radian" in src_units or "rad" in src_units:
                azimuth = np.degrees(azimuth) % 360.0
                elevation = np.degrees(elevation)
        elif "cart" in src_type:
            azimuth, elevation = cartesian_to_spherical_deg(src_pos[:, :3].astype(np.float64))
        else:
            warnings.warn(
                f"未知 SourcePosition.Type={src_type}，默认按球坐标前两列读取(azi, ele)。",
                RuntimeWarning
            )
            azimuth = np.mod(src_pos[:, 0].astype(np.float64), 360.0)
            elevation = src_pos[:, 1].astype(np.float64)

        # 只提取左耳（receiver index = 0）
        left_hrir = ir[:, 0, :].astype(np.float64)  # (M, N)

    # 时域 HRIR -> 频域（rfft）
    left_fft = np.fft.rfft(left_hrir, axis=-1)      # (M, F)
    left_mag = np.abs(left_fft)                     # (M, F)

    # 对数幅度谱（dB），log 前加 eps 防止 log(0)
    left_logmag_db = 20.0 * np.log10(left_mag + eps)

    n_fft = left_hrir.shape[-1]
    freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / fs)

    return HRTFData(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        left_logmag_db=left_logmag_db,
        freqs_hz=freqs,
        sample_rate_hz=fs
    )


# ----------------------------
# 步骤 2：PCA 降维
# ----------------------------

def fit_pca_left_spectrum(
    left_logmag_db: np.ndarray,
    n_components: int = 32
) -> Tuple[PCA, np.ndarray]:
    """
    对左耳对数幅度谱做 PCA。
    """
    if left_logmag_db.ndim != 2:
        raise ValueError("left_logmag_db 必须是二维矩阵 (M, F)。")
    if left_logmag_db.shape[0] < n_components:
        raise ValueError("样本数小于 PCA 主成分数，无法训练。")

    pca = PCA(n_components=n_components, random_state=42)
    z = pca.fit_transform(left_logmag_db)

    # 打印解释方差比
    print("\n[PCA] Explained Variance Ratio:")
    for i, r in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {r:.6f}")
    print(f"  Cumulative: {np.sum(pca.explained_variance_ratio_):.6f}\n")

    return pca, z


# ----------------------------
# 步骤 3：PySR 符号回归训练（按方差占比分配算力）
# ----------------------------

def train_single_latent(i, X, y_col, allocated_iters, current_maxsize):
    """
    顶层 Worker：训练单个潜变量，返回 (i, model)。
    """
    print(f"开始训练 Latent {i} (预算: {allocated_iters}次, 复杂度: {current_maxsize}) ...")

    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=allocated_iters,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp"],
        elementwise_loss="loss(x, y) = (x - y)^2",
        model_selection="accuracy",
        parallelism="multiprocessing",
        procs=2,
        variable_names=["x_pos", "y_pos", "z_pos"],
        populations=10,
        population_size=27,
        maxsize=current_maxsize,
        verbosity=1,
        random_state=42,
        deterministic=False,
    )
    model.fit(X, y_col)
    setattr(model, "allocated_iters_", allocated_iters)
    return i, model

def train_symbolic_regressor(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    latent_y: np.ndarray,
    pca_model: PCA,
    total_budget: int = 500
) -> List:
    """
    训练 PySR：输入单位球面 3D 坐标 [x_pos, y_pos, z_pos]，输出多维潜变量。
    按 PCA 方差占比动态分配每个潜变量的 niterations，总预算由 total_budget 控制。
    """
    # 优先使用本地已安装 Julia，避免 juliapkg 走在线安装流程。
    local_julia = os.path.expanduser("~/.local/bin/julia")
    if os.path.isfile(local_julia):
        os.environ["PYTHON_JULIAPKG_EXE"] = local_julia
        os.environ["PYTHON_JULIAPKG_PROJECT"] = os.path.expanduser(
            "~/.julia/environments/pysr-juliapkg"
        )

    try:
        from pysr import PySRRegressor
    except ImportError as e:
        raise ImportError(
            "未安装 PySR。请先安装 pysr 和 Julia 环境。"
        ) from e

    angles_deg = np.column_stack([
        np.mod(azimuth_deg.astype(np.float64), 360.0),
        elevation_deg.astype(np.float64)
    ])
    X = spherical_deg_to_cartesian_unit(angles_deg)
    Y = np.asarray(latent_y, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X 必须是形状 (M, 3) 的矩阵，对应 [x_pos, y_pos, z_pos]。")
    if Y.ndim != 2:
        raise ValueError("latent_y 必须是二维矩阵 (M, n_components)。")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X 与 Y 的样本数不一致。")

    variance_ratios = np.asarray(pca_model.explained_variance_ratio_, dtype=np.float64)
    if variance_ratios.shape[0] != Y.shape[1]:
        raise ValueError("PCA 主成分数与 latent_y 的列数不一致。")

    tasks = []
    for i in range(Y.shape[1]):
        # 算力分配逻辑（保持不变）
        allocated_iters = int(total_budget * variance_ratios[i])
        allocated_iters = max(40, allocated_iters)
        ratio_percent = variance_ratios[i] * 100.0
        
        # ==========================================
        # 只有前 2 个最重要的主成分（i=0, 1），给它们 55 的大空间
        # 后面的主成分，依然用 30 的小空间防止过拟合。
        # ==========================================
        current_maxsize = 60 if i < 2 else 35

        tasks.append((i, X, Y[:, i], allocated_iters, current_maxsize))

    models: List[Any] = [None] * Y.shape[1]
    max_workers = 4
    print("启动多进程并行训练... (外部并发数: 4, 内部 procs: 2)")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(train_single_latent, i, task_x, y_col, allocated_iters, current_maxsize): i
            for i, task_x, y_col, allocated_iters, current_maxsize in tasks
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                result_i, model = future.result()
                print(f"Latent {result_i} 训练完成！")
                models[result_i] = model
            except Exception as e:
                print(f"Latent {i} 训练失败: {e}")
                raise

    return models


# ----------------------------
# 步骤 4：镜像对称双耳推理
# ----------------------------

def generate_binaural_hrtf(
    azimuth: Union[float, np.ndarray],
    elevation: Union[float, np.ndarray],
    pysr_model: List,
    pca_model: PCA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于镜像先验推理双耳频谱（log-magnitude dB）。

    左耳：将 (azimuth, elevation) 映射到单位球面 3D 坐标 [x, y, z]
    右耳：保持 X/Z 不变，仅对 Y 轴取反，得到 [x, -y, z]
    """
    angles_deg = ensure_2d_angles(azimuth, elevation)
    X_left = spherical_deg_to_cartesian_unit(angles_deg)
    X_right = X_left.copy()
    X_right[:, 1] *= -1.0

    if len(pysr_model) == 0:
        raise ValueError("pysr_model 为空，无法进行推理。")

    # 1) 遍历模型列表分别预测左右耳潜变量，并按列拼接为 (N, n_components)
    z_left = np.column_stack([
        np.asarray(model.predict(X_left), dtype=np.float64).reshape(-1)
        for model in pysr_model
    ])
    z_right = np.column_stack([
        np.asarray(model.predict(X_right), dtype=np.float64).reshape(-1)
        for model in pysr_model
    ])

    # 2) 用 PCA 逆变换恢复完整频谱
    left_logmag_db = pca_model.inverse_transform(z_left)
    right_logmag_db = pca_model.inverse_transform(z_right)

    # 若输入是标量，则返回 1D 频谱
    if np.isscalar(azimuth) and np.isscalar(elevation):
        left_logmag_db = left_logmag_db[0]
        right_logmag_db = right_logmag_db[0]

    return left_logmag_db, right_logmag_db


def get_min_loss_equation(model) -> Optional[dict]:
    """
    从 PySR 的 Hall of Fame 中显式选取 loss 最小的公式记录。
    """
    equations = getattr(model, "equations_", None)
    if equations is None:
        return None

    try:
        loss_values = np.asarray(equations["loss"], dtype=np.float64)
    except Exception:
        return None

    if loss_values.size == 0:
        return None

    best_idx = int(np.argmin(loss_values))

    try:
        best_row = equations.iloc[best_idx]
        equation = str(best_row["equation"])
        loss_value = float(best_row["loss"])
        score_value = float(best_row["score"]) if "score" in best_row.index else np.nan
    except Exception:
        try:
            equation = str(equations["equation"][best_idx])
            loss_value = float(equations["loss"][best_idx])
            score_value = float(equations["score"][best_idx]) if "score" in equations else np.nan
        except Exception:
            return None

    return {
        "equation": equation,
        "loss": loss_value,
        "score": score_value,
    }


def export_pysr_summary_csv(
    pysr_model: List,
    total_budget: int,
    csv_path: str
) -> None:
    """
    导出每个 PC 的最佳公式、Loss 和本次 total_budget 到 CSV。
    """
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["pc_index", "equation", "loss", "allocated_iters", "total_budget"]
        )
        writer.writeheader()

        for idx, model in enumerate(pysr_model):
            equation = ""
            loss_value = np.nan
            allocated_iters = getattr(model, "allocated_iters_", np.nan)

            best_eq = get_min_loss_equation(model)
            if best_eq is not None:
                equation = str(best_eq.get("equation", ""))
                loss_value = best_eq.get("loss", np.nan)

            writer.writerow(
                {
                    "pc_index": idx,
                    "equation": equation,
                    "loss": loss_value,
                    "allocated_iters": allocated_iters,
                    "total_budget": total_budget,
                }
            )


def magnitude_to_minimum_phase_hrir(mag_db: np.ndarray, n_time_samples: int) -> np.ndarray:
    """
    使用倒谱法将对数幅度谱重建为最小相位 HRIR。
    输入 mag_db 形状为 (M, F)，输出 HRIR 形状为 (M, n_time_samples)。
    """
    mag_db = np.asarray(mag_db, dtype=np.float64)
    if mag_db.ndim != 2:
        raise ValueError("mag_db 必须是二维矩阵 (M, F)。")

    # 1) dB 幅度谱 -> 线性幅度谱，并避免 log(0)
    mag = np.maximum(10.0 ** (mag_db / 20.0), 1e-10)

    # 2) 线性幅度谱取自然对数，得到实数对数频谱
    log_mag = np.log(mag)

    # 3) 构造双边对称的完整对数频谱，长度恢复为 n_fft
    full_log_mag = np.concatenate([log_mag, log_mag[:, -2:0:-1]], axis=-1)
    n_fft = full_log_mag.shape[-1]

    # 4) IFFT 得到实倒谱
    ceps = np.fft.ifft(full_log_mag, axis=-1).real

    # 5) 构造最小相位折叠窗
    window = np.zeros(n_fft, dtype=np.float64)
    window[0] = 1.0
    if n_fft % 2 == 0:
        window[n_fft // 2] = 1.0
        window[1:n_fft // 2] = 2.0
    else:
        window[1:(n_fft + 1) // 2] = 2.0

    # 6) 加窗后回到频域，指数化得到最小相位复频谱
    min_phase_log = np.fft.fft(ceps * window, axis=-1)
    min_phase_spec = np.exp(min_phase_log)

    # 7) IFFT 回到时域，并截断为目标 HRIR 长度
    hrir = np.fft.ifft(min_phase_spec, axis=-1).real
    return hrir[:, :n_time_samples]


def export_to_sofa(
    original_sofa_path: str,
    target_sofa_path: str,
    pred_left_db: np.ndarray,
    pred_right_db: np.ndarray
) -> None:
    """
    将预测的左右耳对数幅度谱通过最小相位重建为 HRIR，并写入新的 SOFA 文件。
    """
    shutil.copy(original_sofa_path, target_sofa_path)

    with h5py.File(target_sofa_path, "r+") as f:
        ir_dataset = cast(Any, f["Data.IR"])
        n_samples = ir_dataset.shape[-1]

        hrir_left = magnitude_to_minimum_phase_hrir(pred_left_db, n_samples)
        hrir_right = magnitude_to_minimum_phase_hrir(pred_right_db, n_samples)

        if hrir_left.shape != (ir_dataset.shape[0], n_samples):
            raise ValueError("左耳 HRIR 形状与目标 SOFA 的 Data.IR 不匹配。")
        if hrir_right.shape != (ir_dataset.shape[0], n_samples):
            raise ValueError("右耳 HRIR 形状与目标 SOFA 的 Data.IR 不匹配。")

        new_ir = np.zeros(ir_dataset.shape, dtype=ir_dataset.dtype)
        new_ir[:, 0, :] = hrir_left
        new_ir[:, 1, :] = hrir_right
        ir_dataset[...] = new_ir


# ----------------------------
# 主流程示例
# ----------------------------

def main():
    # 默认加载当前脚本同目录下的 SOFA 文件
    sofa_path = os.path.join(os.path.dirname(__file__), "pp1_HRIRs_measured.sofa")

    # 步骤 1：加载 + 频域转换（仅左耳）
    data = load_sofa_left_logmag(sofa_path=sofa_path, eps=1e-10)
    print(f"[Data] 测点数 M = {data.left_logmag_db.shape[0]}")
    print(f"[Data] 频点数 F = {data.left_logmag_db.shape[1]}")
    print(f"[Data] 采样率 Fs = {data.sample_rate_hz:.1f} Hz")

    # 步骤 2：PCA 降维
    pca_model, latent = fit_pca_left_spectrum(data.left_logmag_db, n_components=24)
    total_budget = 10000

    # 排雷测试：计算 PCA 降维的理论极限 LSD（即使用完美的潜变量反变换）
    reconstructed = pca_model.inverse_transform(latent)  # (M, F)
    lsd_per_sample = np.sqrt(np.mean((data.left_logmag_db - reconstructed) ** 2, axis=1))  # (M,)
    pca_limit_lsd = float(np.mean(lsd_per_sample))
    print(f"当前 PCA 的理论极限平均 LSD 为: {pca_limit_lsd:.3f} dB")

    # 步骤 3：符号回归训练
    pysr_model = train_symbolic_regressor(
        azimuth_deg=data.azimuth_deg,
        elevation_deg=data.elevation_deg,
        latent_y=latent,
        pca_model=pca_model,
        total_budget=total_budget,
    )

    # 打印每个潜变量的最佳表达式
    print("[PySR] 最佳表达式（每个潜变量一个）：")
    for idx, model in enumerate(pysr_model):
        best_eq = get_min_loss_equation(model)
        print(f"  Latent {idx}: {best_eq}")

    # 导出每个 PC 的公式、Loss 和 total_budget，便于后处理分析
    csv_path = os.path.join(os.path.dirname(__file__), "pysr_pc_summary.csv")
    export_pysr_summary_csv(
        pysr_model=pysr_model,
        total_budget=total_budget,
        csv_path=csv_path,
    )
    print(f"[CSV] 已导出 PySR 结果汇总到: {csv_path}")

    # 步骤 4：镜像对称双耳批量推理（全测点一次性向量化）
    pred_left_all, pred_right_all = generate_binaural_hrtf(
        azimuth=data.azimuth_deg,
        elevation=data.elevation_deg,
        pysr_model=pysr_model,
        pca_model=pca_model
    )

    # 全局评估：提取真实左耳频谱矩阵
    true_left_all = data.left_logmag_db

    # 严格按矩阵顺序计算 Global Mean LSD
    squared_diff = (true_left_all - pred_left_all) ** 2
    mse_per_sample = np.mean(squared_diff, axis=1)
    lsd_per_sample = np.sqrt(mse_per_sample)
    global_mean_lsd = float(np.mean(lsd_per_sample))

    target_sofa = "AANF_reconstructed.sofa"
    export_to_sofa(sofa_path, target_sofa, pred_left_all, pred_right_all)
    print(f"解析场重构完成！SOFA 文件已保存至: {target_sofa}")

    print("[Done] Pipeline 执行完成。")
    print("\n" + "=" * 56)
    print("模型精度评估 (Global Mean LSD)")
    print("=" * 56)
    print(f"当前模型的全局平均 LSD: {global_mean_lsd:.3f} dB")

    #自我评估一下。。。hhh。。。sorry
    if global_mean_lsd < 1.5:
        print("🏆 评价：神级公式！(SOTA 水平)")
    elif global_mean_lsd < 2.5:
        print("💡 评价：极其优秀！(完美 Baseline)")
    elif global_mean_lsd < 3.5:
        print("📈 评价：可用状态。模型已跑通。")
    else:
        print("⚠️ 评价：依然存在欠拟合，请检查 maxsize 或 PCA 维度。")


if __name__ == "__main__":
    main()