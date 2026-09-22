import h5py
import numpy as np
import shutil

def get_onset_index(hrir: np.ndarray, threshold_ratio: float = 0.1) -> int:
    """
    通过阈值法寻找冲激响应的真实起振时间（Onset）
    """
    max_val = np.max(np.abs(hrir))
    threshold = max_val * threshold_ratio
    # 找到第一个超过最大值 10% 的采样点索引
    onset_idx = np.where(np.abs(hrir) > threshold)[0][0]
    return onset_idx

def inject_itd_to_aanf(original_sofa: str, aanf_sofa: str, output_sofa: str):
    print("正在进行 ITD 时间差重新加载...")
    
    # 复制一份你的 AANF SOFA 作为修改底本
    shutil.copy(aanf_sofa, output_sofa)
    
    with h5py.File(original_sofa, "r") as f_orig, h5py.File(output_sofa, "r+") as f_out:
        orig_ir = f_orig["Data.IR"][:]
        aanf_ir = f_out["Data.IR"][:]
        
        num_positions = orig_ir.shape[0]
        ir_length = orig_ir.shape[2]
        
        for i in range(num_positions):
            # 1. 提取真实测量的左右耳起振延迟 (Onset Delay)
            onset_l = get_onset_index(orig_ir[i, 0, :])
            onset_r = get_onset_index(orig_ir[i, 1, :])
            
            # 2. 对我们生成的“同时起振”的 AANF_HRIR 进行循环移位（加延迟）
            # 注意：np.roll 会把后面溢出的补到前面，但因为 HRIR 尾部全是 0，所以相当于纯粹的延迟
            aanf_ir[i, 0, :] = np.roll(aanf_ir[i, 0, :], onset_l)
            aanf_ir[i, 1, :] = np.roll(aanf_ir[i, 1, :], onset_r)
            
            # 3. 把延迟前面（roll过来的部分）强行置为 0，保证物理纯净
            aanf_ir[i, 0, :onset_l] = 0.0
            aanf_ir[i, 1, :onset_r] = 0.0
            
        # 覆写保存
        f_out["Data.IR"][...] = aanf_ir
        
    print(f"加载完成，请重新加载这个新文件：{output_sofa}")

if __name__ == "__main__":
    orig = "pp1_HRIRs_measured.sofa"      # 你的原始测量文件
    aanf = "AANF_reconstructed.sofa"      # 你刚用 PySR 生成的文件
    out = "AANF_with_ITD.sofa"            # 加载ITD拟合后的最终文件
    
    inject_itd_to_aanf(orig, aanf, out)