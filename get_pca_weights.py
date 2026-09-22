import numpy as np
import importlib.util

# 动态加载你的基础库
spec = importlib.util.spec_from_file_location("aanf", "AANF-HRTF.py")
aanf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aanf)

print("这次你一定要提现...")
data = aanf.load_sofa_left_logmag("pp1_HRIRs_measured.sofa", eps=1e-10)
pca_model, _ = aanf.fit_pca_left_spectrum(data.left_logmag_db, n_components=24)

# 保存 PCA 权重
np.savez("pca_weights.npz", 
         components=pca_model.components_, 
         mean=pca_model.mean_,
         fs=data.sample_rate_hz)
print("pca_weights.npz 提取成功！")