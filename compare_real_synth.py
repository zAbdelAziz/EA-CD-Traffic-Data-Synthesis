import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

from utils.post_process import postprocess_synth_20




def compare_real_synth(real, synth, clip_values=(93.747, -93.747), atol=1e-6):
    """
    real, synth: arrays of shape [N, D] or [N, T, D]
    """
    real = np.asarray(real)
    synth = np.asarray(synth)

    if real.ndim == 3:
        real_flat = real.reshape(-1, real.shape[-1])
        synth_flat = synth.reshape(-1, synth.shape[-1])
    else:
        real_flat = real
        synth_flat = synth

    D = real_flat.shape[1]
    report = []

    for j in tqdm(range(D)):
        r = real_flat[:, j]
        s = synth_flat[:, j]

        entry = {
            "feature": j,
            "real_mean": float(np.mean(r)),
            "synth_mean": float(np.mean(s)),
            "real_std": float(np.std(r)),
            "synth_std": float(np.std(s)),
            "wasserstein": float(wasserstein_distance(r, s)),
            "ks_stat": float(ks_2samp(r, s).statistic),
            "real_zero_rate": float(np.mean(np.isclose(r, 0.0, atol=atol))),
            "synth_zero_rate": float(np.mean(np.isclose(s, 0.0, atol=atol))),
        }

        for cv in clip_values:
            entry[f"real_rate_eq_{cv}"] = float(np.mean(np.isclose(r, cv, atol=atol)))
            entry[f"synth_rate_eq_{cv}"] = float(np.mean(np.isclose(s, cv, atol=atol)))

        report.append(entry)

    # Correlation structure
    real_corr = np.corrcoef(real_flat, rowvar=False)
    synth_corr = np.corrcoef(synth_flat, rowvar=False)
    corr_diff = np.nanmean(np.abs(real_corr - synth_corr))

    return report, corr_diff


def block_activity(x, block_size=5):
    """
    x: [N, T, D] expected
    returns mean activity per block where activity = last value in each 5-d block > 0.5
    """
    x = np.asarray(x)
    assert x.ndim == 3
    D = x.shape[-1]
    n_blocks = D // block_size
    masks = []
    for b in tqdm(range(n_blocks)):
        mask_col = b * block_size + (block_size - 1)
        masks.append(x[..., mask_col] > 0.5)
    masks = np.stack(masks, axis=-1)  # [N, T, n_blocks]
    return masks.mean(axis=(0, 1))


def temporal_smoothness(x):
    """
    x: [N, T, D]
    lower means smoother
    """
    x = np.asarray(x)
    diffs = np.diff(x, axis=1)
    return np.mean(np.abs(diffs), axis=(0, 1))



synth = np.load('datasets-files/synth/ngsim-N300000.npz')['X']
real = np.load('datasets-files/refactored/ngsim.npz')['X']

# synth = postprocess_synth_20(synth)

# Example:
report, corr_diff = compare_real_synth(real, synth)
print(report)
print('correlation', corr_diff)
#
# real_act = block_activity(real)
# print('Real Act', real_act)
# synth_act = block_activity(synth)
# print('Synth Act', synth_act)
# real_smooth = temporal_smoothness(real)
# print('Real Smooth', real_smooth)
# synth_smooth = temporal_smoothness(synth)
# print('Synth Smooth', synth_smooth)