import numpy as np

IN_PATH = "spiking_weights.npz"
OUT_PATH = "spiking_weights_as_quant_pack.npz"

src = np.load(IN_PATH, allow_pickle=True)

# Map arrays -> weights/biases
W0 = src["arr_1"].astype(np.float32)  # (6,8)
b0 = src["arr_2"].astype(np.float32)  # (8,)
W1 = src["arr_3"].astype(np.float32)  # (8,8)
b1 = src["arr_4"].astype(np.float32)  # (8,)
W2 = src["arr_5"].astype(np.float32)  # (8,8)
b2 = src["arr_6"].astype(np.float32)  # (8,)
W3 = src["arr_7"].astype(np.float32)  # (8,1)
b3 = src["arr_8"].astype(np.float32)  # (1,)

# Your loader does:
# W = (k_q - z_k) * s_k
# If we set s=1 and z=0, then W == k_q exactly (even if it's float).
one = np.float32(1.0)
zero = np.float32(0.0)

out = {
    # "Quant-like" keys expected by your code
    "k0_q": W0, "b0_q": b0, "s_k0": one, "z_k0": zero, "s_b0": one, "z_b0": zero,
    "k1_q": W1, "b1_q": b1, "s_k1": one, "z_k1": zero, "s_b1": one, "z_b1": zero,
    "k2_q": W2, "b2_q": b2, "s_k2": one, "z_k2": zero, "s_b2": one, "z_b2": zero,
    "k3_q": W3, "b3_q": b3, "s_k3": one, "z_k3": zero, "s_b3": one, "z_b3": zero,
}

# Keep arr_0 too (optional), in case you want to inspect it later
if "arr_0" in src:
    out["extra_arr_0"] = src["arr_0"]

np.savez(OUT_PATH, **out)
print(f"Saved converted pack -> {OUT_PATH}")
