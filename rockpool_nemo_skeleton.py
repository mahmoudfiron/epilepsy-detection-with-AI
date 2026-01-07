import numpy as np
from rockpool.nn.modules import Linear, LIF
import time

# ------------------------
# Configuration (match NEMO_RND)
# ------------------------
DT = 5e-4
TAU_MEM = 0.02
TAU_SYN = 0.02

T = 1000
BATCH = 1

# ------------------------
# Input: constant 6-D signal
# ------------------------
x = np.ones((BATCH, T, 6), dtype=np.float32) * 0.5

# ------------------------
# Layer 1
# ------------------------
fc1 = Linear((6, 8))
lif1 = LIF(shape=(8,), tau_mem=TAU_MEM, tau_syn=TAU_SYN, dt=DT)

# ------------------------
# Layer 2
# ------------------------
fc2 = Linear((8, 8))
lif2 = LIF(shape=(8,), tau_mem=TAU_MEM, tau_syn=TAU_SYN, dt=DT)

# ------------------------
# Layer 3
# ------------------------
fc3 = Linear((8, 8))
lif3 = LIF(shape=(8,), tau_mem=TAU_MEM, tau_syn=TAU_SYN, dt=DT)

# ------------------------
# Output layer
# ------------------------
fc_out = Linear((8, 1))

# ------------------------
# Runtime measurement
# ------------------------
start_time = time.perf_counter()

z = fc1(x)[0]
z = lif1(z)[0]

z = fc2(z)[0]
z = lif2(z)[0]

z = fc3(z)[0]
z = lif3(z)[0]

y = fc_out(z)[0]

end_time = time.perf_counter()

elapsed = end_time - start_time

print("Output shape:", y.shape)
print("Mean output activity:", y.mean())

print(f"Total forward-pass time: {elapsed:.6f} seconds")
print(f"Time per timestep: {elapsed / T:.9f} seconds")