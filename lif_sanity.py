import numpy as np
from rockpool.nn.modules import LIF

# One LIF neuron (layer of size 1)
lif = LIF(
    shape=(1,),
    tau_mem=0.02,
    tau_syn=0.02,
    threshold=1.0,
    dt=5e-4
)

# Constant input for 1000 timesteps
T = 1000
u = np.ones((T, 1)) * 1.2

# Run simulation
out = lif(u)

# Inspect what Rockpool actually returns
print("Type of output:", type(out))

if isinstance(out, tuple):
    print("Number of returned elements:", len(out))
    for i, item in enumerate(out):
        if hasattr(item, "shape"):
            print(f"Element {i}: shape = {item.shape}")
        else:
            print(f"Element {i}: type = {type(item)}")
else:
    print("Output shape:", out.shape)