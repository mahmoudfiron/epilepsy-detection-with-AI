import numpy as np
from rockpool.layers.iaf import IAF

# Simulation parameters
T = 200
neurons = 8
dt = 1e-3

# Create IAF (Leaky Integrate-and-Fire) layer
lif = IAF(
    shape=(neurons,),
    dt=dt,
    tau_mem=20e-3,
    v_thresh=1.0,
    record=True
)

# Constant input current
x = np.ones((T, neurons)) * 0.6

# Run simulation
spikes, state, rec = lif(x)

print("Spikes shape:", spikes.shape)
print("Recorded keys:", rec.keys())
print("Total spikes:", spikes.sum())
