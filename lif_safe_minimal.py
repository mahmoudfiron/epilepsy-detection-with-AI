import numpy as np
import matplotlib.pyplot as plt
from rockpool.nn.modules.native.lif import LIF

n_neurons = 4
T = 200

# Create LIF (Rockpool v2 style)
lif = LIF(shape=(n_neurons,))
lif.tau_mem = np.full((n_neurons,), 20e-3, dtype=np.float32)
lif.v_thresh = np.full((n_neurons,), 1.0, dtype=np.float32)
lif.record = True

# Input current
x = np.ones((T, n_neurons), dtype=np.float32) * 0.6

spikes, _, rec = lif(x)

t = np.arange(T)

for i in range(n_neurons):
    plt.plot(t, rec["vmem"][0, :, i], label=f"Neuron {i}")

plt.xlabel("Time step")
plt.ylabel("Membrane voltage")
plt.title("Membrane voltage (LIF neurons)")
plt.legend()
plt.show()

