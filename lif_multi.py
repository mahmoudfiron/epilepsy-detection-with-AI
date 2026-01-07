from rockpool.nn.modules import LIF
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Define a small LIF "layer"
# -----------------------------
# We'll use 4 neurons (still << 100, so we match the hardware limitation style)
n_neurons = 4

# LIF population with 4 neurons
lif_layer = LIF(shape=(n_neurons,))

# -----------------------------
# 2. Build an input signal
# -----------------------------
# Time steps
timesteps = 200

# Here we create different constant inputs for each neuron,
# so we can see different firing behaviours.
# Neuron 0 gets lowest drive, neuron 3 gets highest.
base_drive = np.array([0.3, 0.5, 0.7, 0.9])  # shape (4,)
input_signal = np.tile(base_drive, (timesteps, 1))  # shape (T, 4)

# -----------------------------
# 3. Run the simulation
# -----------------------------
output, state, rec = lif_layer(input_signal)

# rec contains recorded variables over time
# Let's inspect the keys once, just to know what's inside
print("Available record keys:", rec.keys())

# Membrane voltage and spikes for each neuron
vmem = rec["vmem"]        # shape (T, 4)
spikes = rec["spikes"]    # shape (T, 4)

print("vmem shape:", vmem.shape)
print("spikes shape:", spikes.shape)

# -----------------------------
# 4. Plot the results
# -----------------------------
vmem = vmem.squeeze(0)
spikes = spikes.squeeze(0)

time = np.arange(timesteps)

plt.figure(figsize=(10, 6))

# Top subplot: membrane voltages
plt.subplot(2, 1, 1)
for i in range(n_neurons):
    plt.plot(time, vmem[:, i], label=f"Neuron {i}")
plt.ylabel("Membrane Voltage (vmem)")
plt.title("Rockpool LIF Layer (4 neurons)")
plt.legend()

# Bottom subplot: spikes
plt.subplot(2, 1, 2)
for i in range(n_neurons):
    plt.step(time, spikes[:, i] + i * 1.2, where="post", label=f"Neuron {i}")
plt.xlabel("Time step")
plt.ylabel("Spikes (offset per neuron)")

plt.tight_layout()
plt.show()