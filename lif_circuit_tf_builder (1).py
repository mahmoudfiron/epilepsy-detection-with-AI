import tensorflow as tf
from nengo_dl import neuron_builders
from nengo_dl.neuron_builders import SimNeuronsBuilder
from Neuron_Customization.LIFCircuitNeuron import LIFCircuitNeuron


# ----------------------------
# Surrogate spike
# ----------------------------
@tf.custom_gradient
def spike_surrogate(v, alpha):
    v = tf.convert_to_tensor(v, tf.float32)
    alpha = tf.cast(alpha, tf.float32)

    y = tf.cast(v >= 0.0, tf.float32)

    def grad(dy):
        s = tf.sigmoid(alpha * v)
        return dy * (alpha * s * (1.0 - s)), None

    return y, grad


def soft_gate(v, alpha):
    """Smooth gate in (0,1) for training stability."""
    v = tf.convert_to_tensor(v, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    return tf.sigmoid(alpha * v)


def _tf_step_lif_circuit(neuron, dt, J, Vm, Vm_prev, Vf, Vf_prev):
    # cast
    J       = tf.cast(J, tf.float32)
    Vm      = tf.cast(Vm, tf.float32)
    Vm_prev = tf.cast(Vm_prev, tf.float32)
    Vf      = tf.cast(Vf, tf.float32)
    Vf_prev = tf.cast(Vf_prev, tf.float32)
    dt      = tf.cast(dt, tf.float32)

    # neuron params
    Cm   = tf.cast(neuron.Cm, tf.float32)
    Cf   = tf.cast(neuron.Cf, tf.float32)
    VDD  = tf.cast(neuron.VDD, tf.float32)
    VTh  = tf.cast(neuron.VTh, tf.float32)
    IR   = tf.cast(neuron.IR, tf.float32)
    beta = tf.cast(neuron.beta, tf.float32)

    # new params
    leak  = tf.cast(getattr(neuron, "leak", 0.0), tf.float32)
    alpha = tf.cast(getattr(neuron, "alpha", 3.0), tf.float32)

    soft_output = bool(getattr(neuron, "soft_output", True))

    # -----------------------------------------
    # 1) Compute dVf/dt ONLY (do NOT re-integrate Vf)
    # -----------------------------------------
    dVf_dt = (Vf - Vf_prev) / dt

    # -----------------------------------------
    # 2) Hardware-like gate (surrogate)
    # -----------------------------------------
    f = spike_surrogate(Vm - VTh, alpha)

    # -----------------------------------------
    # 3) Update Vm' using circuit equation
    # Cm dVm'/dt = J - leak*Vm' - IR*f + Cf*dVf/dt
    # -----------------------------------------
    dVm_dt = (J - leak * Vm - IR * f + Cf * dVf_dt) / Cm
    Vm_new = Vm + dt * dVm_dt

    # -----------------------------------------
    # 4) Soft crossing event
    # -----------------------------------------
    p_now   = tf.sigmoid(alpha * (Vm_new - VTh))
    p_prev  = tf.sigmoid(alpha * (Vm     - VTh))
    p_cross = p_now * (1.0 - p_prev)

    # -----------------------------------------
    # 5) Soft reset/update (Vm and Vf)
    #    - crossed:  Vm = beta*VDD , Vf = VDD*(1-beta)
    #    - else:     Vf follows Vm (hardware-like)
    # -----------------------------------------
    Vm_reset = beta * VDD
    Vf_reset = VDD * (1.0 - beta)

    Vm_after = (1.0 - p_cross) * Vm_new + p_cross * Vm_reset

    # Vf follows Vm_after when NOT crossing
    Vf_follow = Vm_after
    Vf_after  = (1.0 - p_cross) * Vf_follow + p_cross * Vf_reset

    # -----------------------------------------
    # 6) Output gate AFTER update
    # -----------------------------------------
    if soft_output:
        g = soft_gate(Vm_after - VTh, alpha)
    else:
        g = spike_surrogate(Vm_after - VTh, alpha)

    out_analog = VDD * g
    out_spike  = (1.0 / dt) * g
    out = out_spike if getattr(neuron, "output_mode", "analog") == "spike" else out_analog

    # -----------------------------------------
    # 7) Commit prevs
    # -----------------------------------------
    Vm_prev_new = Vm
    Vf_prev_new = Vf

    return out, Vm_after, Vm_prev_new, Vf_after, Vf_prev_new



class LIFCircuitTFImpl(neuron_builders.TFNeuronBuilder):
    """
    Handles multiple ops by splitting J/state by op sizes then concatenating back.
    MUST return a FLAT tuple: (out, s1, s2, s3, s4) not a dict.
    """

    def __init__(self, ops):
        super().__init__(ops)
        self.neurons = [op.neurons for op in ops]
        self.sizes = [int(op.J.shape[-1]) for op in ops]
        self.state_keys = ["Vm_prime", "Vm_prime_prev", "Vf", "Vf_prev"]

    def step(self, J, dt, **state):
        Js = tf.split(J, self.sizes, axis=-1)
        split_state = {k: tf.split(state[k], self.sizes, axis=-1) for k in self.state_keys}

        outs, Vm_list, Vm_prev_list, Vf_list, Vf_prev_list = [], [], [], [], []

        for i, neuron in enumerate(self.neurons):
            Vm      = split_state["Vm_prime"][i]
            Vm_prev = split_state["Vm_prime_prev"][i]
            Vf      = split_state["Vf"][i]
            Vf_prev = split_state["Vf_prev"][i]

            out_i, Vm_after, Vm_prev_new, Vf_after, Vf_prev_new = _tf_step_lif_circuit(
                neuron, dt, Js[i], Vm, Vm_prev, Vf, Vf_prev
            )

            outs.append(out_i)
            Vm_list.append(Vm_after)
            Vm_prev_list.append(Vm_prev_new)
            Vf_list.append(Vf_after)
            Vf_prev_list.append(Vf_prev_new)

        out         = tf.concat(outs, axis=-1)
        Vm_new      = tf.concat(Vm_list, axis=-1)
        Vm_prev_new = tf.concat(Vm_prev_list, axis=-1)
        Vf_new      = tf.concat(Vf_list, axis=-1)
        Vf_prev_new = tf.concat(Vf_prev_list, axis=-1)

        return out, Vm_new, Vm_prev_new, Vf_new, Vf_prev_new


def register_lif_circuit_builder(verbose=True):
    impl = SimNeuronsBuilder.TF_NEURON_IMPL
    if hasattr(impl, "register"):
        impl.register(LIFCircuitNeuron)(LIFCircuitTFImpl)
    else:
        impl[LIFCircuitNeuron] = LIFCircuitTFImpl

    if verbose:
        print("[OK] Registered LIFCircuitNeuron TF implementation:", LIFCircuitTFImpl)
