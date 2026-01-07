import numpy as np
import nengo
from nengo.params import NumberParam
from nengo.dists import Uniform, Choice


class LIFCircuitNeuron(nengo.neurons.NeuronType):
    """
    Hardware-like LIF circuit neuron (CPU reference implementation).
    """

    #  better init: break symmetry (not all zeros)
    state = {
        "Vm_prime":      Uniform(0, 0.05),
        "Vm_prime_prev": Uniform(0, 0.05),
        "Vf":            Choice([0.0]),
        "Vf_prev":       Choice([0.0]),
    }

    spiking = False  # output is analog (0..VDD) unless output_mode="spike"

    Cm  = NumberParam("Cm", low=0)
    Cf  = NumberParam("Cf", low=0)
    VDD = NumberParam("VDD", low=0)
    VTh = NumberParam("VTh", low=0)
    IR  = NumberParam("IR", low=0)

    #  new params
    leak  = NumberParam("leak", low=0)   # linear leak coefficient
    alpha = NumberParam("alpha", low=0)  # surrogate sharpness (used in TF)

    def __init__(
        self,
        Cm=1e-9, Cf=1e-9, VDD=1.8, VTh=0.6, IR=1e-9,
        leak=0.0,
        alpha=10.0,
        output_mode="analog"   # "analog" or "spike"
    ):
        super().__init__()
        self.Cm = Cm
        self.Cf = Cf
        self.VDD = VDD
        self.VTh = VTh
        self.IR = IR

        self.leak = leak
        self.alpha = alpha

        self.beta = Cm / (Cm + Cf)
        self.output_mode = output_mode  # "analog" or "spike"

    def step(self, dt, J, output, Vm_prime, Vm_prime_prev, Vf, Vf_prev):
        # hard gate (CPU reference)
        f = (Vm_prime >= self.VTh).astype(np.float32)
        dVf_dt = (Vf - Vf_prev) / dt
        Vf_new = Vf + dt * dVf_dt   

        dVm_prime_dt = (J - self.leak * Vm_prime - self.IR * f + self.Cf * dVf_dt) / self.Cm
        Vm_prime_new = Vm_prime + dt * dVm_prime_dt

        crossed = (Vm_prime_new >= self.VTh) & (Vm_prime < self.VTh)

        Vm_after = Vm_prime_new.copy()
        Vf_after = Vf_new.copy()    

        Vm_after[crossed] = self.beta * self.VDD
        Vf_after[crossed] = self.VDD * (1.0 - self.beta)

        not_crossed = ~crossed
        Vf_after[not_crossed] = Vm_after[not_crossed]  # Vf follows Vm' when no crossing



        f_after = (Vm_after >= self.VTh).astype(np.float32)

        if self.output_mode == "spike":
            output[:] = (1.0 / dt) * f_after
        else:
            output[:] = self.VDD * f_after

        Vm_prime_prev[:] = Vm_prime
        Vm_prime[:] = Vm_after
        Vf_prev[:] = Vf
        Vf[:] = Vf_after

    def gain_bias(self, max_rates, intercepts):
        return np.ones_like(max_rates), np.zeros_like(intercepts)
