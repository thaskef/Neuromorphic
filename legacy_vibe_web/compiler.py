"""GenomeCompiler: turn a Genome into a runnable Brian2 network.

This is the other half of the executable contract (G2). The LLM edits the genome;
this module turns the genome into concrete Brian2 objects and runs them.

Scope of this first pass (the MVE substrate):
  * neuron types: LIF, AdaptiveLIF
  * synapse types: excitatory, inhibitory (weight sign); modulatory -> NotImplemented
  * plasticity: static, stdp, three_factor (STDP gated by a neuromodulator scalar)

Deferred to the harness / later phases (and raised clearly, not silently ignored):
  * Izhikevich / Hodgkin-Huxley neurons
  * dynamic neuromodulator schedules (here the neuromodulator is a fixed baseline)
  * neurogenesis (population resize), structural pruning, homeostatic scaling
"""

from __future__ import annotations

from typing import Any

import numpy as np
from brian2 import (
    Network,
    NeuronGroup,
    PoissonGroup,
    PopulationRateMonitor,
    SpikeMonitor,
    Synapses,
    start_scope,
)
from brian2.units import hertz, ms, mV

from genome import Genome, PopulationSpec, SynapseSpec

# --- Neuron equations -------------------------------------------------------

LIF_EQ = """
dv/dt = (EL - v + I_syn) / tau_m : volt (unless refractory)
dI_syn/dt = -I_syn / tau_syn : volt
tau_m : second
tau_syn : second
EL : volt
Vt : volt
"""

ADAPTIVE_LIF_EQ = """
dv/dt = (EL - v + I_syn - w) / tau_m : volt (unless refractory)
dI_syn/dt = -I_syn / tau_syn : volt
dw/dt = -w / tau_w : volt
tau_m : second
tau_syn : second
tau_w : second
EL : volt
Vt : volt
b : volt
"""

# --- Synapse models ---------------------------------------------------------

STATIC_MODEL = "w_syn : volt"
STATIC_ON_PRE = "I_syn_post += w_syn"

STDP_MODEL = """
w_syn : volt
dApre/dt = -Apre / tau_pre : volt (event-driven)
dApost/dt = -Apost / tau_post : volt (event-driven)
"""
# A_plus > 0 (LTP on pre-before-post); A_minus < 0 (LTD on post-before-pre).
STDP_ON_PRE = """
I_syn_post += w_syn
Apre += A_plus
w_syn = clip(w_syn + Apost, w_min, w_max)
"""
STDP_ON_POST = """
Apost += A_minus
w_syn = clip(w_syn + Apre, w_min, w_max)
"""

# Three-factor: identical STDP, but the plasticity update is scaled by a global
# neuromodulator scalar D (dopamine baseline in this pass).
THREE_FACTOR_MODEL = STDP_MODEL
THREE_FACTOR_ON_PRE = """
I_syn_post += w_syn
Apre += A_plus
w_syn = clip(w_syn + Apost * (1.0 + D), w_min, w_max)
"""
THREE_FACTOR_ON_POST = """
Apost += A_minus
w_syn = clip(w_syn + Apre * (1.0 + D), w_min, w_max)
"""


def _neuromodulator_level(genome: Genome) -> float:
    """Return the dopamine baseline (or 0.0) as the three-factor gate D."""
    for m in genome.rules.neuromodulators:
        if m.id == "dopamine":
            return m.baseline
    if genome.rules.neuromodulators:
        return genome.rules.neuromodulators[0].baseline
    return 0.0


def _make_group(pop: PopulationSpec) -> NeuronGroup:
    p = pop.params
    if pop.neuron_type == "LIF":
        eqs, reset = LIF_EQ, "v = EL"
    elif pop.neuron_type == "AdaptiveLIF":
        eqs, reset = ADAPTIVE_LIF_EQ, "v = EL; w += b"
    else:
        raise NotImplementedError(
            f"neuron type '{pop.neuron_type}' is not yet implemented (supported: LIF, AdaptiveLIF)"
        )

    G = NeuronGroup(
        pop.size,
        eqs,
        threshold="v > Vt",
        reset=reset,
        refractory="2*ms",
        method="exact",
        name=pop.id,
    )
    G.tau_m = p.get("tau_m", 20.0) * ms
    G.tau_syn = p.get("tau_syn", 5.0) * ms
    G.EL = p.get("EL", -70.0) * mV
    G.Vt = p.get("Vt", -50.0) * mV
    G.v = G.EL
    G.I_syn = 0 * mV
    if pop.neuron_type == "AdaptiveLIF":
        G.tau_w = p.get("tau_w", 100.0) * ms
        G.b = p.get("b", 2.0) * mV
        G.w = 0 * mV
    return G


def _make_synapse(
    spec: SynapseSpec, src: NeuronGroup, tgt: NeuronGroup, genome: Genome
) -> Synapses:
    p = spec.params
    self_loop = src is tgt

    if spec.synapse_type == "modulatory":
        raise NotImplementedError("modulatory synapses are deferred to the harness phase")

    if spec.plasticity == "static":
        S = Synapses(src, tgt, model=STATIC_MODEL, on_pre=STATIC_ON_PRE)
        S.connect(p=p.get("p", 0.1), condition="i != j" if self_loop else None)
        S.w_syn = p.get("w", 1.0) * mV
        return S

    if spec.plasticity in ("stdp", "three_factor"):
        # Scalar constants are passed via `namespace` (shared across the whole
        # connection class), not as per-synapse state variables. This is the
        # canonical Brian2 STDP pattern and keeps identifiers referenced inside
        # the equations resolvable.
        namespace = {
            "tau_pre": p.get("tau_pre", 20.0) * ms,
            "tau_post": p.get("tau_post", 20.0) * ms,
            "A_plus": p.get("A_plus", 0.1) * mV,
            "A_minus": p.get("A_minus", -0.08) * mV,
            "w_max": p.get("w_max", 5.0) * mV,
            "w_min": p.get("w_min", 0.0) * mV,
        }
        if spec.plasticity == "three_factor":
            model, on_pre, on_post = THREE_FACTOR_MODEL, THREE_FACTOR_ON_PRE, THREE_FACTOR_ON_POST
            namespace["D"] = _neuromodulator_level(genome)
        else:
            model, on_pre, on_post = STDP_MODEL, STDP_ON_PRE, STDP_ON_POST

        S = Synapses(src, tgt, model=model, on_pre=on_pre, on_post=on_post, namespace=namespace)
        S.connect(p=p.get("p", 0.05), condition="i != j" if self_loop else None)
        S.w_syn = p.get("w", 0.5) * mV
        S.Apre = 0 * mV
        S.Apost = 0 * mV
        return S

    raise NotImplementedError(
        f"plasticity rule '{spec.plasticity}' is not yet implemented (supported: static, stdp, three_factor)"
    )


def compile_genome(genome: Genome) -> dict[str, Any]:
    """Build the Brian2 network described by a genome.

    Returns a dict of the network, groups, synapses, and monitors so the harness
    can inspect them and run additional operations.
    """
    start_scope()

    groups = {pop.id: _make_group(pop) for pop in genome.populations}
    synapses = [
        _make_synapse(spec, groups[spec.source], groups[spec.target], genome)
        for spec in genome.synapses
    ]

    spike_monitors = {pid: SpikeMonitor(g) for pid, g in groups.items()}
    rate_monitors = {pid: PopulationRateMonitor(g) for pid, g in groups.items()}

    net = Network()
    net.add(*groups.values(), *synapses, *spike_monitors.values(), *rate_monitors.values())

    return {
        "network": net,
        "groups": groups,
        "synapses": synapses,
        "spike_monitors": spike_monitors,
        "rate_monitors": rate_monitors,
    }


def run_genome(
    genome: Genome,
    duration_ms: float = 100.0,
    input_rate_hz: float = 20.0,
    input_weight_mv: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    """Compile and run a genome, driving its first population with Poisson spikes.

    Returns per-population mean firing rate (Hz) over the run. This is the
    substrate half of the loop: the harness runs it once per episode and feeds the
    rates back to the LLM as observables.
    """
    np.random.seed(seed)
    compiled = compile_genome(genome)
    net: Network = compiled["network"]
    groups: dict[str, NeuronGroup] = compiled["groups"]
    rate_monitors: dict[str, PopulationRateMonitor] = compiled["rate_monitors"]

    first_pop = genome.populations[0]
    sensory = groups[first_pop.id]
    stim = PoissonGroup(sensory.N, rates=input_rate_hz * hertz)
    stim_syn = Synapses(
        stim, sensory, model="w_in : volt", on_pre="I_syn_post += w_in"
    )
    stim_syn.connect(p=0.5)
    stim_syn.w_in = input_weight_mv * mV

    net.add(stim, stim_syn)
    net.run(duration_ms * ms)

    rates: dict[str, float] = {}
    for pid, mon in rate_monitors.items():
        # `mon.rate` is a VariableView; index it to get the underlying array.
        rates[pid] = float(mon.rate[:].mean() / hertz)
    return rates
