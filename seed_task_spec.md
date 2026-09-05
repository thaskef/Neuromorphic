# Neuromorphic Co-Development: Seed Task Specification

**Status:** Draft  
**Date:** 2026-09-01  
**Phase:** MVE (Minimum Viable Experiment)  
**Priority:** Critical Path  

---

## Overview

The **delayed match-to-sample (DMS)** task is the seed task for the neuromorphic-LLM co-development framework. It is chosen because:

1. **Temporal processing requirement** — The network must maintain information across a delay period, which stresses the SNN's working memory capabilities and cannot be trivially solved by LLMs without external memory.
2. **Clear reward signal** — Binary success/failure provides unambiguous feedback for plasticity and LLM evaluation.
3. **Scalable complexity** — Can be made arbitrarily hard by increasing the delay, number of distractors, or feature space.
4. **Well-understood** — Extensively studied in neuroscience; known that recurrent networks with plasticity can learn it.
5. **Minimal sensory apparatus** — Requires only simple input encoding (no complex vision/language preprocessing needed for MVE).

---

## Task Description

### Trial Structure

```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│   Phase      │  Duration    │   Input      │  Required    │  Reward      │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Sample       │   T_sample   │   S (stim)   │  Encode      │   -         │
│ Delay 1      │   T_delay1   │   -          │  Maintain    │   -         │
│ Distractor 1 │   T_distr    │   D1 ≠ S     │  Ignore      │   -         │
│ Delay 2      │   T_delay2   │   -          │  Maintain    │   -         │
│ Test         │   T_test     │   S or D_new │  Match       │   +1 or -1  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### For MVE (Phase 1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `T_sample` | 100 ms | Long enough for stable encoding |
| `T_delay1` | 200 ms | Tests working memory |
| `T_distr` | 100 ms | Distractor presentation |
| `T_delay2` | 200 ms | Tests maintenance through interference |
| `T_test` | 100 ms | Match/non-match decision |
| **Total trial** | **700 ms** | Fits in reasonable simulation time |

### Stimulus Encoding

- **Stimulus space:** 10 binary features (2^10 = 1024 possible stimuli)
- **Encoding:** Each feature mapped to a sub-population of 100 neurons in the sensory input layer
- **Feature activation:** 50 Hz Poisson spike train for ON features, 0 Hz for OFF
- **Stimulus S:** Random 10-bit vector
- **Distractor D1:** Random 10-bit vector ≠ S (Hamming distance ≥ 1)
- **Test stimulus:** S (match) or random D_new ≠ S (non-match), each with p=0.5

### Network Input

```
Sensory Layer (1000 neurons) 
  ├─ Feature 0 (100 neurons)
  ├─ Feature 1 (100 neurons)
  │   ...
  └─ Feature 9 (100 neurons)
       
       ↓ spikes (encoded stimuli)
       
Associative Layer (8000 neurons, from genome)
  → Recurrent connectivity with STDP
  → Forms attractor states for remembered stimuli
       
       ↓ 
       
Memory Readout (via linear classifier on associative layer activity)
  → Match decision
```

### Reward Signal

- **+1 reward** if network correctly identifies match (test = S)
- **-1 reward** if network incorrectly identifies non-match (test = D_new)
- Reward delivered at end of test phase
- Neuromodulator (dopamine) released based on reward

---

## Network Architecture for MVE

### Populations

| Name | Size | Role | Neuron Type |
|------|------|------|-------------|
| `sensory` | 1000 | Input encoding | LIF (excitatory) |
| `associative` | 8000 | Working memory, pattern completion | LIF (excitatory) with adaptation |
| `inhibitory` | 2000 | Stabilization | LIF (inhibitory) |

### Connectivity

| Source | Target | Type | Plasticity | Probability | Weight |
|--------|--------|------|------------|-------------|--------|
| sensory | associative | Static | None | Full | 1.0 |
| associative | associative | STDP | Pair-based STDP | 0.02 | 0.5 |
| associative | inhibitory | STDP | Pair-based STDP | 0.05 | 0.8 |
| inhibitory | associative | Static | None | 0.05 | -1.5 |
| inhibitory | inhibitory | Static | None | 0.02 | -1.0 |

### Plasticity Rules

**STDP on associative↔associative:**
- A_plus = 0.1, A_minus = 0.1
- τ_plus = 20 ms, τ_minus = 20 ms
- w_min = 0.0, w_max = 2.0

**STDP on associative→inhibitory:**
- A_plus = 0.05, A_minus = 0.05
- τ_plus = 20 ms, τ_minus = 20 ms
- w_min = 0.0, w_max = 1.0

---

## Success Criteria

### For MVE (Genome + Compiler)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Network spiking** | > 5 Hz mean rate | Across associative population |
| **Sample encoding** | > 80% accuracy | Sensory → associative activation |
| **Delay maintenance** | > 50% accuracy | Pattern persistence across delay |
| **Distractor rejection** | > 70% accuracy | Non-match not recalled |
| **Full task accuracy** | > 60% | Across 100 trials |

### For LLM Guidance (Phase 2)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **LLM vs Random** | LLM > Random + 2σ | Fitness score comparison |
| **LLM vs Frozen** | LLM > Frozen + 2σ | Fitness score comparison |
| **Intervention quality** | > 50% accepted | % of LLM proposals that improve fitness |
| **Mechanistic hypotheses** | > 3 valid | LLM identifies correct mechanisms |

---

## Implementation Checklist

### Genome Specification

- [x] Neuron types: excitatory, inhibitory LIF
- [x] Synapse types: static, STDP
- [x] Populations: sensory (1000), associative (8000), inhibitory (2000)
- [x] Connectivity rules: all 4 connections
- [x] Plasticity rules: STDP for associative connections
- [x] Observables: firing rates, weights, topology

### Compiler Requirements

- [x] Support LIF neuron compilation
- [x] Support static synapse compilation
- [x] Support STDP synapse compilation (using Brian2's built-in mechanism)
- [x] Support sparse random connectivity
- [x] Spike monitors for all populations
- [x] Weight tracking for plastic synapses

### Task Interface

- [x] Stimulus encoder (10-bit → sensory spikes)
- [x] Trial scheduler (sample → delay → distractor → delay → test)
- [x] Reward calculator (match vs non-match)
- [x] Performance tracker (accuracy, reaction time)
- [ ] Neuromodulator release on reward

### LLM Interface

- [ ] Observable extractor (firing rates, weight means, topology)
- [ ] Genome snapshotter (for LLM inspection)
- [ ] Intervention applier (LLM proposals → genome changes)
- [ ] Hypothesis tracker (LLM reasoning → metadata)

---

## Next Steps

1. **Complete compiler STDP implementation** — Use Brian2's `ExponentialSTDP` or implement custom STDP in synapse equations
2. **Build stimulus encoder** — Map 10-bit vectors to Poisson spike trains in sensory layer
3. **Implement trial runner** — Orchestrate trial phases, inject stimuli, deliver reward
4. **Create evaluation harness** — Run trials, compute accuracy, return observables to LLM
5. **Test with hand-coded genome** — Verify task is learnable before LLM guidance

---

## References

- **Brian2 STDP example:** `examples/synapses/STDP.py` in Brian2 distribution
- **DMS in spiking networks:** Deco & Rolls (2005), "A cortical mechanism for short-term memory"
- **Neuromorphic DMS:** essentials et al., Loihi implementation of working memory tasks
