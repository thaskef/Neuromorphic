"""Developmental genome: the serializable contract between the LLM society and the SNN substrate.

The LLM never edits synaptic weights directly. It edits this genome — a typed,
versioned specification of populations, synapses, and developmental rules. A
GenomeCompiler (compiler.py) turns the genome into a runnable simulator network.

The genome is the single most important artifact in the co-development loop: it is
the *only* thing the LLM reads and writes, and the only thing the compiler reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

# --- Supported primitives ---------------------------------------------------
# These are the known values the compiler can execute. The LLM may propose new
# ones; unknown values surface as a clear compiler error, not a silent no-op.

NEURON_TYPES = ("LIF", "AdaptiveLIF", "Izhikevich")
SYNAPSE_TYPES = ("excitatory", "inhibitory", "modulatory")
PLASTICITY_RULES = ("static", "stdp", "three_factor", "homeostatic")
REACTIVATION_PREDICTORS = ("none", "seasonal", "learned")


class PopulationSpec(BaseModel):
    """A homogeneous population of neurons."""

    id: str = Field(..., description="Unique population identifier")
    neuron_type: str = "LIF"
    size: int = Field(gt=0)
    params: dict[str, float] = Field(default_factory=dict)


class SynapseSpec(BaseModel):
    """A class of connections between two populations."""

    source: str
    target: str
    synapse_type: str = "excitatory"
    plasticity: str = "static"
    params: dict[str, float] = Field(default_factory=dict)


class NeuromodulatorSpec(BaseModel):
    """A global signal (dopamine / acetylcholine / serotonin analogue) that gates plasticity."""

    id: str
    baseline: float = 0.0
    time_constant_ms: float = 50.0


class RetentionTier(BaseModel):
    """One rung of the predictive-retention ladder (active -> latent -> archived -> pruned)."""

    name: str
    access_latency_ms: float = 0.0
    trigger: str = ""


class TemporalDynamicsRule(BaseModel):
    """Predictive retention: governs hibernation vs. abandonment vs. resurrection.

    This is the genome-level answer to "will I need this topic again?". A static
    time/usage pruner cannot distinguish a temporarily dormant topic from a
    permanently abandoned one, so retention is expressed as a learnable policy.
    """

    hibernation_threshold_days: float = 180.0  # demote to latent after this much inactivity
    abandonment_threshold_days: float = 365.0  # archive after this much
    prune_threshold_days: float = 730.0        # prune only after this much AND never reactivated
    recency_weight: float = 0.5
    frequency_weight: float = 0.3
    semantic_similarity_weight: float = 0.2
    prewarm_probability: float = 0.5            # predicted P(reactivation) that triggers pre-warming
    reactivation_predictor: str = "learned"     # none | seasonal | learned
    tiers: list[RetentionTier] = Field(
        default_factory=lambda: [
            RetentionTier(name="active", access_latency_ms=0.0),
            RetentionTier(name="latent", access_latency_ms=2000.0),
            RetentionTier(name="archived", access_latency_ms=10000.0),
            RetentionTier(name="pruned", access_latency_ms=0.0, trigger="rebuild"),
        ]
    )

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "TemporalDynamicsRule":
        if not (self.hibernation_threshold_days < self.abandonment_threshold_days < self.prune_threshold_days):
            raise ValueError(
                "retention thresholds must be ordered: hibernation < abandonment < prune"
            )
        return self

    @model_validator(mode="after")
    def _predictor_known(self) -> "TemporalDynamicsRule":
        if self.reactivation_predictor not in REACTIVATION_PREDICTORS:
            raise ValueError(
                f"reactivation_predictor must be one of {REACTIVATION_PREDICTORS}"
            )
        return self


class GlobalRules(BaseModel):
    """Global, population-independent developmental rules."""

    neuromodulators: list[NeuromodulatorSpec] = Field(default_factory=list)
    learning_rate: float = 1e-3
    neurogenesis_rate: float = 0.0
    pruning_rate: float = 0.0
    homeostatic_target_rate: float | None = None
    consolidation_enabled: bool = False
    gating: dict[str, float] = Field(default_factory=dict)
    temporal_dynamics: TemporalDynamicsRule = Field(default_factory=TemporalDynamicsRule)


class Genome(BaseModel):
    """A versioned developmental genome."""

    version: int = 1
    populations: list[PopulationSpec] = Field(default_factory=list)
    synapses: list[SynapseSpec] = Field(default_factory=list)
    rules: GlobalRules = Field(default_factory=GlobalRules)

    @model_validator(mode="after")
    def _validate_references(self) -> "Genome":
        ids = [p.id for p in self.populations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"population ids must be unique, got {ids}")
        for s in self.synapses:
            if s.source not in ids:
                raise ValueError(f"synapse source '{s.source}' is not a known population")
            if s.target not in ids:
                raise ValueError(f"synapse target '{s.target}' is not a known population")
        return self

    # --- serialization ------------------------------------------------------

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Genome":
        return cls.model_validate_json(text)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # --- versioning ---------------------------------------------------------

    def next_version(self) -> "Genome":
        """Return a copy with the version incremented (the post-intervention genome)."""
        out = self.model_copy(deep=True)
        out.version = self.version + 1
        return out

    def diff(self, other: "Genome") -> dict[str, Any]:
        """Structural diff used to describe an intervention to the LLM and to roll back."""
        a_pops = {p.id: p for p in self.populations}
        b_pops = {p.id: p for p in other.populations}
        a_syns = {(s.source, s.target): s for s in self.synapses}
        b_syns = {(s.source, s.target): s for s in other.synapses}

        return {
            "populations": {
                "added": [pid for pid in b_pops if pid not in a_pops],
                "removed": [pid for pid in a_pops if pid not in b_pops],
                "changed": [
                    pid for pid in a_pops if pid in b_pops and a_pops[pid] != b_pops[pid]
                ],
            },
            "synapses": {
                "added": [k for k in b_syns if k not in a_syns],
                "removed": [k for k in a_syns if k not in b_syns],
                "changed": [
                    k for k in a_syns if k in b_syns and a_syns[k] != b_syns[k]
                ],
            },
            "rules_changed": self.rules != other.rules,
        }


class GenomeHistory:
    """Append-only history of genome versions with rollback."""

    def __init__(self) -> None:
        self._versions: list[Genome] = []

    def record(self, genome: Genome) -> None:
        self._versions.append(genome.model_copy(deep=True))

    @property
    def current(self) -> Genome:
        if not self._versions:
            raise IndexError("history is empty")
        return self._versions[-1]

    @property
    def n(self) -> int:
        return len(self._versions)

    def rollback(self, n: int = 1) -> Genome:
        if n >= len(self._versions):
            raise ValueError(f"cannot roll back {n} of {len(self._versions)} versions")
        del self._versions[-n:]
        return self.current

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([g.model_dump() for g in self._versions], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GenomeHistory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        h = cls()
        h._versions = [Genome.model_validate(d) for d in data]
        return h


def seed_genome(n_sensory: int = 1000, n_recurrent: int = 5000, n_readout: int = 500) -> Genome:
    """The canonical MVE substrate: sensory -> recurrent -> readout, with STDP
    in the recurrent layer and a dopamine-gated (three-factor) readout path."""
    return Genome(
        version=1,
        populations=[
            PopulationSpec(id="sensory", neuron_type="LIF", size=n_sensory),
            PopulationSpec(
                id="recurrent",
                neuron_type="AdaptiveLIF",
                size=n_recurrent,
                params={"tau_w": 100.0, "b": 2.0},
            ),
            PopulationSpec(id="readout", neuron_type="LIF", size=n_readout),
        ],
        synapses=[
            SynapseSpec(
                source="sensory", target="recurrent",
                synapse_type="excitatory", plasticity="static",
                params={"p": 0.1, "w": 1.0},
            ),
            SynapseSpec(
                source="recurrent", target="recurrent",
                synapse_type="excitatory", plasticity="stdp",
                params={"p": 0.05, "w": 0.5, "A_plus": 0.1, "A_minus": -0.08,
                        "w_max": 5.0, "w_min": 0.0},
            ),
            SynapseSpec(
                source="recurrent", target="readout",
                synapse_type="excitatory", plasticity="three_factor",
                params={"p": 0.2, "w": 0.5, "A_plus": 0.1, "A_minus": -0.08,
                        "w_max": 5.0, "w_min": 0.0},
            ),
            SynapseSpec(
                source="readout", target="recurrent",
                synapse_type="inhibitory", plasticity="static",
                params={"p": 0.1, "w": -1.0},
            ),
        ],
        rules=GlobalRules(
            neuromodulators=[NeuromodulatorSpec(id="dopamine", baseline=0.0, time_constant_ms=50.0)],
            learning_rate=1e-3,
            temporal_dynamics=TemporalDynamicsRule(),
        ),
    )
