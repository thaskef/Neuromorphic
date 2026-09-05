"""Closed intervention DSL: the only way the LLM may modify the genome.

Ported from the earlier ``legacy_vibe_web/interventions.py`` (G3) and adapted to
the canonical ``DevelopmentalGenome`` schema. Each intervention is a typed,
validated verb. Applying an intervention returns a new genome (new ``genome_id``,
``parent_genome`` set to the input's id). Invalid proposals raise a ValueError
whose message is designed to be fed back to the LLM so it can self-correct.

The verb set is deliberately small and closed. The LLM proposes *developmental*
interventions (change a rule, add a population, rewire a connection class), never
individual weight edits.

Schema mapping (legacy -> canonical):

- legacy ``Genome.populations[id]``           -> ``populations[name]`` (PopulationSpec)
- legacy ``Genome.synapses[(src, tgt)]``      -> ``connectivity_rules[(src, tgt)]``
  (ConnectivityRule) which references a ``synapse_types[name]`` (SynapseTypeSpec)
- legacy ``Genome.rules.neuromodulators``     -> ``neuromodulators`` (NeuromodulatorSpec)
- legacy ``Genome.rules.gating``              -> ``attention_gating`` (AttentionGatingSpec)
- legacy ``Genome.rules.temporal_dynamics``   -> ``temporal_dynamics`` (TemporalDynamicsSpec)
- legacy ``Genome.rules.pruning_rate``        -> ``pruning_rules`` (PruningRule)
- legacy ``Genome.rules.learning_rate``       -> ``attention_gating.learning_rate``

A legacy "synapse" conflated *connection* (source->target) with *type*
(excitatory/inhibitory + plasticity). The canonical schema separates these, so
the connection verbs operate on ``connectivity_rules`` and, where a type is
needed, ensure a matching ``SynapseTypeSpec`` exists.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from genome_schema import (
    ConnectivityPattern,
    ConnectivityRule,
    DevelopmentalGenome,
    NeuromodulatorSpec,
    NeuromodulatorType,
    NeuronModel,
    NeuronTypeSpec,
    PopulationSpec,
    PruningRule,
    PruningStrategy,
    SynapseType,
    SynapseTypeSpec,
)

# --- Verb payloads ----------------------------------------------------------


class SetParameter(BaseModel):
    """Change one scalar parameter of a population, connection, or rule.

    ``target`` is one of:
      * ``"<population>"``            -> a PopulationSpec field
      * ``"<src>-><tgt>"``            -> a ConnectivityRule field
      * ``"synapse_type:<name>"``     -> a SynapseTypeSpec field
      * ``"plasticity:<name>"``       -> a PlasticityRule field
      * ``"global"``                  -> an attention_gating / temporal_dynamics field
    """

    verb: Literal["set_parameter"] = "set_parameter"
    target: str
    parameter: str
    value: float


class AddPopulation(BaseModel):
    """Neurogenesis: create a new neuron population."""

    verb: Literal["add_population"] = "add_population"
    id: str
    neuron_type: str = "LIF"
    size: int = Field(gt=0)
    params: dict[str, float] = Field(default_factory=dict)


class RemovePopulation(BaseModel):
    """Cell death: remove a population and all connections touching it."""

    verb: Literal["remove_population"] = "remove_population"
    id: str


class AddSynapseType(BaseModel):
    """Create a new connection class between two populations."""

    verb: Literal["add_synapse_type"] = "add_synapse_type"
    source: str
    target: str
    synapse_type: str = "excitatory"  # excitatory | inhibitory
    plasticity: str = "static"        # static | stdp | three_factor
    params: dict[str, float] = Field(default_factory=dict)


class RemoveSynapseType(BaseModel):
    """Strip a connection class (structural pruning at the class level)."""

    verb: Literal["remove_synapse_type"] = "remove_synapse_type"
    source: str
    target: str


class SetPlasticityRule(BaseModel):
    """Switch or parameterize a connection class's plasticity rule."""

    verb: Literal["set_plasticity_rule"] = "set_plasticity_rule"
    source: str
    target: str
    plasticity: str  # static | stdp | three_factor | ...
    params: dict[str, float] = Field(default_factory=dict)


class SetConnectivityRule(BaseModel):
    """Rewire a connection class (probability, weight, pattern, sign)."""

    verb: Literal["set_connectivity_rule"] = "set_connectivity_rule"
    source: str
    target: str
    params: dict[str, float] = Field(default_factory=dict)


class SetNeuromodulatorSchedule(BaseModel):
    """Add or reconfigure a global neuromodulator signal."""

    verb: Literal["set_neuromodulator_schedule"] = "set_neuromodulator_schedule"
    id: str
    baseline: float = 0.0
    time_constant_ms: float = 50.0


class SetLearningRateSchedule(BaseModel):
    """Change the global learning rate (attention_gating.learning_rate)."""

    verb: Literal["set_learning_rate_schedule"] = "set_learning_rate_schedule"
    learning_rate: float


class Prune(BaseModel):
    """Adjust the global pruning rate (a PruningRule.fraction)."""

    verb: Literal["prune"] = "prune"
    pruning_rate: float = Field(ge=0.0)


class AddGate(BaseModel):
    """Add a context-gating population to the attention-gating spec."""

    verb: Literal["add_gate"] = "add_gate"
    name: str
    strength: float


class SetTemporalDynamics(BaseModel):
    """Modify the predictive-retention policy (hibernation vs. abandonment vs. resurrection)."""

    verb: Literal["set_temporal_dynamics"] = "set_temporal_dynamics"
    field: str
    value: float


Intervention = Annotated[
    Union[
        SetParameter,
        AddPopulation,
        RemovePopulation,
        AddSynapseType,
        RemoveSynapseType,
        SetPlasticityRule,
        SetConnectivityRule,
        SetNeuromodulatorSchedule,
        SetLearningRateSchedule,
        Prune,
        AddGate,
        SetTemporalDynamics,
    ],
    Field(discriminator="verb"),
]

# --- Helpers ----------------------------------------------------------------

# Legacy field-name aliases for temporal_dynamics (legacy TemporalDynamicsRule
# used "hibernation/abandonment/prune" thresholds; the canonical schema names
# them "latent/archive/prune").
_TEMPORAL_ALIASES = {
    "hibernation_threshold_days": "latent_threshold_days",
    "abandonment_threshold_days": "archive_threshold_days",
    "prune_threshold_days": "prune_threshold_days",
}

# legacy plasticity string -> canonical SynapseType
_PLASTICITY_MAP = {
    "static": SynapseType.STATIC,
    "stdp": SynapseType.STD,
    "three_factor": SynapseType.THREE_FACTOR,
    "hebbian": SynapseType.HEBBIAN,
}


def _find_population(g: DevelopmentalGenome, name: str) -> PopulationSpec | None:
    return next((p for p in g.populations if p.name == name), None)


def _find_connectivity(g: DevelopmentalGenome, source: str, target: str) -> ConnectivityRule | None:
    return next((c for c in g.connectivity_rules if c.source == source and c.target == target), None)


def _find_synapse_type(g: DevelopmentalGenome, name: str) -> SynapseTypeSpec | None:
    return next((s for s in g.synapse_types if s.name == name), None)


def _find_neuron_type(g: DevelopmentalGenome, name: str) -> NeuronTypeSpec | None:
    return next((nt for nt in g.neuron_types if nt.name == name), None)


def _default_synapse_type_name(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def _new_child(parent: DevelopmentalGenome) -> DevelopmentalGenome:
    """Deep-copy the genome and mark it as a child of ``parent``."""
    out = parent.model_copy(deep=True)
    out.parent_genome = parent.genome_id
    out.genome_id = f"genome_{uuid.uuid4().hex[:8]}"
    return out


# --- Application ------------------------------------------------------------


def apply_intervention(genome: DevelopmentalGenome, intervention: Intervention) -> DevelopmentalGenome:
    """Apply one intervention, returning a new (deep-copied) genome.

    The input genome is never mutated. Raises ValueError on an invalid target so
    the harness can feed the message back to the LLM.
    """
    out = _new_child(genome)
    v = intervention

    if isinstance(v, SetParameter):
        if v.target == "global":
            # Try attention_gating then temporal_dynamics.
            if hasattr(out.attention_gating, v.parameter):
                setattr(out.attention_gating, v.parameter, v.value)
            elif hasattr(out.temporal_dynamics, v.parameter):
                setattr(out.temporal_dynamics, v.parameter, v.value)
            else:
                raise ValueError(f"unknown global field '{v.parameter}'")
        elif "synapse_type:" in v.target:
            name = v.target.split(":", 1)[1]
            st = _find_synapse_type(out, name)
            if st is None:
                raise ValueError(f"unknown synapse type '{name}'")
            setattr(st, v.parameter, v.value)
        elif "plasticity:" in v.target:
            name = v.target.split(":", 1)[1]
            pr = next((r for r in out.plasticity_rules if r.name == name), None)
            if pr is None:
                raise ValueError(f"unknown plasticity rule '{name}'")
            setattr(pr, v.parameter, v.value)
        elif "->" in v.target:
            src, tgt = v.target.split("->")
            conn = _find_connectivity(out, src, tgt)
            if conn is None:
                raise ValueError(f"unknown connection '{v.target}'")
            setattr(conn, v.parameter, v.value)
        else:
            pop = _find_population(out, v.target)
            if pop is None:
                raise ValueError(f"unknown population '{v.target}'")
            setattr(pop, v.parameter, v.value)

    elif isinstance(v, AddPopulation):
        if _find_population(out, v.id) is not None:
            raise ValueError(f"population '{v.id}' already exists")
        # Ensure the referenced neuron type exists; if not, create a default LIF
        # type (matching the legacy DSL's implicit default).
        if _find_neuron_type(out, v.neuron_type) is None:
            out.neuron_types.append(NeuronTypeSpec(name=v.neuron_type, model=NeuronModel.LIF))
        out.populations.append(
            PopulationSpec(
                name=v.id,
                neuron_type=v.neuron_type,
                size=v.size,
                role=v.params.get("role", "general"),
                noise_std=v.params.get("noise_std", 0.0),
            )
        )

    elif isinstance(v, RemovePopulation):
        before = len(out.populations)
        out.populations = [p for p in out.populations if p.name != v.id]
        if len(out.populations) == before:
            raise ValueError(f"population '{v.id}' does not exist")
        out.connectivity_rules = [
            c for c in out.connectivity_rules if c.source != v.id and c.target != v.id
        ]

    elif isinstance(v, AddSynapseType):
        if _find_connectivity(out, v.source, v.target) is not None:
            raise ValueError(f"connection '{v.source}->{v.target}' already exists")
        if _find_population(out, v.source) is None or _find_population(out, v.target) is None:
            raise ValueError(f"connection references unknown population")

        sign = -1.0 if v.synapse_type == "inhibitory" else 1.0
        plasticity = _PLASTICITY_MAP.get(v.plasticity, SynapseType.STATIC)
        w = v.params.get("w", v.params.get("weight", 1.0)) * sign

        st_name = _default_synapse_type_name(v.source, v.target)
        if _find_synapse_type(out, st_name) is None:
            out.synapse_types.append(
                SynapseTypeSpec(
                    name=st_name,
                    synapse_type=plasticity,
                    weight=w,
                    tau_plus=v.params.get("tau_pre", v.params.get("tau_plus")),
                    tau_minus=v.params.get("tau_post", v.params.get("tau_minus")),
                    a_plus=v.params.get("A_plus"),
                    a_minus=abs(v.params["A_minus"]) if "A_minus" in v.params else None,
                    w_min=v.params.get("w_min"),
                    w_max=v.params.get("w_max"),
                )
            )

        out.connectivity_rules.append(
            ConnectivityRule(
                name=st_name,
                source=v.source,
                target=v.target,
                pattern=ConnectivityPattern(v.params.get("pattern", "sparse_random")),
                probability=v.params.get("p", v.params.get("probability")),
                synapse_type=st_name,
                weight_mean=w,
            )
        )

    elif isinstance(v, RemoveSynapseType):
        before = len(out.connectivity_rules)
        out.connectivity_rules = [
            c for c in out.connectivity_rules if (c.source, c.target) != (v.source, v.target)
        ]
        if len(out.connectivity_rules) == before:
            raise ValueError(f"connection '{v.source}->{v.target}' does not exist")

    elif isinstance(v, SetPlasticityRule):
        conn = _find_connectivity(out, v.source, v.target)
        if conn is None:
            raise ValueError(f"unknown connection '{v.source}->{v.target}'")
        st = _find_synapse_type(out, conn.synapse_type)
        if st is None:
            raise ValueError(f"connection '{conn.name}' references unknown synapse type '{conn.synapse_type}'")
        st.synapse_type = _PLASTICITY_MAP.get(v.plasticity, st.synapse_type)
        for key, val in v.params.items():
            setattr(st, key, val)

    elif isinstance(v, SetConnectivityRule):
        conn = _find_connectivity(out, v.source, v.target)
        if conn is None:
            raise ValueError(f"unknown connection '{v.source}->{v.target}'")
        if "pattern" in v.params:
            conn.pattern = ConnectivityPattern(v.params.pop("pattern"))
        for key, val in v.params.items():
            setattr(conn, key, val)

    elif isinstance(v, SetNeuromodulatorSchedule):
        mod = next((m for m in out.neuromodulators if m.name == v.id), None)
        if mod is None:
            out.neuromodulators.append(
                NeuromodulatorSpec(
                    name=v.id,
                    neuromodulator_type=NeuromodulatorType.DOPAMINE if v.id == "dopamine" else NeuromodulatorType.CUSTOM,
                    baseline=v.baseline,
                    tau_decay=v.time_constant_ms,
                )
            )
        else:
            mod.baseline = v.baseline
            mod.tau_decay = v.time_constant_ms

    elif isinstance(v, SetLearningRateSchedule):
        out.attention_gating.learning_rate = v.learning_rate

    elif isinstance(v, Prune):
        rule = next((r for r in out.pruning_rules if r.strategy == PruningStrategy.MAGNITUDE), None)
        if rule is None:
            out.pruning_rules.append(
                PruningRule(name="prune_magnitude", strategy=PruningStrategy.MAGNITUDE, fraction=v.pruning_rate)
            )
        else:
            rule.fraction = v.pruning_rate

    elif isinstance(v, AddGate):
        if v.name not in out.attention_gating.gates_populations:
            out.attention_gating.gates_populations.append(v.name)

    elif isinstance(v, SetTemporalDynamics):
        field = _TEMPORAL_ALIASES.get(v.field, v.field)
        if not hasattr(out.temporal_dynamics, field):
            raise ValueError(f"unknown temporal_dynamics field '{v.field}'")
        setattr(out.temporal_dynamics, field, v.value)

    else:  # pragma: no cover - exhaustive by construction
        raise ValueError(f"unhandled intervention verb: {type(v).__name__}")

    # Re-validate so enum coercion and cross-reference checks run. This catches
    # cases like a dangling synapse reference or an out-of-order threshold.
    return DevelopmentalGenome.model_validate(out.model_dump())


_intervention_adapter = TypeAdapter(Intervention)


def parse_intervention(payload: str | dict) -> Intervention:
    """Parse a raw LLM proposal into a validated Intervention, or raise a
    ValidationError whose message the LLM can use to self-correct."""
    if isinstance(payload, str):
        return _intervention_adapter.validate_json(payload)
    return _intervention_adapter.validate_python(payload)


def list_verbs() -> list[str]:
    """The closed verb set, for building the LLM prompt's function schema."""
    return sorted({m.model_fields["verb"].default for m in _VERB_MODELS})


_VERB_MODELS = [
    SetParameter,
    AddPopulation,
    RemovePopulation,
    AddSynapseType,
    RemoveSynapseType,
    SetPlasticityRule,
    SetConnectivityRule,
    SetNeuromodulatorSchedule,
    SetLearningRateSchedule,
    Prune,
    AddGate,
    SetTemporalDynamics,
]
