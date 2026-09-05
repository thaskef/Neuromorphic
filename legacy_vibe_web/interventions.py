"""Closed intervention DSL: the only way the LLM may modify the genome.

Each intervention is a typed, validated verb. Applying an intervention returns a
new genome (version + 1). Invalid proposals raise a ValidationError whose message
is designed to be fed back to the LLM so it can self-correct (G3).

The verb set is deliberately small and closed. The LLM proposes *developmental*
interventions (change a rule, add a population, rewire a connection class), never
individual weight edits.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from genome import Genome, PopulationSpec, SynapseSpec, NeuromodulatorSpec

# --- Verb payloads ----------------------------------------------------------


class SetParameter(BaseModel):
    """Change one scalar parameter of a population, synapse, or global rule."""

    verb: Literal["set_parameter"] = "set_parameter"
    target: str = Field(..., description="population id | synapse 'src->tgt' | 'rules'")
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
    """Cell death: remove a population and all synapses touching it."""

    verb: Literal["remove_population"] = "remove_population"
    id: str


class AddSynapseType(BaseModel):
    """Create a new connection class between two populations."""

    verb: Literal["add_synapse_type"] = "add_synapse_type"
    source: str
    target: str
    synapse_type: str = "excitatory"
    plasticity: str = "static"
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
    plasticity: str
    params: dict[str, float] = Field(default_factory=dict)


class SetConnectivityRule(BaseModel):
    """Rewire a connection class (connection probability, weight, sign)."""

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
    """Change the global learning rate."""

    verb: Literal["set_learning_rate_schedule"] = "set_learning_rate_schedule"
    learning_rate: float


class Prune(BaseModel):
    """Adjust the global pruning rate."""

    verb: Literal["prune"] = "prune"
    pruning_rate: float = Field(ge=0.0)


class AddGate(BaseModel):
    """Add a context-gating signal (name -> strength)."""

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

# --- Application ------------------------------------------------------------


def _synapse_index(g: Genome) -> dict[tuple[str, str], SynapseSpec]:
    return {(s.source, s.target): s for s in g.synapses}


def apply_intervention(genome: Genome, intervention: Intervention) -> Genome:
    """Apply one intervention, returning a new (version-incremented) genome.

    The input genome is never mutated. Raises ValueError on an invalid target so
    the harness can feed the message back to the LLM.
    """
    out = genome.model_copy(deep=True)
    v = intervention

    if isinstance(v, SetParameter):
        if v.target == "rules":
            if not hasattr(out.rules, v.parameter):
                raise ValueError(f"unknown global rule '{v.parameter}'")
            setattr(out.rules, v.parameter, v.value)
        elif "->" in v.target:
            key = tuple(v.target.split("->"))
            syn = _synapse_index(out).get(key)
            if syn is None:
                raise ValueError(f"unknown synapse '{v.target}'")
            syn.params[v.parameter] = v.value
        else:
            pop = next((p for p in out.populations if p.id == v.target), None)
            if pop is None:
                raise ValueError(f"unknown population '{v.target}'")
            pop.params[v.parameter] = v.value

    elif isinstance(v, AddPopulation):
        if any(p.id == v.id for p in out.populations):
            raise ValueError(f"population '{v.id}' already exists")
        out.populations.append(
            PopulationSpec(id=v.id, neuron_type=v.neuron_type, size=v.size, params=v.params)
        )

    elif isinstance(v, RemovePopulation):
        before = len(out.populations)
        out.populations = [p for p in out.populations if p.id != v.id]
        if len(out.populations) == before:
            raise ValueError(f"population '{v.id}' does not exist")
        out.synapses = [
            s for s in out.synapses if s.source != v.id and s.target != v.id
        ]

    elif isinstance(v, AddSynapseType):
        if (v.source, v.target) in _synapse_index(out):
            raise ValueError(f"synapse '{v.source}->{v.target}' already exists")
        out.synapses.append(
            SynapseSpec(
                source=v.source, target=v.target,
                synapse_type=v.synapse_type, plasticity=v.plasticity, params=v.params,
            )
        )

    elif isinstance(v, RemoveSynapseType):
        before = len(out.synapses)
        out.synapses = [
            s for s in out.synapses if (s.source, s.target) != (v.source, v.target)
        ]
        if len(out.synapses) == before:
            raise ValueError(f"synapse '{v.source}->{v.target}' does not exist")

    elif isinstance(v, SetPlasticityRule):
        syn = _synapse_index(out).get((v.source, v.target))
        if syn is None:
            raise ValueError(f"unknown synapse '{v.source}->{v.target}'")
        syn.plasticity = v.plasticity
        syn.params.update(v.params)

    elif isinstance(v, SetConnectivityRule):
        syn = _synapse_index(out).get((v.source, v.target))
        if syn is None:
            raise ValueError(f"unknown synapse '{v.source}->{v.target}'")
        syn.params.update(v.params)

    elif isinstance(v, SetNeuromodulatorSchedule):
        mod = next((m for m in out.rules.neuromodulators if m.id == v.id), None)
        if mod is None:
            out.rules.neuromodulators.append(
                NeuromodulatorSpec(
                    id=v.id, baseline=v.baseline, time_constant_ms=v.time_constant_ms
                )
            )
        else:
            mod.baseline = v.baseline
            mod.time_constant_ms = v.time_constant_ms

    elif isinstance(v, SetLearningRateSchedule):
        out.rules.learning_rate = v.learning_rate

    elif isinstance(v, Prune):
        out.rules.pruning_rate = v.pruning_rate

    elif isinstance(v, AddGate):
        out.rules.gating[v.name] = v.strength

    elif isinstance(v, SetTemporalDynamics):
        if not hasattr(out.rules.temporal_dynamics, v.field):
            raise ValueError(f"unknown temporal_dynamics field '{v.field}'")
        setattr(out.rules.temporal_dynamics, v.field, v.value)

    else:  # pragma: no cover - exhaustive by construction
        raise ValueError(f"unhandled intervention verb: {type(v).__name__}")

    # Re-validate and bump version. Re-validation catches cases like an
    # out-of-order retention threshold or a dangling synapse reference.
    out = out.next_version()
    return Genome.model_validate(out.model_dump())


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
