"""
Stage 1b: group a patient's events into "episodes" (one per encounter).

Each Episode collects everything that happened at, or as a result of, one
visit: what was newly diagnosed or resolved, which medicines were started or
stopped, which tests were done and their results, and a snapshot of the
conditions and medicines that were already active beforehand. This is the
fact sheet the LLM will later write a GP note or patient letter from.

Synthea often records screening questionnaires and follow-on actions an hour
or two after the encounter's official end time, so an event is linked to the
most recent encounter that started before it, provided it falls within
LINK_GRACE of that encounter's end. Anything left over is gathered into
"between-visit" episodes (for example a prescription change with no visit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .ccda_parser import Event, Patient

LINK_GRACE = timedelta(hours=12)


@dataclass
class Episode:
    index: int
    start: datetime
    stop: Optional[datetime]
    encounter: str                      # e.g. "General examination of patient (procedure)"
    setting: str                        # e.g. "primary care", "emergency", "hospital"
    age: int
    new_conditions: list[Event] = field(default_factory=list)
    resolved_conditions: list[Event] = field(default_factory=list)
    started_medications: list[Event] = field(default_factory=list)
    stopped_medications: list[Event] = field(default_factory=list)
    procedures: list[Event] = field(default_factory=list)
    vitals: list[Event] = field(default_factory=list)
    labs: list[Event] = field(default_factory=list)
    surveys: list[Event] = field(default_factory=list)
    immunizations: list[Event] = field(default_factory=list)
    careplans_started: list[Event] = field(default_factory=list)
    careplans_stopped: list[Event] = field(default_factory=list)
    allergies: list[Event] = field(default_factory=list)
    other: list[Event] = field(default_factory=list)
    # Snapshot of what was already active when the episode began.
    active_conditions: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)

    # Which list each (category, is_stop) pair goes into.
    _SLOTS = {
        ("condition", False): "new_conditions",
        ("condition", True): "resolved_conditions",
        ("medication", False): "started_medications",
        ("medication", True): "stopped_medications",
        ("careplan", False): "careplans_started",
        ("careplan", True): "careplans_stopped",
        ("procedure", False): "procedures",
        ("vital", False): "vitals",
        ("lab", False): "labs",
        ("survey", False): "surveys",
        ("immunization", False): "immunizations",
        ("allergy", False): "allergies",
    }

    def add(self, event: Event, is_stop: bool = False) -> None:
        getattr(self, self._SLOTS.get((event.category, is_stop), "other")).append(event)

    @property
    def is_between_visits(self) -> bool:
        return self.setting == "between visits"

    def to_dict(self) -> dict:
        out = {
            "index": self.index,
            "start": self.start.isoformat(),
            "stop": self.stop.isoformat() if self.stop else None,
            "encounter": self.encounter,
            "setting": self.setting,
            "age": self.age,
            "active_conditions": self.active_conditions,
            "current_medications": self.current_medications,
        }
        for slot in sorted(set(self._SLOTS.values())) + ["other"]:
            out[slot] = [e.to_dict() for e in getattr(self, slot)]
        return out


# ---------------------------------------------------------------- building episodes

def guess_setting(encounter_description: str) -> str:
    """Rough care setting from the encounter name (Synthea's tables omit the encounter class)."""
    d = encounter_description.lower()
    rules = [
        ("emergency", "emergency"),
        ("hospital admission", "hospital"), ("inpatient", "hospital"), ("admission", "hospital"),
        ("urgent care", "urgent care"),
        ("dental", "dental"),
        ("prenatal", "maternity"), ("obstetric", "maternity"), ("postnatal", "maternity"),
        ("home visit", "home visit"), ("home health", "home visit"),
        ("telemedicine", "telephone/video"), ("telephone", "telephone/video"),
        ("hospice", "hospice"), ("skilled nursing", "nursing facility"),
        ("rehabilitation", "rehabilitation"),
    ]
    for keyword, setting in rules:
        if keyword in d:
            return setting
    return "primary care"


def build_episodes(patient: Patient) -> list[Episode]:
    encounters = sorted((e for e in patient.events if e.category == "encounter"), key=lambda e: e.start)
    episodes = [
        Episode(index=0, start=e.start, stop=e.stop, encounter=e.description,
                setting=guess_setting(e.description), age=patient.age_on(e.start))
        for e in encounters
    ]

    def find_episode(t: datetime) -> Optional[Episode]:
        best = None
        for ep in episodes:
            if ep.start <= t and t <= (ep.stop or ep.start) + LINK_GRACE:
                best = ep          # keep the latest matching encounter
        return best

    # Every non-encounter event is linked at its start; conditions, medications and
    # care plans are linked a second time at their stop date (resolved / stopped).
    pending: list[tuple[datetime, Event, bool]] = []
    for e in patient.events:
        if e.category == "encounter":
            continue
        pending.append((e.start, e, False))
        if e.stop and e.category in ("condition", "medication", "careplan"):
            pending.append((e.stop, e, True))

    loose: dict[str, Episode] = {}
    for t, event, is_stop in pending:
        ep = find_episode(t)
        if ep is None:
            # No visit: group with other loose events from the same day.
            key = t.date().isoformat()
            if key not in loose:
                loose[key] = Episode(index=0, start=t, stop=None, encounter="No encounter recorded",
                                     setting="between visits", age=patient.age_on(t))
            ep = loose[key]
        ep.add(event, is_stop)

    episodes = sorted(episodes + list(loose.values()), key=lambda ep: ep.start)
    for i, ep in enumerate(episodes, start=1):
        ep.index = i
        _fill_snapshot(ep, patient)
    return episodes


def _fill_snapshot(ep: Episode, patient: Patient) -> None:
    """Conditions and medications that were active just before this episode started."""
    def active(e: Event) -> bool:
        return e.start < ep.start and (e.stop is None or e.stop > ep.start)

    ep.active_conditions = _unique(e.description for e in patient.events
                                   if e.category == "condition" and active(e))
    ep.current_medications = _unique(e.description for e in patient.events
                                     if e.category == "medication" and active(e))


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------- readable text

def short_name(event: Event) -> str:
    """Make coded names easier to read, e.g. 'Glucose [Mass/volume] in Blood' -> 'Glucose'."""
    name = event.description
    if event.category == "survey":
        m = re.search(r"\(([A-Z][A-Z0-9-]+)\)", name) or re.search(r"\[([A-Z][A-Z0-9-]+)\]", name)
        return f"{m.group(1)} total score" if m else name
    if event.category in ("lab", "vital"):
        name = re.sub(r"\s+by .*$", "", name)                                  # method
        name = re.sub(r"\s+(in|of) (Blood|Serum or Plasma|Red Blood Cells|Urine|Plasma|Serum)$", "", name)
        name = re.sub(r"\s*\[([^\]]*)\]", _bracket, name)                      # LOINC property
    return name.strip()


# LOINC "[property]" brackets: most just say what kind of quantity it is and can go,
# a few are the only thing that distinguishes one test from another and are kept.
_KEEP_BRACKETS = {"distwidth": " distribution width", "entitic mean volume": " mean volume"}


def _bracket(match: re.Match) -> str:
    return _KEEP_BRACKETS.get(match.group(1).lower(), "")


def _round_value(value: str) -> str:
    """'65.63 mg/dL' -> '65.6 mg/dL'; leaves text values alone."""
    m = re.match(r"^(-?\d+\.\d+)(.*)$", value)
    return f"{float(m.group(1)):.1f}{m.group(2)}" if m else value


def episode_to_text(ep: Episode) -> str:
    """A compact plain-text fact sheet for one episode (used in LLM prompts)."""
    lines = [f"EPISODE {ep.index}: {ep.start:%d %B %Y} (age {ep.age})",
             f"Encounter: {ep.encounter}  [setting: {ep.setting}]"]

    def section(title: str, items: list[str]) -> None:
        if items:
            lines.append(f"{title}:")
            lines.extend(f"  - {item}" for item in items)

    # Synthea records a renewal as "stopped" and "started" at the same visit; show it as a renewal.
    renewed = _unique(e.description for e in ep.started_medications)
    renewed = [d for d in renewed if d in {e.description for e in ep.stopped_medications}]
    rerecorded = [d for d in _unique(e.description for e in ep.new_conditions)
                  if d in {e.description for e in ep.resolved_conditions}]

    section("Already-active conditions", ep.active_conditions)
    section("Medications already being taken", ep.current_medications)
    section("NEW conditions / findings recorded",
            _unique(e.description for e in ep.new_conditions if e.description not in rerecorded))
    section("Conditions / findings RESOLVED",
            _unique(e.description for e in ep.resolved_conditions if e.description not in rerecorded))
    section("Conditions / findings reviewed and still current", rerecorded)
    section("Allergies recorded", [e.description for e in ep.allergies])
    section("Medications STARTED",
            _unique(e.description for e in ep.started_medications if e.description not in renewed))
    section("Medications STOPPED",
            _unique(e.description for e in ep.stopped_medications if e.description not in renewed))
    section("Medications RENEWED (continuing)", renewed)
    section("Procedures / assessments", _unique(e.description for e in ep.procedures))
    section("Vital signs", [f"{short_name(e)}: {_round_value(e.value)}" for e in ep.vitals])
    section("Questionnaire scores", [f"{short_name(e)}: {e.value}" for e in ep.surveys])

    panels: dict[str, list[str]] = {}
    for e in ep.labs:
        panels.setdefault(e.panel or "Other tests", []).append(f"{short_name(e)} {_round_value(e.value)}")
    section("Lab results", [f"{panel}: " + "; ".join(results) for panel, results in panels.items()])

    section("Immunizations given", [e.description for e in ep.immunizations])
    section("Care plans started", [e.description for e in ep.careplans_started])
    section("Care plans ended", [e.description for e in ep.careplans_stopped])
    section("Other", [e.description for e in ep.other])
    return "\n".join(lines)


def patient_header_text(patient: Patient) -> str:
    return "\n".join([
        f"Patient: {patient.full_name} ({patient.gender}, born {patient.birth_date:%d %B %Y})",
        f"Race/ethnicity: {patient.race}, {patient.ethnicity}",
        f"Lives in: {patient.city}, {patient.state}",
        f"Primary care practice: {patient.practice} ({patient.practice_city})",
    ])
