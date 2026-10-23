"""
Stage 2a: decide which documents to write for each episode.

Every primary-care visit gets a GP consultation note. Visits elsewhere (A&E,
hospital, urgent care) get a letter to the GP from the clinician who saw the
patient. Patient-written messages are sprinkled in at random, more often
around visits where something new happened. The random choices are seeded
from the patient's ID, so re-running on the same file gives the same plan.

The probabilities below are the main knobs for how "chatty" the patient is.

Long records (an elderly patient can have hundreds of visits) can be capped
with max_documents: the documents about the most eventful episodes are kept
(new diagnoses, new or stopped medicines, hospital letters, test results) and
routine repeat visits are dropped first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .episodes import Episode

# Chance of a patient message BEFORE a visit where a new condition is recorded
P_MESSAGE_BEFORE = 0.35
# Chance of a patient message AFTER a visit where medicines or conditions changed
P_MESSAGE_AFTER = 0.30
# ...raised to this when the visit recorded a new social or emotional circumstance
P_MESSAGE_AFTER_SOCIAL = 0.55
# Chance of a patient message for a "between visits" episode (illness resolved, prescription ran out)
P_MESSAGE_BETWEEN = 0.6

# Settings where the patient is seen by someone other than their own GP.
LETTER_SETTINGS = {"emergency", "hospital", "urgent care", "nursing facility", "hospice", "rehabilitation"}
# Settings the GP record would not normally document at all.
SKIP_SETTINGS = {"dental"}

SOCIAL_WORDS = ("employment", "unemployed", "stress", "isolation", "social contact", "violence",
                "abuse", "housing", "homeless", "criminal", "transport", "refugee", "prison",
                "education", "income", "financial", "insurance")


@dataclass
class DocumentPlan:
    doc_id: str            # stable ID used for checkpointing, e.g. "ep010-gp_note"
    episode: Episode
    kind: str              # gp_note | clinic_letter | patient_before | patient_after | patient_between
    date: datetime


def is_social(description: str) -> bool:
    return any(word in description.lower() for word in SOCIAL_WORDS)


def plan_documents(episodes: list[Episode], seed: str, max_documents: int | None = None) -> list[DocumentPlan]:
    rng = random.Random(seed)
    plans: list[DocumentPlan] = []

    for i, ep in enumerate(episodes):
        previous_start = episodes[i - 1].start if i > 0 else ep.start - timedelta(days=30)
        next_start = episodes[i + 1].start if i + 1 < len(episodes) else ep.start + timedelta(days=60)

        def add(kind: str, date: datetime) -> None:
            plans.append(DocumentPlan(f"ep{ep.index:03d}-{kind}", ep, kind, date))

        if ep.setting in SKIP_SETTINGS:
            continue

        if ep.is_between_visits:
            if rng.random() < P_MESSAGE_BETWEEN:
                add("patient_between", ep.start)
            continue

        changed = bool(ep.new_conditions or ep.started_medications or ep.stopped_medications)
        social = any(is_social(e.description) for e in ep.new_conditions + ep.resolved_conditions)

        if ep.new_conditions and rng.random() < P_MESSAGE_BEFORE:
            days_before = min(rng.randint(1, 6), max(1, (ep.start - previous_start).days - 1))
            add("patient_before", ep.start - timedelta(days=days_before))

        add("clinic_letter" if ep.setting in LETTER_SETTINGS else "gp_note", ep.start)

        if changed and rng.random() < (P_MESSAGE_AFTER_SOCIAL if social else P_MESSAGE_AFTER):
            days_after = min(rng.randint(3, 14), max(1, (next_start - ep.start).days - 1))
            add("patient_after", ep.start + timedelta(days=days_after))

    if max_documents and len(plans) > max_documents:
        ranked = sorted(plans, key=lambda p: (-_priority(p), rng.random()))
        plans = ranked[:max_documents]

    plans.sort(key=lambda p: p.date)
    return plans


def _priority(plan: DocumentPlan) -> float:
    """How much an episode's document adds to the story; higher is kept first."""
    ep = plan.episode
    started = {e.description for e in ep.started_medications}
    stopped = {e.description for e in ep.stopped_medications}
    new = {e.description for e in ep.new_conditions} - {e.description for e in ep.resolved_conditions}
    new.discard("Medication review due (situation)")

    score = (3 * len(new)
             + 2 * len(started - stopped)
             + 1 * len(stopped - started)
             + 1 * len(ep.resolved_conditions)
             + 0.5 * bool(ep.labs) + 0.5 * bool(ep.surveys) + 0.5 * bool(ep.immunizations)
             + 0.5 * bool(ep.procedures))
    if plan.kind == "clinic_letter":
        score += 2
    if ep.index == 1:
        score += 5                  # keep the opening document
    if plan.kind.startswith("patient"):
        score = score * 0.5 - 0.1   # always ranked below the visit note it belongs with
    return score
