"""
Stage 1a: read a Synthea C-CDA (XML) file into simple Python objects.

Synthea writes every section of the C-CDA twice: once as a human-readable
HTML-style table (the <text> block) and once as coded "entries". The tables
are simple and consistent (Start | Stop | Description | Code | [Value]), so we
read those. The one exception is the Diagnostic Results section, whose table
only names each lab panel; the individual results and their values are read
from the structured entries instead.

Only the Python standard library is used, so this runs anywhere (including
Google Colab) without extra installs.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

NS = {"hl7": "urn:hl7-org:v3", "xsi": "http://www.w3.org/2001/XMLSchema-instance"}

# LOINC section codes used by Synthea -> our event category.
SECTION_CATEGORIES = {
    "48765-2": "allergy",
    "10160-0": "medication",
    "30954-2": "lab",          # Diagnostic Results (handled specially)
    "11450-4": "condition",
    "47519-4": "procedure",
    "46240-8": "encounter",
    "8716-3": "vital",
    "11369-6": "immunization",
    "18776-5": "careplan",
    "29762-2": "social",       # Social History (smoking status only)
    "47420-5": "survey",       # Functional Status = questionnaire scores
}


@dataclass
class Event:
    """One dated item from the record (a diagnosis, a prescription, a lab result...)."""
    category: str
    start: datetime
    description: str
    stop: Optional[datetime] = None
    code: str = ""
    value: str = ""
    panel: str = ""            # for lab results: the panel the result belongs to

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["stop"] = self.stop.isoformat() if self.stop else None
        return d


@dataclass
class Patient:
    id: str
    given_name: str
    family_name: str
    gender: str
    birth_date: datetime
    race: str = ""
    ethnicity: str = ""
    language: str = ""
    city: str = ""
    state: str = ""
    practice_name: str = ""
    practice_city: str = ""
    record_date: Optional[datetime] = None
    events: list[Event] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.given_name} {self.family_name}"

    def age_on(self, when: datetime) -> int:
        b = self.birth_date
        return when.year - b.year - ((when.month, when.day) < (b.month, b.day))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["birth_date"] = self.birth_date.date().isoformat()
        d["record_date"] = self.record_date.isoformat() if self.record_date else None
        d["events"] = [e.to_dict() for e in self.events]
        return d


# ---------------------------------------------------------------- helpers

def _parse_time(text: str | None) -> Optional[datetime]:
    """Accept both table format (2016-12-05T22:20:00Z) and HL7 format (20161205222000)."""
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    digits = re.sub(r"\D", "", text)[:14].ljust(14, "0")
    return datetime.strptime(digits, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _strip_synthea_digits(name: str) -> str:
    """Synthea appends numbers to names to make them unique (Alexandra16 -> Alexandra)."""
    return re.sub(r"\d+", "", name).strip()


def _clean_value(value: str) -> str:
    """Tidy table values such as '1 {score}' -> '1'."""
    return re.sub(r"\s*\{[^}]*\}", "", value).strip()


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _find(el: ET.Element, path: str) -> ET.Element | None:
    return el.find(path, NS)


# ---------------------------------------------------------------- sections

def _read_table(section: ET.Element, category: str) -> list[Event]:
    """Read the narrative <table> of a section into Events."""
    events = []
    table = _find(section, "hl7:text/hl7:table")
    if table is None:
        return events
    headers = [_text(th).lower() for th in table.findall(".//hl7:thead//hl7:th", NS)]
    for tr in table.findall(".//hl7:tbody/hl7:tr", NS):
        cells = dict(zip(headers, [_text(td) for td in tr.findall("hl7:td", NS)]))
        start = _parse_time(cells.get("start"))
        if start is None or not cells.get("description"):
            continue
        events.append(Event(
            category=category,
            start=start,
            stop=_parse_time(cells.get("stop")),
            description=cells["description"],
            code=cells.get("code", ""),
            value=_clean_value(cells.get("value", "")),
        ))
    return events


def _format_value(value_el: ET.Element | None) -> str:
    """Turn an HL7 <value> element into readable text (e.g. '65.63 mg/dL')."""
    if value_el is None:
        return ""
    xsi_type = value_el.get(f"{{{NS['xsi']}}}type", "")
    if xsi_type == "PQ":
        unit = _clean_value(value_el.get("unit", ""))
        return f"{value_el.get('value', '')} {unit}".strip()
    if xsi_type == "CD":
        return value_el.get("displayName", "")
    return _text(value_el) or value_el.get("value", "")


# Synthea also files questionnaires (PHQ-2, AUDIT-C, PRAPARE...) under Diagnostic Results.
_QUESTIONNAIRE = re.compile(r"questionnaire|survey|screening|assessment|score|\d+ item|\[[A-Z][A-Z0-9-]+\]", re.I)


def _read_results(section: ET.Element) -> list[Event]:
    """Read lab panels (organizers) and their individual results from structured entries."""
    events = []
    for organizer in section.findall(".//hl7:entry/hl7:organizer", NS):
        panel_code = _find(organizer, "hl7:code")
        panel = panel_code.get("displayName", "") if panel_code is not None else ""
        category = "survey" if _QUESTIONNAIRE.search(panel) else "lab"
        low = _find(organizer, "hl7:effectiveTime/hl7:low")
        panel_time = _parse_time(low.get("value") if low is not None else None)
        for obs in organizer.findall("hl7:component/hl7:observation", NS):
            code_el = _find(obs, "hl7:code")
            time_el = _find(obs, "hl7:effectiveTime")
            start = _parse_time(time_el.get("value") if time_el is not None else None) or panel_time
            if start is None or code_el is None:
                continue
            events.append(Event(
                category=category,
                start=start,
                description=code_el.get("displayName", ""),
                code=f"http://loinc.org {code_el.get('code', '')}",
                value=_format_value(_find(obs, "hl7:value")),
                panel=panel,
            ))
    return events


def _read_social(section: ET.Element) -> list[Event]:
    """Synthea's Social History section only holds smoking status."""
    events = []
    for obs in section.findall(".//hl7:entry/hl7:observation", NS):
        code_el, value_el = _find(obs, "hl7:code"), _find(obs, "hl7:value")
        time_el = _find(obs, "hl7:effectiveTime")
        start = _parse_time(time_el.get("value") if time_el is not None else None)
        if start and code_el is not None:
            events.append(Event(
                category="social",
                start=start,
                description=code_el.get("displayName", ""),
                value=_format_value(value_el),
            ))
    return events


# ---------------------------------------------------------------- main entry point

def parse_ccda(path: str | Path) -> Patient:
    """Parse one Synthea C-CDA XML file into a Patient with a list of Events."""
    root = ET.parse(path).getroot()

    pr = _find(root, "hl7:recordTarget/hl7:patientRole")
    pt = _find(pr, "hl7:patient")
    org = _find(root, "hl7:author/hl7:assignedAuthor/hl7:representedOrganization")

    def attr(el, path, name="displayName"):
        found = _find(el, path) if el is not None else None
        return found.get(name, "") if found is not None else ""

    patient = Patient(
        id=attr(pr, "hl7:id", "extension"),
        given_name=_strip_synthea_digits(_text(_find(pt, "hl7:name/hl7:given"))),
        family_name=_strip_synthea_digits(_text(_find(pt, "hl7:name/hl7:family"))),
        gender={"F": "female", "M": "male"}.get(attr(pt, "hl7:administrativeGenderCode", "code"), "unknown"),
        birth_date=_parse_time(attr(pt, "hl7:birthTime", "value")),
        race=attr(pt, "hl7:raceCode"),
        ethnicity=attr(pt, "hl7:ethnicGroupCode"),
        language=attr(pt, "hl7:languageCommunication/hl7:languageCode", "code"),
        city=_text(_find(pr, "hl7:addr/hl7:city")),
        state=_text(_find(pr, "hl7:addr/hl7:state")),
        practice_name=_text(_find(org, "hl7:name")) if org is not None else "",
        practice_city=_text(_find(org, "hl7:addr/hl7:city")).title() if org is not None else "",
        record_date=_parse_time(attr(root, "hl7:effectiveTime", "value")),
    )

    for section in root.iter(f"{{{NS['hl7']}}}section"):
        code = attr(section, "hl7:code", "code")
        category = SECTION_CATEGORIES.get(code)
        if category is None:
            continue
        if category == "lab":
            patient.events.extend(_read_results(section))
        elif category == "social":
            patient.events.extend(_read_social(section))
        else:
            patient.events.extend(_read_table(section, category))

    # Questionnaire scores appear in both Functional Status and Results: keep one copy.
    unique = {(e.category, e.start, e.description, e.value): e for e in patient.events}
    patient.events = sorted(unique.values(), key=lambda e: e.start)
    return patient
