"""
Stage 2b-4: generate the documents for one patient and assemble the output file.

    profile  -> one LLM call invents a consistent background ("story bible")
    documents -> one LLM call per planned document, each given the profile,
                 summaries of earlier documents, and that episode's fact sheet
    assemble -> one plain-text file, documents in date order, each with a header

Everything is saved as it is produced, in output/<Name>/:
    profile.json        the invented background
    documents.jsonl     one line per finished document (the checkpoint)
    prompts/            the exact prompt sent for each document, for inspection
    ground_truth.json   the parsed record, and which episodes were written up
Re-running skips documents already in documents.jsonl, so an interrupted Colab
session can simply be resumed.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import prompts
from .ccda_parser import Patient, parse_ccda
from .episodes import Episode, build_episodes, episode_to_text, patient_header_text, short_name
from .planner import DocumentPlan, plan_documents

STORY_SO_FAR_LENGTH = 12        # how many earlier document summaries each prompt sees
CHILD_AGE = 16                  # below this, patient messages are written by a parent
SUMMARY_MAX_ITEMS = 60          # cap on conditions / medications listed in the record summary
SUMMARY_MAX_SCORES = 6          # most recent questionnaire scores listed per questionnaire


# ---------------------------------------------------------------- record summary for the profile

def _year_ranges(events) -> str:
    """Merge one item's periods into year ranges, e.g. '2010-2016, 2019-ongoing'.

    Synthea records every renewal of a long-term prescription as a separate entry
    (hundreds, for some patients), so merging keeps the summary short."""
    merged: list[list] = []
    for start, stop in sorted(((e.start.year, e.stop.year if e.stop else None) for e in events),
                              key=lambda span: (span[0], span[1] is None, span[1] or 0)):
        if merged and (merged[-1][1] is None or start <= merged[-1][1] + 1):
            if merged[-1][1] is not None:
                merged[-1][1] = None if stop is None else max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    text = ", ".join(f"{a}-{b if b is not None else 'ongoing'}" for a, b in merged)
    return text + (f" ({len(events)} entries)" if len(events) > 1 else "")


def _grouped_lines(events) -> list[str]:
    groups = defaultdict(list)
    for e in events:
        groups[e.description].append(e)
    lines = [f"  - {name}: {_year_ranges(evs)}" for name, evs in groups.items()]
    if len(lines) > SUMMARY_MAX_ITEMS:
        lines = lines[:SUMMARY_MAX_ITEMS] + [f"  - ...and {len(lines) - SUMMARY_MAX_ITEMS} more"]
    return lines


def record_summary(patient: Patient, episodes: list[Episode]) -> str:
    """A compact whole-life summary of the record, used to invent the profile."""
    by_cat = defaultdict(list)
    for e in patient.events:
        by_cat[e.category].append(e)

    lines = [patient_header_text(patient), ""]
    lines.append("Conditions and social findings (years active):")
    lines += _grouped_lines(by_cat["condition"])
    lines.append("Medications (years taken):")
    lines += _grouped_lines(by_cat["medication"])
    if by_cat["allergy"]:
        lines.append("Allergies: " + "; ".join(dict.fromkeys(e.description for e in by_cat["allergy"])))
    if by_cat["careplan"]:
        lines.append("Care plans:")
        lines += _grouped_lines(by_cat["careplan"])

    scores = defaultdict(list)
    for e in by_cat["survey"]:
        scores[short_name(e)].append(f"{e.value} ({e.start.year})")
    if scores:
        lines.append("Questionnaire scores over time (most recent):")
        lines += [f"  - {name}: {', '.join(vals[-SUMMARY_MAX_SCORES:])}" for name, vals in scores.items()]
    for e in by_cat["social"]:
        lines.append(f"{e.description}: {e.value} ({e.start.year})")

    settings = defaultdict(int)
    for ep in episodes:
        if not ep.is_between_visits:
            settings[ep.setting] += 1
    lines.append(f"Encounters {episodes[0].start.year}-{episodes[-1].start.year}: "
                 + ", ".join(f"{n} {s}" for s, n in settings.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------- profile

def make_profile(patient: Patient, episodes: list[Episode], client, prompt_dir: Path) -> dict:
    prompt = prompts.PROFILE_PROMPT.format(
        record_summary=record_summary(patient, episodes),
        first_year=episodes[0].start.year, last_year=episodes[-1].start.year,
        practice=patient.practice)
    (prompt_dir / "000-profile.txt").write_text(prompt)
    for attempt in range(1, 4):
        reply = client.chat(prompts.SYSTEM_PROMPT, prompt, json_mode=True)
        (prompt_dir / f"000-profile.reply{attempt}.txt").write_text(reply)
        profile = _parse_profile(reply)
        if profile:
            if not isinstance(profile.get("gp_history"), list) or not profile["gp_history"]:
                profile["gp_history"] = [{"name": "Dr Morgan", "from_year": episodes[0].start.year,
                                          "to_year": episodes[-1].start.year}]
            return profile
    raise RuntimeError("The model did not return a usable profile after 3 attempts; "
                       f"its replies are in {prompt_dir}/000-profile.reply*.txt")


def _parse_profile(reply: str) -> dict | None:
    """Read the profile JSON, tolerating extra text around it or an outer wrapper object."""
    for candidate in (reply, reply[reply.find("{"):reply.rfind("}") + 1]):
        try:
            profile = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(profile, dict) and len(profile) == 1 and isinstance(next(iter(profile.values())), dict):
            profile = next(iter(profile.values()))          # e.g. {"profile": {...}}
        if isinstance(profile, dict) and len(profile) >= 5:
            return profile
    return None


def profile_text(profile: dict) -> str:
    """Show the profile JSON as readable 'key: value' lines for the prompt."""
    lines = []
    for key, value in profile.items():
        label = key.replace("_", " ").capitalize()
        if isinstance(value, list):
            items = ["; ".join(f"{v}" for v in item.values()) if isinstance(item, dict) else str(item)
                     for item in value]
            lines.append(f"{label}:\n" + "\n".join(f"  - {item}" for item in items))
        elif value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def clinician_for(profile: dict, year: int) -> str:
    history = profile.get("gp_history") or [{"name": "Dr Smith"}]
    for gp in history:
        try:
            if int(gp.get("from_year", 0)) <= year <= int(gp.get("to_year", 9999)):
                return gp["name"]
        except (TypeError, ValueError):
            continue
    return history[-1].get("name", "Dr Smith")


def writer_for(profile: dict, patient: Patient, age: int) -> str:
    name = profile.get("preferred_name") or patient.given_name
    guardian = profile.get("parent_or_guardian")
    if age < CHILD_AGE and guardian:
        return f"{guardian}, writing on behalf of {name} (aged {age})"
    return f"the patient, {name} {patient.family_name}"


# ---------------------------------------------------------------- one document

def _length(plan: DocumentPlan) -> str:
    ep = plan.episode
    amount = sum(len(getattr(ep, slot)) for slot in set(Episode._SLOTS.values()))
    return prompts.LENGTHS[plan.kind][0 if amount < 5 else 1 if amount < 20 else 2]


def build_prompt(plan: DocumentPlan, patient: Patient, profile: dict, summaries: list[dict]) -> str:
    ep = plan.episode
    age = patient.age_on(plan.date)
    task = prompts.TASKS[plan.kind].format(
        clinician=clinician_for(profile, plan.date.year),
        practice=patient.practice,
        writer=writer_for(profile, patient, age),
        date=f"{plan.date:%d %B %Y}", visit_date=f"{ep.start:%d %B %Y}",
        age=age, setting=ep.setting, city=patient.city, state=patient.state,
        length=_length(plan))

    recent = summaries[-STORY_SO_FAR_LENGTH:]
    story = "\n".join(f"- {s['date'][:10]} ({prompts.DOCUMENT_TITLES[s['kind']]}): {s['summary']}" for s in recent)
    if len(summaries) > len(recent):
        story = f"({len(summaries) - len(recent)} earlier documents not shown.)\n" + story

    return prompts.DOCUMENT_PROMPT.format(
        patient_header=patient_header_text(patient),
        profile=profile_text(profile),
        story_so_far=story or "(This is the first document in the record.)",
        fact_sheet=episode_to_text(ep),
        task=task)


def split_summary(text: str) -> tuple[str, str]:
    """Separate the document from the '### SUMMARY' the model appends, and tidy markdown."""
    # Accept "### SUMMARY", "**Summary:**", "SUMMARY: text..."; use the last one, since a
    # hospital letter may legitimately contain its own "Summary" heading.
    markers = list(re.finditer(r"^[\s#*]*SUMMARY\b[\s:*#]*", text, flags=re.M | re.I))
    if markers:
        body, summary = text[:markers[-1].start()].strip(), text[markers[-1].end():].strip()
    else:
        body, summary = text.strip(), ""
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)            # **bold** -> bold
    body = re.sub(r"^#+\s*", "", body, flags=re.M)          # stray markdown headings
    if not summary:
        summary = " ".join(re.split(r"(?<=[.!?])\s+", body)[:2])[:400]
    return body, " ".join(summary.split())


# ---------------------------------------------------------------- whole patient

def generate_patient(ccda_path: str | Path, client, out_dir: str | Path = "output",
                     max_episodes: int | None = None, max_documents: int | None = None, log=print) -> Path:
    patient = parse_ccda(ccda_path)
    episodes = build_episodes(patient)
    if max_episodes:
        episodes = episodes[:max_episodes]

    # Synthea names repeat once their digits are stripped, so add the start of the patient ID.
    stem = f"{patient.given_name}_{patient.family_name}_{patient.id[:8]}"
    work = Path(out_dir) / stem
    (work / "prompts").mkdir(parents=True, exist_ok=True)

    profile_path = work / "profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
    else:
        log(f"Inventing a background for {patient.full_name}...")
        profile = make_profile(patient, episodes, client, work / "prompts")
        profile_path.write_text(json.dumps(profile, indent=2))

    plans = plan_documents(episodes, seed=patient.id, max_documents=max_documents)
    # The ground truth covers every episode; "documents" says which ones were written up.
    (work / "ground_truth.json").write_text(json.dumps(
        {"patient": patient.to_dict(), "episodes": [ep.to_dict() for ep in episodes],
         "documents": [{"doc_id": p.doc_id, "episode": p.episode.index, "kind": p.kind,
                        "date": p.date.isoformat()} for p in plans]}, indent=2))
    done_path = work / "documents.jsonl"
    done = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            doc = json.loads(line)
            done[doc["doc_id"]] = doc
    log(f"{len(plans)} documents planned for {len(episodes)} episodes"
        f"{f' (capped at {max_documents})' if max_documents and len(plans) == max_documents else ''}; "
        f"{len(done)} already written.")

    summaries = [done[p.doc_id] for p in plans if p.doc_id in done]
    for n, plan in enumerate(plans, start=1):
        if plan.doc_id in done:
            continue
        # Only documents dated before this one count as "the story so far".
        earlier = [s for s in summaries if s["date"] < plan.date.isoformat()]
        prompt = build_prompt(plan, patient, profile, earlier)
        (work / "prompts" / f"{plan.doc_id}.txt").write_text(prompt)

        started = time.time()
        body, summary = split_summary(client.chat(prompts.SYSTEM_PROMPT, prompt))
        doc = {"doc_id": plan.doc_id, "episode": plan.episode.index, "kind": plan.kind,
               "date": plan.date.isoformat(), "author": author_for(plan, patient, profile),
               "text": body, "summary": summary, "model": getattr(client, "model", "")}
        with done_path.open("a") as f:
            f.write(json.dumps(doc) + "\n")
        done[plan.doc_id] = doc
        summaries.append(doc)
        log(f"  [{n}/{len(plans)}] {plan.date:%Y-%m-%d} {prompts.DOCUMENT_TITLES[plan.kind]} "
            f"({len(body.split())} words, {time.time() - started:.0f}s)")

    out_file = Path(out_dir) / f"{stem}.txt"
    out_file.write_text(assemble(patient, profile, [done[p.doc_id] for p in plans]))
    log(f"Wrote {out_file}")
    return out_file


def generate_many(ccda_paths: list, client, out_dir: str | Path = "output", workers: int = 3,
                  max_episodes: int | None = None, max_documents: int | None = None) -> list[Path]:
    """Generate several patients at once.

    Documents for one patient must be written in order (each sees the earlier ones), but
    different patients are independent. A GPU can write several responses together for
    little more than the cost of one, so running `workers` patients side by side raises
    throughput. The Ollama server must allow as many parallel requests (OLLAMA_NUM_PARALLEL).
    """
    paths = [Path(p) for p in ccda_paths]
    results, failures = [], []
    started = time.time()

    def run(i: int, path: Path) -> Path:
        tag = f"[{i}/{len(paths)} {path.stem.split('_')[0]}]"
        return generate_patient(path, client, out_dir=out_dir, max_episodes=max_episodes,
                                max_documents=max_documents, log=lambda msg: print(f"{tag} {msg}", flush=True))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, i, p): p for i, p in enumerate(paths, start=1)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as err:          # keep going; a re-run will retry this patient
                failures.append(futures[future])
                print(f"FAILED {futures[future].name}: {err}", flush=True)

    print(f"\nFinished {len(results)} of {len(paths)} patients in {(time.time() - started) / 60:.0f} minutes.")
    if failures:
        print("Failed (re-run to retry):", ", ".join(p.name for p in failures))
    return sorted(results)


def author_for(plan: DocumentPlan, patient: Patient, profile: dict) -> str:
    if plan.kind == "gp_note":
        return f"{clinician_for(profile, plan.date.year)}, {patient.practice}"
    if plan.kind == "clinic_letter":
        return f"{plan.episode.setting.capitalize()} clinician (see letter)"
    return writer_for(profile, patient, patient.age_on(plan.date)).replace("the patient, ", "")


RULE = "=" * 80
THIN_RULE = "-" * 80


def assemble(patient: Patient, profile: dict, docs: list[dict]) -> str:
    docs = sorted(docs, key=lambda d: d["date"])
    out = [RULE,
           "SYNTHETIC PATIENT RECORD - FOR TEACHING USE ONLY",
           "This patient and every document below are fictional. They were written by an AI",
           "model from a simulated record generated by Synthea (https://synthea.mitre.org).",
           RULE,
           f"Patient:  {patient.full_name}",
           f"Born:     {patient.birth_date:%d %B %Y}",
           f"Sex:      {patient.gender}",
           f"Address:  {patient.city}, {patient.state}",
           f"Practice: {patient.practice}",
           f"Documents: {len(docs)}",
           ""]
    for i, doc in enumerate(docs, start=1):
        out += [THIN_RULE,
                f"Document {i} of {len(docs)}",
                f"Date:   {doc['date'][:10]}",
                f"Type:   {prompts.DOCUMENT_TITLES[doc['kind']]}",
                f"Author: {doc['author']}",
                THIN_RULE,
                doc["text"],
                ""]
    return "\n".join(out)
