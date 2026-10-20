"""
Parse one Synthea C-CDA file and show what the generator will work from.

Usage:
    python inspect_record.py samples/Alexandra16_Mosciski958.xml

Writes two files to output/:
    <name>.timeline.txt       readable episode-by-episode fact sheets
    <name>.ground_truth.json  the full parsed record (patient + episodes)
"""

import json
import sys
from pathlib import Path

from gpnotes.ccda_parser import parse_ccda
from gpnotes.episodes import build_episodes, episode_to_text, patient_header_text


def main(path: str, out_dir: str = "output") -> None:
    patient = parse_ccda(path)
    episodes = build_episodes(patient)

    stem = f"{patient.given_name}_{patient.family_name}"
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    text = patient_header_text(patient) + "\n\n" + "\n\n".join(episode_to_text(ep) for ep in episodes)
    (out / f"{stem}.timeline.txt").write_text(text + "\n")
    (out / f"{stem}.ground_truth.json").write_text(json.dumps(
        {"patient": patient.to_dict(), "episodes": [ep.to_dict() for ep in episodes]}, indent=2))

    print(patient_header_text(patient))
    print(f"\n{len(patient.events)} events grouped into {len(episodes)} episodes "
          f"({sum(ep.is_between_visits for ep in episodes)} between-visit).")
    print(f"Wrote {out / (stem + '.timeline.txt')} and {out / (stem + '.ground_truth.json')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
