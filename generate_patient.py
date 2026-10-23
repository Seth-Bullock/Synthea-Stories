"""
Generate the GP notes and patient letters for one or more Synthea C-CDA files.

Usage:
    python generate_patient.py samples/Alexandra16_Mosciski958.xml
    python generate_patient.py path/to/ccda_folder/ --model gemma3:12b --workers 3
    python generate_patient.py samples/Alexandra16_Mosciski958.xml --dry-run --max-episodes 5

Needs an Ollama server running (see the Colab notebook), unless --dry-run is
given, in which case no model is called and placeholder text is written; this
is useful for checking the pipeline and reading the prompts in output/<Name>/prompts/.
"""

import argparse
from pathlib import Path

from gpnotes.generate import generate_many
from gpnotes.llm import DryRunClient, OllamaClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ccda", help="a C-CDA .xml file, or a folder of them")
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model name (default: gemma3:12b)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server address")
    parser.add_argument("--num-ctx", type=int, default=8192, help="model context window in tokens")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--out", default="output", help="output folder (default: output)")
    parser.add_argument("--max-episodes", type=int, help="only use the first N episodes (for quick tests)")
    parser.add_argument("--max-documents", type=int,
                        help="cap on documents per patient; the most eventful episodes are kept")
    parser.add_argument("--workers", type=int, default=1,
                        help="patients to generate at once; start Ollama with OLLAMA_NUM_PARALLEL set to match")
    parser.add_argument("--dry-run", action="store_true", help="don't call a model; write placeholders")
    args = parser.parse_args()

    client = DryRunClient() if args.dry_run else OllamaClient(
        model=args.model, host=args.host, num_ctx=args.num_ctx, temperature=args.temperature)

    source = Path(args.ccda)
    files = sorted(source.glob("*.xml")) if source.is_dir() else [source]
    generate_many(files, client, out_dir=args.out, workers=args.workers, max_episodes=args.max_episodes,
                  max_documents=args.max_documents)


if __name__ == "__main__":
    main()
