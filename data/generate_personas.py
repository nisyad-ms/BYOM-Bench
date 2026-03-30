#!/usr/bin/env python
"""Generate base personas for BYOM-Bench.

Reads domains and weights from domains.txt, then uses an LLM to generate
personas of people who would realistically use AI assistants in those domains.

Usage:
    uv run python data/generate_personas.py
    uv run python data/generate_personas.py --output data/base_personas.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so we can import byom_bench
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from byom_bench.client import LLMClient

DATA_DIR = Path(__file__).resolve().parent
DOMAINS_FILE = DATA_DIR / "domains.txt"
DEFAULT_OUTPUT = DATA_DIR / "base_personas.json"

SYSTEM_PROMPT = """\
You are an expert at creating realistic, diverse user personas for AI assistant benchmarks.

Each persona is a short sentence with exactly two parts: [who they are] + [one stable fact about their life].
The fact should be something enduring — where they work, their life situation, their background — NOT a current task or activity.

Rules:
- Each persona must be a single short sentence, starting with "A" or "An"
- Structure: [identity/role] + [one stable life fact]
- Good facts: employer, life circumstance, background, career stage, family situation
- Bad facts: current task, project they're working on, specific tool they're using
- Keep it under 15 words
- Make personas diverse across: age, gender, experience level, life circumstances
- Do NOT include names
- Do NOT repeat the same role across personas in a batch"""

EXISTING_EXAMPLES = """\
A 31-year old senior software developer working at a large tech company
A small business owner who recently expanded to a second location
A 22-year-old computer science undergraduate graduating in 6 months
A 35-year old bilingual call center agent supporting both English and Spanish speakers
A mid-career real estate agent with a client base of over 200
A retired couple with mobility challenges who travel frequently
A junior paralegal at a small immigration law firm
A 50-year-old restaurant owner with no prior tech experience
A graduate student in public health on a tight research deadline
A high school student who runs a small online jewelry business"""


def parse_domains(path: Path) -> list[tuple[str, int]]:
    """Parse domains.txt into (domain_name, count) pairs."""
    domains = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(",", 1)
        name = parts[0].strip()
        count = int(parts[1].strip())
        domains.append((name, count))
    return domains


def generate_for_domain(client: LLMClient, domain: str, count: int) -> list[str]:
    """Generate personas for a single domain."""
    prompt = f"""\
Generate exactly {count} personas of people who would use an AI assistant for tasks related to: {domain}

Each persona should be short: [who they are] + [one fact]. Under 15 words.

Examples (from other domains):
{EXISTING_EXAMPLES}

Output exactly {count} personas, one per line. No numbering, no bullets, no blank lines."""

    response = client.complete_chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4096,
    )

    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    # Remove any accidental numbering (e.g., "1. A ..." or "- A ...")
    cleaned = []
    for line in lines:
        # Strip leading "1. ", "- ", "• ", etc.
        for prefix in ("- ", "• ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix) :]
        if line and line[0].isdigit():
            # Strip "1. " or "1) " patterns
            parts = line.split(". ", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                line = parts[1]
            else:
                parts = line.split(") ", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    line = parts[1]
        if line:
            cleaned.append(line)

    return cleaned[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate base personas for BYOM-Bench")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output file path")
    parser.add_argument("--domain", type=str, default=None, help="Regenerate only this domain (must match a name in domains.txt)")
    args = parser.parse_args()

    if not DOMAINS_FILE.exists():
        print(f"domains.txt not found at {DOMAINS_FILE}")
        sys.exit(1)

    all_domains = parse_domains(DOMAINS_FILE)

    if args.domain:
        matching = [(name, count) for name, count in all_domains if name == args.domain]
        if not matching:
            domain_names = [name for name, _ in all_domains]
            print(f"Domain '{args.domain}' not found. Available: {domain_names}")
            sys.exit(1)
        domains = matching
    else:
        domains = all_domains

    total = sum(count for _, count in domains)
    print(f"Generating {total} personas across {len(domains)} domains:")
    for domain, count in domains:
        print(f"  {domain}: {count}")

    # Load existing file if regenerating a single domain
    if args.domain and args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            result: dict[str, list[str]] = json.load(f)
    else:
        result = {}

    client = LLMClient()

    for domain, count in domains:
        print(f"\nGenerating {count} personas for '{domain}'...")
        personas = generate_for_domain(client, domain, count)
        print(f"  Got {len(personas)} personas")
        if len(personas) != count:
            print(f"  WARNING: Expected {count}, got {len(personas)}")
        result[domain] = personas

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_generated = sum(len(v) for v in result.values())
    print(f"\nWrote {total_generated} personas across {len(result)} domains to {args.output}")


if __name__ == "__main__":
    main()
