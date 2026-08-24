"""Generate syntactic caption variants for every dataset prompt.

Every prompt has the shape "A person <verb-3sg> <rest>." — the variants keep
the semantic content identical and vary only the surface form (subject noun,
tense/aspect, frame). This isolates the treatment: if one fixed string per
action is what starves text-motion alignment, purely syntactic variation
should move the retrieval metric; if deeper caption semantics are needed,
it will not.

Output: JSON {canonical: [variants...]} (canonical included as variant 0).
"""
import argparse
import json
import re
from pathlib import Path

# irregular / non -s third-person forms in the prompt inventory
ING = {
    "does": "doing", "sits": "sitting", "runs": "running", "hops": "hopping",
    "claps": "clapping", "waves": "waving", "shrugs": "shrugging",
    "balances": "balancing", "dances": "dancing", "bounces": "bouncing",
    "shuffles": "shuffling", "wiggles": "wiggling", "stretches": "stretching",
    "marches": "marching", "punches": "punching", "crouches": "crouching",
    "reaches": "reaching", "touches": "touching", "catches": "catching",
    "dribbles": "dribbling", "jumps": "jumping", "steps": "stepping",
    "skips": "skipping", "taps": "tapping", "spins": "spinning",
    "swims": "swimming", "jogs": "jogging", "bows": "bowing",
}
BASE_EXC = {"does": "do", "stretches": "stretch", "marches": "march",
            "punches": "punch", "crouches": "crouch", "reaches": "reach",
            "touches": "touch", "catches": "catch", "balances": "balance",
            "dances": "dance", "bounces": "bounce", "shuffles": "shuffle",
            "wiggles": "wiggle", "waves": "wave", "dribbles": "dribble"}


def base_form(verb: str) -> str:
    if verb in BASE_EXC:
        return BASE_EXC[verb]
    if verb.endswith("ies"):
        return verb[:-3] + "y"
    if verb.endswith("s"):
        return verb[:-1]
    return verb


def ing_form(verb: str) -> str:
    if verb in ING:
        return ING[verb]
    base = base_form(verb)
    if base.endswith("e") and not base.endswith("ee"):
        return base[:-1] + "ing"
    return base + "ing"


def variants(prompt: str) -> list[str]:
    match = re.fullmatch(r"A person (\w+)(?: (.*))?\.", prompt)
    if not match:
        return [prompt]
    verb, rest = match.group(1), match.group(2) or ""
    b, g = base_form(verb), ing_form(verb)
    rest_dot = f" {rest}." if rest else "."
    out = [
        prompt,
        f"Someone {verb}{rest_dot}",
        f"A stick figure {verb}{rest_dot}",
        f"The person {verb}{rest_dot}",
        f"A person is {g}{rest_dot}",
        f"Someone is {g}{rest_dot}",
        f"A small figure {verb}{rest_dot}",
        f"Watch a person {b}{rest_dot}",
        f"One person {verb}{rest_dot}",
    ]
    # dedupe while keeping order
    seen, unique = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True, help="clips.json for the prompt inventory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    clips = json.loads(Path(args.clips).read_text())
    prompts = sorted({row["text"] for row in clips.values()})
    bank = {p: variants(p) for p in prompts}
    counts = {p: len(v) for p, v in bank.items()}
    Path(args.out).write_text(json.dumps(bank, indent=1) + "\n")
    print(f"{len(bank)} prompts, variants per prompt: min {min(counts.values())} "
          f"max {max(counts.values())}")
    sample = prompts[0]
    for v in bank[sample]:
        print("  ", v)


if __name__ == "__main__":
    main()
