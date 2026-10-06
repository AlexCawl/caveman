"""
Run each prompt through Claude Code in three conditions and snapshot the
real LLM outputs:

  1. baseline      — no extra system prompt at all
  2. terse         — system prompt: "Answer concisely."
  3. terse+skill   — system prompt: "Answer concisely.\n\n{SKILL.md}"

The honest delta is (3) vs (2): how much does the SKILL itself add on top
of a plain "be terse" instruction? Comparing (3) vs (1) conflates the
skill with the generic terseness ask, which is what the previous version
of this harness did.

This is the source-of-truth generator. It calls a real LLM and produces
evals/snapshots/results.json. Run it locally when SKILL.md files change.
The CI-side `measure.py` only reads the snapshot and counts tokens.

Every call is isolated from the machine running it: user settings are not
loaded (so no installed plugins or their SessionStart hooks), no MCP servers,
no installed skills, and the cwd is an empty temp dir so no CLAUDE.md is
discovered. Without this a plugin's injected ruleset, or an MCP auth nag the
model repeats in its answer, lands in every arm including the baseline.

Requires:
  - `claude` CLI on PATH (Claude Code), authenticated

Run: uv run python evals/llm_run.py

Environment:
  CAVEMAN_EVAL_MODEL   optional --model flag value passed through to claude
  CAVEMAN_EVAL_LANG    prompt language: picks prompts/<lang>.txt and the
                       terse prefix for that language (default: en). Any
                       language other than en writes snapshots/results.<lang>.json
  CAVEMAN_EVAL_SKILLS  optional comma-separated skill ids (e.g.
                       caveman,ultracave,megacave) restricting the skill arms;
                       default runs every skills/*/SKILL.md. Unknown ids abort.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Windows consoles and piped stdout default to the ANSI code page (cp1252),
# which cannot encode the arrows, em-dashes and minus signs printed below —
# a diagnostic that crashes instead of printing is worse than useless
# (#203/#459). Replace unencodable characters rather than raising.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


EVALS = Path(__file__).parent
SKILLS = EVALS.parent / "skills"

# One terse control per language. The control must be in the prompt's
# language: an English "Answer concisely." on a Portuguese question also
# nudges the model toward English, which is a second variable.
TERSE_PREFIXES = {
    "en": "Answer concisely.",
    "pt": "Responda de forma concisa.",
    "fr": "Réponds de façon concise.",
}

LANG = os.environ.get("CAVEMAN_EVAL_LANG", "en")
PROMPTS = EVALS / "prompts" / f"{LANG}.txt"
SNAPSHOT = EVALS / "snapshots" / (
    "results.json" if LANG == "en" else f"results.{LANG}.json"
)


def claude_bin() -> str:
    """Resolve the CLI through PATHEXT. npm installs it as claude.CMD on
    Windows and CreateProcess does not apply PATHEXT, so a bare "claude"
    raises FileNotFoundError there."""
    return shutil.which("claude") or "claude"


def run_claude(prompt: str, cwd: str, system_file: Path | None = None) -> str:
    cmd = [claude_bin(), "-p", "--setting-sources", "project",
           "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence"]
    # The skill arm's system prompt is a multi-line SKILL.md with quotes,
    # backticks and `&`. On Windows the CLI is claude.CMD, which cmd.exe
    # re-parses, and that mangles such an argument into an empty prompt
    # ("Input must be provided either through stdin or as a prompt
    # argument"). A file sidesteps every shell.
    if system_file is not None:
        cmd += ["--system-prompt-file", str(system_file)]
    if model := os.environ.get("CAVEMAN_EVAL_MODEL"):
        cmd += ["--model", model]
    cmd.append(prompt)
    out = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=cwd,
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
    )
    return out.stdout.strip()


def claude_version() -> str:
    try:
        out = subprocess.run(
            [claude_bin(), "--version"], capture_output=True, text=True,
            check=True, encoding="utf-8", errors="replace",
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    # Checked here, not at import, so tests can import the module.
    if not PROMPTS.exists():
        available = sorted(p.stem for p in (EVALS / "prompts").glob("*.txt"))
        raise SystemExit(
            f"No prompt set at {PROMPTS}. "
            f"CAVEMAN_EVAL_LANG={LANG!r}; available: {', '.join(available)}"
        )
    if LANG not in TERSE_PREFIXES:
        raise SystemExit(
            f"CAVEMAN_EVAL_LANG={LANG}: no terse control in TERSE_PREFIXES; "
            f"add one next to prompts/{LANG}.txt"
        )
    terse_prefix = TERSE_PREFIXES[LANG]

    prompts = [p.strip() for p in PROMPTS.read_text(encoding="utf-8").splitlines() if p.strip()]
    skills = sorted(p.name for p in SKILLS.iterdir() if (p / "SKILL.md").exists())
    if only := os.environ.get("CAVEMAN_EVAL_SKILLS"):
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        if missing := sorted(wanted - set(skills)):
            sys.exit(f"CAVEMAN_EVAL_SKILLS: no skills/<id>/SKILL.md for {', '.join(missing)}")
        skills = [s for s in skills if s in wanted]

    print(
        f"=== {len(prompts)} prompts × ({len(skills)} skills + 2 control arms) ===",
        flush=True,
    )

    snapshot: dict = {
        "metadata": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "claude_cli_version": claude_version(),
            "model": os.environ.get("CAVEMAN_EVAL_MODEL", "default"),
            "n_prompts": len(prompts),
            "lang": LANG,
            "terse_prefix": terse_prefix,
        },
        "prompts": prompts,
        "arms": {},
    }

    # System prompts get their own temp dir so the cwd stays empty.
    with tempfile.TemporaryDirectory(prefix="caveman-eval-") as cwd, \
            tempfile.TemporaryDirectory(prefix="caveman-eval-sys-") as sysdir:
        print("baseline (no system prompt)", flush=True)
        snapshot["arms"]["__baseline__"] = [run_claude(p, cwd) for p in prompts]

        print("terse (control: terse instruction only, no skill)", flush=True)
        terse_file = Path(sysdir) / "terse.md"
        terse_file.write_text(terse_prefix, encoding="utf-8")
        snapshot["arms"]["__terse__"] = [
            run_claude(p, cwd, terse_file) for p in prompts
        ]

        for skill in skills:
            skill_md = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
            skill_file = Path(sysdir) / f"{skill}.md"
            skill_file.write_text(f"{terse_prefix}\n\n{skill_md}", encoding="utf-8")
            print(f"  {skill}", flush=True)
            snapshot["arms"][skill] = [
                run_claude(p, cwd, skill_file) for p in prompts
            ]

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {SNAPSHOT}")


if __name__ == "__main__":
    main()
