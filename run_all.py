"""Run the whole pipeline end to end.

    python run_all.py              # everything
    python run_all.py prepare      # one stage

Stages are ordered; each depends on the output of the previous one.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
STAGES = ["prepare", "graph", "baselines", "gnns", "evaluate", "cost", "explain"]


def run(stage: str) -> None:
    print(f"\n{'='*62}\n  {stage}\n{'='*62}")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "src" / f"{stage}.py")],
                       cwd=ROOT / "src")
    if r.returncode:
        sys.exit(f"stage {stage!r} failed with code {r.returncode}")
    print(f"  {stage} completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    wanted = sys.argv[1:] or STAGES
    for s in wanted:
        if s not in STAGES:
            sys.exit(f"unknown stage {s!r}; choose from {STAGES}")
        run(s)
