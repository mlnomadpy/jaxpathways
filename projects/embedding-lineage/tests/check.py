"""Run one cumulative stage using the learner starter by default."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', default='starter', help='starter (default), solution, or a Python file path')
    parser.add_argument('--stage', choices=['1', '2', '3', '4', 'all'], required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    implementation = args.implementation
    if implementation == 'starter':
        implementation = str(ROOT / 'starter/model.py')
    elif implementation == 'solution':
        implementation = str(ROOT / 'solution/model.py')
    else:
        candidate = Path(implementation)
        if not candidate.is_absolute(): candidate = Path.cwd() / candidate
        implementation = str(candidate.resolve())
    stages = [1, 2, 3, 4] if args.stage == 'all' else [int(args.stage)]
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if args.implementation == 'solution' and args.stage == 'all' and args.output is None:
        with TemporaryDirectory(prefix='embedding-lineage-reference-') as temp:
            output = temp
            for stage in stages:
                command = [sys.executable, str(ROOT / 'run.py'), '--implementation', implementation,
                           '--stage', str(stage), '--output', output]
                subprocess.run(command, check=True, cwd=Path.cwd(), env=child_env)
    else:
        output = args.output or 'runs/embedding-lineage/learner-run'
        for stage in stages:
            command = [sys.executable, str(ROOT / 'run.py'), '--implementation', implementation,
                       '--stage', str(stage), '--output', output]
            subprocess.run(command, check=True, cwd=Path.cwd(), env=child_env)
    print(f'PASS embedding-lineage stages {", ".join(map(str, stages))}')

if __name__ == '__main__': main()
