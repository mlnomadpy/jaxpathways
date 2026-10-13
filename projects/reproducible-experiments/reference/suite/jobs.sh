#!/usr/bin/env sh
set -eu
PYTHON=${PYTHON:-python3}
WORKSHOP_SUITE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Independent sequential jobs; review before using an approved accelerator host.
"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id baseline-s11 --timeout 120
"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id baseline-s23 --timeout 120
"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id baseline-s37 --timeout 120
"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id candidate-s11 --timeout 120
"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id candidate-s23 --timeout 120
"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id candidate-s37 --timeout 120
