# Project workspace practice

Python 3.10+; standard library only. No JAX installation, account or accelerator required.
Start in this extracted folder. On Windows use `py` instead of `python3`.

```sh
python3 run.py --help
python3 run.py prepare --config configs/baseline.json --run-id baseline-01
python3 -m unittest discover -s tests -v
```

This prepares inputs and provenance; it does not train a model, generate metrics,
or implement checkpoint recovery. Read `GUIDE.md` for the full course section.
A run's `manifest.json` records the environment, source/config/data hashes and
Git state when available. `source/`, `data.snapshot` and `config.json` retain the
small inputs. Preserve `notes.md` with your hypothesis and later observed results.
Do not edit a prepared run to pretend it used different inputs; create a new run.

The tiny CSV is a teaching fixture. Copying it is suitable for this exercise.
For large datasets, keep a versioned storage URI, immutable revision, content
manifest and split definition instead. Runs are ignored by Git; archive evidence
separately before deleting a workspace. Nothing is uploaded automatically.
