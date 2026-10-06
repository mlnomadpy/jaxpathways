# 11 — Data and backups

Existing location: `notebook.html#backups`. Goal: protect local work without interrupting ordinary learning.

## Main view

```text
[← My learning]
Your data stays on this browser.
Take a backup before switching devices or clearing browser data.

[Download backup]                      [Restore backup]

Included: reading positions, practice/checkpoints, project reports,
selected plans, and saved artifact records.
Linked notebooks and code are not copied into the backup.
> What is stored and how merging works
```

Download uses the existing versioned backup contract. Restore opens the native JSON file picker using a real button. No hidden mouse-only import trigger. No cloud sync claim.

## Restore preview — proposed addition

```text
Restore this backup?
Compatible format / source filename
Summary of incoming records
Merge with work already on this browser.
Existing records will follow the documented collision policy.
[Restore and merge]                    [Cancel]
```

The preview must use the actual importer rules, not invent a last-write-wins policy. Inspect version, shape, IDs, and collisions before a write. Earlier accepted notebook formats remain supported. No automatic destructive replacement mode.

## Success / failure

```text
Restore complete.
Actual counts: added / updated / skipped records
[Return to My learning]                [Download merged backup]
```

Wrong file or malformed JSON: explain the problem and offer Choose another file; leave existing data intact. Unsupported version: identify it and offer current backup export. Partial storage failure: report actual outcome, retain incoming content for recovery, and offer export; never say all restored unless every intended write succeeded.

## Mobile

Stack the two main action buttons. Preview and result are full-width sections beneath the heading. Long filenames wrap. Return to the trigger after canceling a disclosure or confirmation. Restore completion goes to the result heading and is announced.

Acceptance: import supports keyboard and file-picker cancellation; merged exports include all current supported record families; records referenced by absent lessons remain recoverable and inspectable.
