# C04 acquisition review and blocked recipe

Sol independently matched three acquired archive identities and official MD5
values, all 5,639 regular member hashes, and 5,472 decoded audio/transcript
identities. Acquisition is accepted within that measured scope.

The inventory and training recipe are not accepted. The retained original lock
is historical and rejected for grouping/target identity. Independent review
found three defects that Luna is repairing:

1. Chapter minutes were parsed as book ID, and authors as book title. Correcting
   the metadata gives 79 source components, 78 with eligible utterances, and two
   cross-source components. The earlier 72/five counts are superseded.
2. Inventory target hashes use zero for spaces while the declared vocabulary
   reserves zero for CTC blank and one for spaces. This affects 5,466 rows.
3. Identical valid feature frames produce different valid logits when padded
   widths have different parity. The recorded maximum difference is 2.28483.

The existing synthetic checker passed within its exercised scope. Sol's rerun
changed only the checker's external scratch path in a temporary copy, preserving
historical output; the receipt records both identities. That pass did not cover
the independent padding-width counterexample or establish a complete frontend,
encoder, recovery, training or ASR-quality result.

`feasibility-only.json` shows that current sources can support a small custom
pilot without larger downloads or relaxed duration rules. Its illustrative
whole-component allocation is not an approved split. The general fit pool after
quarantine is only about 16 minutes. Exact roles, seconds/frame budgets, runtime
bindings and all remaining lifecycle gates require another review.

No corpus optimization was authorized or performed at this handoff. Audio files,
trained arrays and full utterance inventories are not included in this compact
packet. External reproducer and source snapshot paths are retained in the repair
ledger and receipts; `sha256.json` identifies these retained review records.
