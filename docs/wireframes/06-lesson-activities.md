# 06 — Lesson activity views

Views inside the reader, not five mandatory separate pages. Use only activities supported by the authored lesson. Proposed prediction interactions require real distractors and explanatory feedback; do not generate superficial quizzes from prose automatically.

## A. Setup / first run

```text
Set up your learning workspace
You will save a program and verify one change.
[Download starter workspace]

Terminal instructions
(●) macOS / Linux          ( ) Windows PowerShell

1 Unpack the workspace
2 Create the tested environment          [Copy command]
3 Run the saved program                  [Copy command]
EXPECTED OUTPUT
(actual expected output from lesson)

[It matches]                  [Help with an error]
> Use a notebook instead
```

OS choices are visible radio tiles; device detection may suggest a tile but never silently choose commands. Preserve the Windows execution-validation caveat. It matches records a self-report only. Help with an error reveals named symptoms: Missing module / Wrong interpreter / File not found / Different output / Something else. Each opens the relevant existing troubleshooting explanation; no paste-an-error field is required.

Mobile stacks OS tiles and code panels; long commands scroll within the panel, with a wrap option. Next activity does not mark setup complete.

## B. Understand

```text
How does this function change?
f(x) = x²         x = 3         slope = 6
Meaningful tangent / computation diagram
Short explanation + worked example
> A deeper explanation
[Next: predict a result]
```

Diagram labels have readable text equivalents. Optional interaction changes authored example values through labeled choices, not freeform input. A static example is acceptable where interaction adds no insight.

## C. Predict

```text
Before running: what will grad(f)(3.0) return?
( ) 3.0       ( ) 6.0       ( ) 9.0
[Check prediction]

AFTER CHECK
6.0 is correct: the derivative is 2x.
If you chose 9.0, you evaluated the function rather than its derivative.
[Try again]                  [Continue to Run]
```

No answer selected: Check is unavailable with Choose an answer helper text. Check does not run Python. Reveal feedback only after submission; announce it once. Wrong answers lead to explanation, not a score penalty. Save an attempt separately from practice. No confetti, streak pressure, or claim of mastery.

## D. Run

```text
Run your experiment
[Download notebook]       > Python script / setup help
CODE                                      [Copy]
EXPECTED RESULT
(actual output / tolerance / shape contract)
[My output matches]                 [Help with an error]
```

Downloads never run or install automatically. Expected output is a comparison target, not a captured user result. Numerical tolerances come from the lesson, never invented by the UI. Show reference code under Inspect the reference after the learner instructions.

## E. Change

```text
Change one condition
Recommended experiment: change the batch size.
Prediction: which axis should change?
[Leading axis] [Feature axis] [Neither]
[Reveal explanation]

Modify your downloaded code, run it, and compare.
> Another experiment: change the dtype
[Continue to Check]
```

Offer one meaningful recommended experiment and optional alternatives. If a lesson calls for a derivation or code writing, keep that real task; clickable choices cannot replace it.

## F. Check / reflect

```text
Check your understanding
Authored misconception question + selectable answers
[Check answer]

PRACTICE REPORT
[ ] I ran my changed example and compared the result.
> Keep a note or link to my work (optional)
[Next lesson]
```

Practice reporting is reversible. Save work carries lesson and activity context automatically. Continue is available whether or not a note is saved. Feedback survives activity navigation; storage failure shows Not saved on this browser while preserving the current attempt in memory.
