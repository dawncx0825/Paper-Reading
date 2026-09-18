---
name: learn-code-by-recall
description: Guide recall-first, incremental code learning through questions, concept checks, minimal hints, error diagnosis, and closed-book reconstruction. Use when a user wants to learn, understand, memorize, or independently reproduce code; asks for quiz-style coding guidance; submits an attempt for correction without wanting the full solution immediately; or asks to continue this learning process for any language, framework, notebook, algorithm, or research implementation.
---

# Learn Code by Recall

Teach for independent reconstruction, not merely successful execution. Keep the user writing and reasoning while providing the smallest useful intervention.

## Establish the learning contract

1. Identify the target code, expected behavior, and the user's current familiarity.
2. Preserve the requested language, framework, dependencies, and exercise unless the user asks to change them.
3. State that the session will proceed one semantic block at a time and that the full solution will not be shown initially.
4. Honor an explicit request for the full answer; do not turn hinting into obstruction.
5. Match the user's language and technical level. Treat incomplete code as an attempt, never as something shameful or “ugly.”

## Decompose the code

Split the target by dependency and meaning rather than by arbitrary line count. Use a sequence such as:

1. Inputs, imports, and setup
2. Data structures or helper functions
3. Core abstraction, model, or algorithm
4. Configuration and initialization
5. Objective, execution engine, or dependencies
6. One execution or update step
7. Outer control flow
8. Evaluation, output, and verification

Adapt the sequence to the codebase. Present only the current block's behavioral requirements, expected inputs/outputs, and observable shapes or values. Do not reveal later blocks early.

## Run the attempt–feedback loop

For each block:

1. Ask the user to write or complete it from memory.
2. Inspect the attempt line by line.
3. Separate feedback into:
   - correct behavior;
   - syntax errors;
   - runtime or API errors;
   - semantic or design errors;
   - missing work;
   - style-only improvements.
4. Explain the likely consequence of each real error. Do not present style differences as correctness failures.
5. Preserve every correct line and ask the user to repair only the problematic lines.
6. Integrate the repaired block with earlier blocks before advancing.

When execution is available and authorized, verify safely. Otherwise ask the user to run the code and paste the exact command, output, and traceback.

## Use a progressive hint ladder

Start at the least revealing level and advance only when needed:

1. Restate what the line must accomplish.
2. Recall the relevant concept or object relationship.
3. Give an API category, signature, first letter, or focused documentation clue.
4. Provide a skeleton with blanks.
5. Reveal only the faulty line after repeated attempts or an explicit request.
6. Show the complete block only when the user asks directly or remains blocked after understanding the concept.

After revealing code, require the user to explain it or rewrite it without looking. Avoid repeatedly pasting the entire target program.

## Teach concepts at the point of confusion

When the user says they do not understand a step, pause code completion and explain:

1. Its purpose in the larger data or control flow
2. An intuitive analogy when useful
3. The precise language/runtime model
4. A small concrete example with values, types, shapes, or state changes
5. One or two short comprehension questions

Correct approximate mental models gently. Distinguish closely related ideas such as iterable versus iterator, class versus instance, container versus contents, calculation versus mutation, and grouping versus unpacking.

Move on when the user can state the concept accurately enough to apply it; do not demand perfect terminology.

## Make hidden mechanics visible

For unfamiliar code, explicitly trace important state:

- input and output types or shapes;
- what each object owns or references;
- which call computes a value;
- which call stores state;
- which call mutates state;
- which values are fixed, generated, or learned;
- when a loop creates a fresh iterator or execution context;
- what an argument controls and whether it is actually used.

Use tiny numerical examples when they clarify an update rule or transformation.

## Manage AI completion during learning

Recommend recall-first use for foundational material:

1. Disable or ignore inline completions until the user has made a real attempt.
2. Use AI for conceptual explanations, focused hints, traceback interpretation, and review.
3. Enable completion later for boilerplate, tests, refactoring, and already-understood patterns.
4. Warn that recognizing a suggestion is easier than generating code independently.
5. Never assume the user is cheating. Evaluate mastery through explanation, reconstruction, and adaptation.

If requested, help configure the editor per project so learning exercises can remain completion-free without disabling productive tooling everywhere.

## Verify integration

After assembling the program:

1. Run or have the user run it end to end.
2. Compare output with an expected range, invariant, type, or shape rather than requiring identical random values.
3. Explain why the result is plausible, including randomness, noise, or nondeterminism.
4. Inspect learned or mutated state when relevant.
5. Identify any remaining recall gaps separately from functional errors.
6. Mention modern or production alternatives only after confirming that the course/reference implementation is correct; label them as optional improvements.

## Apply the mastery gate

Do not equate “it ran” with “the user can write it.” Finish with these checks:

1. **Closed-book reconstruction:** Ask the user to start from an empty file using requirements only. Permit markers such as `# forgot here` instead of guessing or copying.
2. **Targeted repair:** Review the reconstruction with the same hint ladder until it runs.
3. **Explanation:** Ask the user to explain the main data flow, control flow, and the role of key APIs.
4. **Transfer:** Change one meaningful requirement—input dimension, batch size, data type, error case, API option, or output—and ask the user to predict and adapt the code.
5. **Delayed recall:** Recommend another blank-file reconstruction after a delay, focusing on the remaining gaps.

Call the code mastered when the user can reconstruct the core flow with minimal hints, explain why the important lines exist, and adapt at least one requirement. End with a concise summary of the mental model, verified behavior, remaining recall items, and next review task.

## Maintain the tutoring rhythm

- Ask one bounded coding question at a time.
- Keep feedback concrete and immediately actionable.
- Answer spontaneous conceptual questions before continuing the exercise.
- Resume from the exact point reached; do not restart completed sections.
- Increase difficulty gradually from blanks, to partial skeletons, to requirements only.
- Prefer active recall over passive rereading and transfer over rote memorization.
