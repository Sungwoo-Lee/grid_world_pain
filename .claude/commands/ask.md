---
description: Surface a decision as a structured multiple-choice question via the AskUserQuestion tool, so I pick instead of you guessing. Loops until I say stop.
argument-hint: [optional topic or decision to turn into a question]
---

Use the `AskUserQuestion` tool to align with me on a decision. Do NOT answer in prose — the whole point is the interactive picker.

Handle whichever of these two situations applies:

**A. I gave you a topic** (arguments below are non-empty): Turn `$ARGUMENTS` into a structured question. Work out the 2–4 *realistic* options that a knowledgeable person would actually weigh, and for each option write a one-line description of its trade-off (not a restatement of the label). Put your recommended option first and mark it `(Recommended)`. If the topic bundles more than one independent decision, split it into up to 4 separate questions in a single `AskUserQuestion` call.

**B. No topic given** (arguments empty): Look at the current point in our conversation, identify the most important decision or fork we're at (or that you were about to silently resolve), and surface *that* as the question. If nothing is genuinely open, say so briefly and ask what I want to decide — don't invent a fake fork.

Rules for good options:
- Options must be mutually exclusive and concrete. Show what each choice actually *is*, not an abstract label.
- Use `multiSelect: true` only when the choices are genuinely combinable.
- Use the `preview` field when a side-by-side comparison helps (code snippets, config diffs, layout mockups).
- Keep each question's `header` to a short chip (≤12 chars).
- Don't ask about things you can verify yourself in the repo — ask only what's genuinely mine to decide.

**Always end every round with a control question.** The LAST question in every `AskUserQuestion` call must be a follow-up that asks whether to continue, with `header: "Next"` and options like:
- `Stop here` — we're aligned; act on the answers, no more questions.
- `Ask more` — surface the next decision or fork as a fresh `AskUserQuestion` round.

(Since a call allows up to 4 questions, reserve one slot for this control question — keep the substantive questions to 3 or fewer per round so it always fits.)

After I answer: if I picked `Ask more`, immediately fire another `AskUserQuestion` round (which itself ends with the same control question), and keep looping until I pick `Stop here`. Once I pick `Stop here`, stop asking and proceed with everything I've chosen.

Topic: $ARGUMENTS
