---
name: wake
description: "Schedule a self-wakeup in THIS session using the ScheduleWakeup tool, then re-arm until a done-condition is met (a lightweight self-paced poller). Use on the explicit slash command /wake or phrases like 'wake yourself in N minutes and check X', 'poll X every N min until done', 'keep checking X and ping me when it finishes', 'remind yourself to re-check X'. Takes an interval (e.g. 5m, 20min, 1h) plus a task that includes a DONE condition. On each firing it runs the task, reports one line of progress, and — if not done — calls ScheduleWakeup again with the same /wake input so the next firing re-enters this skill; when the condition is met it reports and does NOT reschedule, so the loop ends. Session-bound and short-horizon by design: ScheduleWakeup clamps the delay to 60-3600s (1 min to 1 hour) and re-invokes the CURRENT session, so it dies if the session ends. For recurring automation that must survive the session or fire days out, use /schedule (cron cloud agent) instead; for a fixed-interval in-session loop over a slash command, /loop is the general tool. Trigger /wake stop to cancel a pending wakeup."
---

# /wake — self-paced wakeup poller on `ScheduleWakeup`

Schedule a future re-invocation of THIS session and, by default, keep re-arming until a
condition is met. This is a thin, user-facing wrapper over the `ScheduleWakeup` tool — the
same primitive that paces `/loop` dynamic mode — packaged as a named command with a
poll-until-done default.

## What it is / is NOT

- **IS**: "wake me in N minutes, do a check, and repeat until X is true, then stop and report."
- **IS NOT**: cross-session or long-horizon scheduling. `ScheduleWakeup` clamps the delay to
  **60–3600s** and re-invokes the **current session** only. If this session ends, nothing wakes.
  - Need it to survive the session / fire on a calendar (daily 9am, tomorrow 3pm) → **`/schedule`**.
  - Need a plain fixed-interval loop over a slash command while you work → **`/loop`**.

## Invocation

```
/wake <interval> <task with a done-condition>
/wake stop
```

- `<interval>`: human duration — `90s`, `5m`, `20min`, `1h`. Parse to seconds, then **clamp to [60, 3600]**.
  If the user gives no interval, default to `600` (10 min).
- `<task>`: what to do on each firing. It should contain (or imply) a **DONE condition** —
  the thing that, once true, ends the poll. If none is stated, ask for one, or treat the first
  successful completion as done (one-shot).
- `/wake stop`: cancel any pending wakeup — call `ScheduleWakeup` with `{ stop: true }` and nothing else.

## Procedure (run every time the skill fires)

1. **Do the task / check.** Run whatever the task specifies (read a file, check a run's status,
   grep a log, query GPU state, etc.). Keep it to what's needed to evaluate the done-condition.
2. **Report one line** of progress to the user (what you checked, the current state). Terse.
3. **Evaluate the done-condition.**
   - **Met** → report the result plainly and **do NOT reschedule**. Not calling `ScheduleWakeup`
     is what ends the loop. Optionally note the loop is complete.
   - **Not met** → call `ScheduleWakeup` with:
     - `delaySeconds`: the parsed, clamped interval.
     - `prompt`: the **same `/wake <interval> <task>` input, verbatim**, so the next firing
       re-enters this skill and repeats the poll. (Never the `<<autonomous-loop-dynamic>>`
       sentinel — that's for autonomous, prompt-less loops, not this user-driven one.)
     - `reason`: one short, specific sentence on what you're waiting for and the chosen delay
       (e.g. "run still training at step 40k/100k; re-check in 5m"). This is shown back to the user.
4. **On the next firing**, the harness re-invokes this session with that prompt → start again at step 1.

## Picking the delay

Match the delay to how fast the watched state actually changes — don't poll a slow thing every
minute. A run that checkpoints every ~10 min deserves a ~600s cadence, not 60s. When a background
task (a subagent, a `run_in_background` job) is the real signal, prefer a longer fallback cadence
(the harness will re-invoke you the moment that task completes anyway) — see the ScheduleWakeup
guidance on fallback heartbeats. Always `log`/state the cadence you chose and why, via `reason`.

## Guardrails

- **Session-bound**: state this to the user if they ask for anything longer than 1 hour or
  "even after I close it" — redirect to `/schedule`.
- **No silent infinite polls**: if a sensible max number of ticks or a wall-clock bound is implied,
  honor it and tell the user when you stop for that reason rather than because the condition was met.
- **Idempotent stop**: `/wake stop` should be safe to call even if nothing is pending.
