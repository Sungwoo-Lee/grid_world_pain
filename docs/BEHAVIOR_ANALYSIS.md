# Behavioral Analysis Guidelines

This document outlines how to measure "pain-like" behavior in our JAX-based grid-world RL setup (actions = **L/R/U/D**, plus **Rest** and **Eat**). We treat pain as a persistent cost/impairment state and quantify changes in the agent's policy and physiology.

---

## 1. Ongoing Pain-like State and Relief-seeking

### Professor's Guidelines
* **Rest-seeking / rest fraction:** % of time steps choosing **Rest**, especially while injured vs. not injured.
* **Latency to first rest after injury:** How quickly the agent switches to resting once injury occurs.
* **Rest clustering:** Rest in long bouts vs. short intermittent rests (bout length distribution).
* **Heal efficiency:** HP gained per unit time spent resting; also “over-resting” after HP is already high.

### Grid World Implementation
* **Rest Fraction**: Calculate the mean of `event_rested` (or where `action == 'Rest'`). 
  > **Formula**: `sum(event_rested) / total_steps` (Filter by `injury > 0` for injured states).
* **Latency to Rest**: The number of steps between a damage event (`damage_total > 0`) and the first subsequent `event_rested == True`.
* **Rest Clustering**: Analyze the `rest_streak` column. Calculate the mean and maximum of `rest_streak` values where the streak ends (i.e., step $t$ where $streak_t > 0$ and $streak_{t+1} = 0$).
* **Heal Efficiency**: The rate of change in the `injury` column during `event_rested` periods.
  > **Note**: Our environment features "recovery acceleration" (exponential healing during streaks), which can be visualized by plotting `injury` vs. `rest_streak`.

---

## 2. Movement Suppression and Functional Impairment

### Professor's Guidelines
* **Total distance traveled / steps per episode:** Injury should reduce movement if moving is costly.
* **Exploration reduction:** Unique grid cells visited; area covered; entropy of visited states.
* **Speed proxy:** Average non-rest steps per 100 ticks; time-to-goal inflation.
* **Avoidance of effortful terrain:** Fraction of steps through high-cost cells.

### Grid World Implementation
* **Total Distance**: Calculate the cumulative sum of Manhattan distance between `(pos_r, pos_c)` at step $t$ and $t-1$.
* **Exploration Area**: Count the number of unique coordinate pairs `(pos_r, pos_c)` logged in `ep_stats.csv`.
* **Speed Proxy**: Percentage of steps where the action is one of `[Up, Right, Down, Left]`.
* **Functional Inflation**: Compare `step` count to reach the nearest food source when `injury == 0` vs. `injury > threshold`.

---

## 3. Avoidance / Guarding Analogs (Protective Behavior)

### Professor's Guidelines
* **Risk avoidance:** Time spent in “danger zones” (cells that increase injury probability) vs. safe zones.
* **Path choice shift:** Compare shortest path length vs. chosen path length; detours taken to reduce expected injury.
* **Re-entry rate to harmful zones:** How often the agent re-enters risky areas after being injured.

### Grid World Implementation
* **Risk-Zone Occupancy**: Count steps where `pos_r, pos_c` equals the coordinates of a known threat (e.g., `res_i` where `type == danger` or `predator_i`).
* **Hazard Sensitivity**: Check `damage_danger`, `damage_predator`, and `damage_obstacle` columns. High avoidance is indicated by a reduction in these values over training episodes.
* **Re-entry Analysis**: Once `damage_total > 0` occurs at a specific coordinate, flag that coordinate and count subsequent entries to that same location within the episode.

---

## 4. Trade-offs: Pain vs. Hunger (Motivational Conflict)

### Professor's Guidelines
* **Eat-vs-rest prioritization under conflict:** When both satiety is low and HP is low, which action comes first?
* **Thresholds / policy switching points:** Satiety level at which agent chooses Eat despite injury; HP level at which it stops resting despite hunger.
* **Opportunity-cost sensitivity:** How much reward the agent gives up to rest/eat.

### Grid World Implementation
* **Conflict Resolution**: Identify steps where `satiation < threshold` AND `injury > threshold`. Observe the `action` taken.
* **Motivational Switching**: Plot `satiation` vs. `injury` at the moments of action transitions (e.g., switching from `Rest` to `Move` or `Eat`).
* **Drive Comparison**: Analyze the `drive_hunger` vs. `drive_injury` columns in `ep_stats.csv`. These represent the internal scalars driving the homeostatic reward.

---

## 5. Decision Quality Under Pain

### Professor's Guidelines
* **Goal achievement rate / success probability** under different injury severities.
* **Regret / suboptimality:** Difference between achieved return and an oracle planner.
* **Policy stability:** Does injury make behavior more stochastic or more conservative?

### Grid World Implementation
* **Success Rate**: Calculate % of episodes ending in `termination_reason == 1` (Max Steps) vs. `termination_reason == 4` (Death by Injury).
* **Efficiency Loss**: Compare the `reward_extrinsic` (food eaten) per step in high-injury vs. low-injury episodes.
* **Suboptimality**: Measure the detour ratio (actual path length / Manhattan distance to food) during injured states.

---

## 6. “Affective” Analogs (Aversion and Relief Valuation)

### Professor's Guidelines
* **Value of relief:** Estimated Q(Rest) − Q(Move) while injured; bigger gap = stronger preference for relief.
* **Conditioned place preference analog:** Preference time in “safe areas” after injury.
* **Negative reinforcement signature:** Does the probability of choosing Rest increase after Rest improves HP?

### Grid World Implementation
* **Preference Strength**: In `evaluation.py`, analyze the action logits (the `logits` variable during inference). The difference between the logit for `Rest` and the next best action is a direct proxy for the "Value of Relief."
* **Learning-from-Relief**: During training, correlate the increase in `reward_homeostatic` (relief reward) with the frequency of `Rest` actions in subsequent steps.

---

## Summary Panel Metrics

For a standard "Behavioral Battery" report, aggregate the following from `ep_stats.csv`:

1.  **Rest Fraction** (Injured vs. Control)
2.  **Exploration Efficiency** (Unique Cells / Total Steps)
3.  **Risk Exposure** (Total Damage Taken per Episode)
4.  **Healing Latency** (Steps to first Rest after Injury)
5.  **Conflict Bias** (Frequency of Eat-first vs. Rest-first under dual-need)
6.  **Survival Robustness** (Mean Satiation/Injury before episode termination)
