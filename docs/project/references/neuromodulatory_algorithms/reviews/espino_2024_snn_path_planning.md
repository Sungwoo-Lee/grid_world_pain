---
title: "A Rapid Adapting and Continual Learning Spiking Neural Network Path Planning Algorithm for Mobile Robots"
authors:
  - Harrison Espino
  - Robert Bain
  - Jeffrey L. Krichmar
year: 2024
venue: "IEEE Robotics and Automation Letters, 9(11), 9542–9549"
slug: espino_2024_snn_path_planning
source_pdf: "sources/Espino et al. 2024 - A rapid adapting and continual learning Spiking Neural Network path planning algorithm for mobile robots.pdf"
topic: neuromodulatory_algorithms
---

# Espino, Bain & Krichmar (2024) — A rapid adapting and continual learning SNN path planner

## Plain-English entry point

This paper from the Krichmar lab at UC Irvine asks a very concrete robotics question: **can a ground robot learn a map of the world by driving around on it, and then plan good paths over that map, using only a spiking neural network (SNN) — the kind of brain-inspired computation that runs efficiently on low-power neuromorphic chips?** The motivation is that conventional planners (A*, RRT*) are fast but cannot learn from experience, and deep-reinforcement-learning planners can learn but need huge offline training and expensive onboard compute.

The authors build a system with two pieces. The first is the **Spiking Wavefront Planner (SWP)**: every waypoint in a 17 × 17 grid in a real outdoor park (Aldrich Park, UCI) is an integrate-and-fire neuron, and a spike injected at the start neuron *propagates* through the grid with axonal delays. Larger delays mean harder/costlier terrain, so the wave naturally bends around obstacles and the back-traced path is the shortest in *cost*, not in distance. The second is **E-Prop**, a local learning rule that approximates back-propagation-through-time using an eligibility trace, which updates those axonal delays online from sensor measurements.

Three real costs are measured by a Clearpath Jackal robot — wheel current draw, LiDAR-detected obstacles, and IMU-measured slope — and learned continuously over ~12 hours of outdoor driving. The result: the SWP plans paths that are significantly **shorter and cheaper** than A*, RRT*, or D* Lite when measured on the *learned* cost map, can adapt to a newly placed obstacle in just one or two trials, and — crucially — runs as pure spiking arithmetic, so it could be deployed on Intel Loihi or SpiNNaker for low-power autonomous navigation. The work matters here as a real-world demonstration that **biologically plausible online learning (E-Prop on SNN) can solve a continual-learning task**, not just a toy benchmark.

## Section-ordered backbone

**I. Introduction.** Three open problems in robot navigation are named: (1) building a usable cost map from noisy multi-sensor data, (2) continual learning without periodic offline retraining, (3) flexibility under environmental change and shifting navigation goals (e.g. minimise battery use today, avoid foot traffic tomorrow). The paper's contribution is a navigation system that does all three by combining the Spiking Wavefront Planner with E-Prop, tested on a Clearpath Jackal ground robot in Aldrich Park. Spiking implementation makes future deployment on neuromorphic hardware feasible.

**II. Related works.** Three families of prior planners are reviewed. *Graph- and sampling-based planners* (A*, D* Lite, RRT, RRT*, RRT^X) are mature and provably optimal under their cost assumptions but assume a fixed cost map. *SLAM* (EKF-SLAM, GraphSLAM, ORB-SLAM, RatSLAM) recovers geometry but not traversability — slope, unevenness, foot traffic — and assumes the trajectory is decided externally. *Active SLAM* combines mapping with planning but optimises map quality rather than task-specific costs. *Deep-learning planners* (imitation learning, model-free RL) require either expert demos or extensive offline training plus heavy compute, so they cannot continually learn in the field.

**III. Background.**
- **III-A. Spiking Wavefront Planner (SWP).** Each grid cell is an integrate-and-fire neuron. Membrane voltage $v_i$, recovery variable $u_i$, and input current $I_i$ are updated discretely; spikes propagate through axonal delays $D_{ij}$ to neighbours. Delays — *not* synaptic weights — encode the traversal cost. A path is read out by initiating a spike at the start neuron and back-tracing the first arrival at the goal.
- **III-B. E-Prop.** From Bellec et al. 2020 (*Nat. Commun.*), E-Prop is a local approximation of BPTT for spiking RNNs that uses eligibility traces. Espino et al. apply E-Prop *to the delays* $D_{ij}$ along traversed paths, with eligibility decided by how recently a neuron spiked during the most recent wave propagation. Learning rate $\delta = 0.5$, trace decay $\tau = 25$.

**IV. Experimental setup.** Robot: Clearpath Jackal with NovaTel GPS (1.2 m precision) and Microstrain 3DM-GX5 IMU. Environment: hilly Aldrich Park with paved road, grass, dirt road, trees, benches, foot traffic. Grid: 17 × 17, 5.1 m spacing. Training: 350 trials over 4 days / ~12 hours. Comparison planners: A*, RRT*, D* Lite, and a naïve shortest-Euclidean-distance planner. 25 random physical paths plus exhaustive simulation of 57 086 paths on the learned cost map.

**V. Methods.**
- **V-A. Cost measures.** Three costs are tracked per edge: **current cost** (minimum of left/right wheel current, to filter turning spikes), **obstacle cost** (fraction of traversal time with LiDAR-detected obstacles within 2 m and 270° aperture), **slope cost** (pitch + roll from IMU). An intraversable-location cost (max delay = 10) is added if a waypoint is unreachable within 45 s. Costs are normalised separately to 1–10 (clamped at mean ± 2 SD) and then added when a combined map is needed.
- **V-B. Environment mapping.** Each trial: random end point sampled from a Lévy flight (good exploration mix of local and long jumps); SWP plans path; robot orients to bearing via heading control (rotate-and-go when angle error < π/12); E-Prop updates $D_{ij}$ along the executed path. The robot stops, rotates, and continues if heading-bearing error grows; if a waypoint is unreachable within 45 s the robot retreats.

**VI. Results.**
- **A. Environment mapping.** Current cost is nearly uniform across grass / dirt / pavement but elevated at tree-edge transitions. Obstacle cost is sparse and localised at trees, benches, and foot-traffic zones. Slope cost concentrates in the south-east hill quadrant. The combined-cost planner picks a route along grass that avoids both obstacles and the steepest slope.
- **B. Continual learning and adaptation.** MSE between current and final $D_{ij}$ drops steadily across the 12-hour run. When an obstacle is placed in a learned path, the robot updates its cost map within one or two trials (paths re-route first up a hill, then off the road through flat grass).
- **C. Comparisons.**
  - *Physical trials* (n = 25): SWP gives significantly shorter paths than RRT* and lower obstacle cost; A* is the closest competitor with no significant differences.
  - *Exhaustive simulation* (n = 57 086): SWP significantly beats RRT*, A*, D* Lite on every cost measure and on path length. Advantage *grows with path length*.
  - *Dynamic-replanning comparison* (n = 10 000 simulated mid-route cost changes): D* Lite wins on path length and obstacle cost; SWP wins on slope and current.
  - *Runtime*: A* and D* Lite are fastest on standard hardware; SWP and RRT* are slower today but SWP is parallelisable on neuromorphic chips.

**VII. Conclusion and future work.** SWP + E-Prop gives continual online cost-map learning, rapid adaptation, and lower-cost paths than classical planners. Limitations: costs are point-experiential (no generalisation across waypoints), learning is slow per edge, and standard-hardware runtime is high. Future directions: vision-based self-labelling, biologically inspired memory replay (Espino, Bain & Krichmar 2023 IJCNN), more robust obstacle detection, **neuromodulation for combining cost maps** (Xing, Zou & Krichmar 2020 — neuromodulated patience), deployment on Loihi or DYNAP-SE.

## Phase 1 — Undergraduate-level synthesis

Think of the robot's world as a grid of waypoints. The big idea is to put **one spiking neuron at each waypoint** and connect each neuron to its eight neighbours through links that have an *axonal delay*. A bigger delay means it is "more expensive" (in time, energy, obstacle exposure, or slope) to travel along that link.

To plan a path from A to B, the robot makes neuron A spike. That spike travels through the network, choosing the route with the shortest total delay (= cheapest cost). When the wave reaches B, the system traces backwards through which neighbour fired first, and that sequence is the planned path. This is the *Spiking Wavefront Planner*.

But the robot does not know the delays at the start — they all begin at 1. So as the robot drives along the planned path, three sensors measure how hard it actually was: how much current the wheels drew, how often LiDAR saw an obstacle, how steep the slope felt to the IMU. After each trip a learning rule called *E-Prop* takes those measurements and **updates the delays along the path** to better match reality. Over 12 hours of driving in Aldrich Park, the delays converge to a true cost map.

The result: the SNN planner finds shorter and cheaper paths than the textbook planners A* and RRT*, and it can react to a newly placed obstacle after just one or two trips. The system is "lifelong" because there is no offline retraining phase — every drive is a training step.

Why does this matter? Spiking arithmetic is exactly what neuromorphic chips (Intel Loihi, SpiNNaker, DYNAP-SE) accelerate, so a future small drone could run this planner at milliwatt power. And because the same network can plan over *any* combination of costs (just add the delays), the same robot can re-prioritise "avoid foot traffic" today and "save battery" tomorrow without retraining.

## Phase 2 — Graduate-level deep dive

### Spiking neuron dynamics

Each grid cell $i$ is a simplified integrate-and-fire neuron with discrete-time membrane potential update (Eq. 1):

$$
v_i(t+1) \;=\; u_i(t) + I_i(t+1),
$$

with **recovery variable** $u_i$ (Eq. 2):

$$
u_i(t) \;=\;
\begin{cases}
\beta & \text{if } v_i(t) = 1 \\
\min\bigl(u_i(t-1) + 1, \; 0\bigr) & \text{otherwise.}
\end{cases}
$$

Here $\beta = -10$ is a deep refractory offset chosen large enough that a spike cannot reactivate previously visited nodes — this is essential for the wavefront to propagate uniquely away from the source. The recovery variable then climbs linearly back to a baseline of zero, so any visited neuron is "frozen out" for $|\beta| = 10$ time-steps.

The **input current** sums delayed spikes from the eight neighbours $j \in \mathcal{N}(i)$ (Eq. 3):

$$
I_i(t+1) \;=\; \sum_{j=1}^{N} \mathbb{1}\!\bigl[\, d_{ij}(t) = 1 \,\bigr],
$$

i.e., a neighbour contributes 1 unit of current when its remaining-delay counter $d_{ij}$ reaches 1. That counter is initialised by a neighbour spike and decremented per time-step (Eq. 4):

$$
d_{ij}(t+1) \;=\;
\begin{cases}
D_{ij} & \text{if } v_j(t) \ge 1 \\
\max\bigl( d_{ij}(t) - 1, \; 0 \bigr) & \text{otherwise.}
\end{cases}
$$

$D_{ij}$ is the **learned axonal delay** from $j$ to $i$ and encodes the traversal cost of edge $(j, i)$. Initially $D_{ij} = 1$ for all edges (uninformative prior).

### Wavefront semantics

If a spike is injected at start neuron $s$ at $t = 0$, the time at which destination neuron $g$ first spikes equals the minimum over all paths $\pi$ from $s$ to $g$ of $\sum_{(j,i) \in \pi} D_{ij}$, exactly the shortest-cost path. Back-tracing follows, for each neuron, the *neighbour whose delivered spike triggered its own spike* — the resulting node sequence is the optimal path. This is functionally equivalent to Dijkstra on a graph with non-negative integer edge weights but implemented purely with local spike propagation.

### E-Prop delay update

After the robot physically traverses the planned path, the **delay update rule** (Eq. 5) is:

$$
D_{ij}(T+1) \;=\; D_{ij}(T) \;+\; \delta \cdot e_i(t) \bigl( m_{xy} - D_{ij}(T) \bigr),
$$

with $\delta = 0.5$ the learning rate, $e_i(t)$ the eligibility trace of neuron $i$, $m_{xy}$ the (normalised, integer-clamped) measured cost at the physical location $(x, y)$ corresponding to neuron $i$, and $T$ the trial index. This is a Robbins-Monro style update of the form $D \leftarrow D + \delta \cdot e \cdot (m - D)$, equivalent to an exponentially weighted moving average toward the measured cost gated by recency-of-firing.

The **eligibility trace** (Eq. 6) follows:

$$
e_i(t+1) \;=\;
\begin{cases}
1 & \text{if } v_j(t) \ge 1 \\
e_i(t) - \dfrac{e_i(t)}{\tau} & \text{otherwise,}
\end{cases}
$$

with decay constant $\tau = 25$. The trace is reset to 1 each time the neuron spikes during the wave and decays multiplicatively otherwise, so only neurons that participated in the most recent wave-front near the goal are eligible for update — this is the spatial credit assignment that E-Prop provides without storing the full BPTT graph.

### Cost-map normalisation

Each cost measure $c \in \{\text{current}, \text{obstacle}, \text{slope}\}$ is independently normalised to integers in $[1, 10]$:

$$
m^{(c)}_{xy} \;=\; \text{round}\!\left( 1 + 9 \cdot \frac{\text{clip}\bigl(m^{(c)}_{xy,\text{raw}}, \, \mu_c - 2\sigma_c, \, \mu_c + 2\sigma_c\bigr) - (\mu_c - 2\sigma_c)}{4\sigma_c} \right),
$$

with $\mu_c, \sigma_c$ obtained from a pre-training calibration drive between waypoints. The intraversable-waypoint cost is set to the ceiling 10. To combine costs, delays are summed across selected costs and renormalised back to $[1, 10]$ to maintain fast wave propagation.

### Computational mechanism summary

The whole system is a *self-supervised online cost-map learner*:

1. Plan via a spike wave through current $D_{ij}$.
2. Drive and measure $m^{(c)}_{xy}$.
3. Compute eligibility $e_i(t)$ from the wave.
4. Update $D_{ij}$ via Eq. (5) and re-normalise.

There are no labels, no demonstrations, no offline replay, and no global error signal. All learning is local to the (neuron, delay) tuple. The state required during a trial is small: one $D_{ij}$ per directed edge ($\le 8 \times 17^2$), one eligibility scalar per neuron, and the current spike state. This is what makes the algorithm a candidate for neuromorphic implementation — Intel Loihi and DYNAP-SE both support programmable synaptic delays and local update rules.

### Empirical findings

On exhaustively simulated long paths the SWP's mean cost is significantly lower than A*, D* Lite, RRT*, and the naïve Euclidean planner; performance gap *increases* with path length, indicating SWP's advantage compounds when many edge choices are available. On dynamic mid-route replanning the SWP and D* Lite are comparable (D* Lite wins length; SWP wins slope and current). Adaptation latency to a newly placed obstacle is **one to two trials**.

## Connections to other papers in this corpus

- **Direct Krichmar-lab lineage.** This paper is the latest in a chain of Krichmar SWP works: it cites Hwu, Wang, Oros & Krichmar 2018 (the original SWP on neuromorphic TrueNorth) and Krichmar, Ketz, Pilly & Soltoggio 2022 (SWP + E-Prop in grid-world simulation). The 2024 paper is the first to put both together on a physical robot in an outdoor environment.
- **Replay link.** Reference [32] — "Selective memory replay improves exploration in a spiking wavefront planner" (Espino, Bain & Krichmar 2023, IJCNN) — explicitly connects this line to the *episodic replay* mechanism of Kudithipudi et al. 2022 (`kudithipudi_2022_lifelong_learning`, this batch). The future-work section names replay as the next addition.
- **Neuromodulation link.** Reference [35] — Xing, Zou & Krichmar 2020, *Neuromodulated patience for robot and self-driving vehicle navigation* — is named in the conclusion as the future mechanism for **combining cost maps**. This is the corpus-internal link to the neuromodulation-as-gain papers (Rodriguez-Garcia et al. 2026 `rodriguezgarcia_2026_ne_stability_gap` in this batch; Ferguson & Cardin 2020 `ferguson_cardin_2020_gain_modulation`, Wainstein et al. 2025 `wainstein_2025_gain_perceptual_switches`, Shine et al. 2021 `shine_2021_cellular_to_dynamics` in other batches).
- **Spiking continual-learning siblings.** AlKilany & Goodman 2025 (`alkilany_goodman_2025_snn_dynamic_sensory`, this batch) is the other SNN paper in this batch and shares the spiking-arithmetic / neuromorphic-deployability framing. Together with Tambaş et al. 2025 (`tambas_2025_krotov_hopfield_rbm`, this batch), these three papers form the "neuromorphic-substrate" wing of the corpus.
- **Krichmar / lifelong-learning umbrella.** Krichmar is a co-author on Kudithipudi et al. 2022 (`kudithipudi_2022_lifelong_learning`, this batch), Avery & Krichmar 2017 (`avery_krichmar_2017_models_neuromodulation`, other batch), Cox & Krichmar 2009 (`cox_krichmar_2009_neuromodulation_robot`, other batch), Hwu & Krichmar 2020 (`hwu_krichmar_2020_schemas`, other batch), Chiba & Krichmar 2020 (`chiba_krichmar_2020_self_monitoring`, other batch), Krichmar 2013 (`krichmar_2013_anxious_curious`, other batch), and Krichmar & Hwu 2022 (`krichmar_hwu_2022_neurorobotic_principles`, other batch). This paper sits in the *robot-deployment* corner of that body of work.
- **E-Prop / Bellec et al.** Bellec et al. 2020 *Nat. Commun.* "A solution to the learning dilemma for recurrent networks of spiking neurons" is the upstream methodological paper for the learning rule, not in the corpus directly but cross-referenced by AlKilany & Goodman 2025 and by Espino's previous work.
