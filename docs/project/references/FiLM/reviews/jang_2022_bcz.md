---
title: "BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning"
authors: ["Eric Jang", "Alex Irpan", "Mohi Khansari", "Daniel Kappler", "Frederik Ebert", "Corey Lynch", "Sergey Levine", "Chelsea Finn"]
year: 2022
venue: "Conference on Robot Learning (CoRL 2021)"
slug: jang_2022_bcz
source_pdf: docs/project/references/FiLM/sources/Jang et al. 2022 - BC-Z - Zero-shot task generalization with robotic imitation learning.pdf
topic: FiLM
---

# BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning

## Plain-English entry point

This paper is the canonical "FiLM-on-a-policy" application in robotics. The team at Google Robotics, Everyday Robots, UC Berkeley, and Stanford asks a simple-sounding question: **can a real robot arm perform a manipulation task it has never been shown — when the user describes that task in natural language or by demonstrating it on video?** Their answer is empirically yes: BC-Z (Behavior Cloning, Zero-shot) reaches 44 % success on 24 brand-new manipulation tasks without a single robot demonstration of those tasks.

The recipe combines three pieces. *(1)* **Behavior cloning** — learn a policy by imitating an expert who teleoperates the robot, rather than by trial and error (reinforcement learning). *(2)* **Task conditioning** — instead of training one policy per task, train a single policy that takes the *task description* as an input alongside the image; the description is encoded into a **task embedding** $z$, a 512-dim vector that summarises "what the user wants". *(3)* **FiLM conditioning** — the task embedding is injected into the visuomotor policy by **Feature-wise Linear Modulation**: at every residual block of the ResNet-18 image encoder, the task vector is linearly projected into channel-wise scales and shifts that re-modulate that block's feature maps. This is the same FiLM mechanism Perez et al. 2018 introduced for visual question answering — BC-Z is the load-bearing demonstration that the same idea scales to real-robot control.

The other half of the paper is data engineering: a teleoperation rig that mixes pure demonstration with **HG-DAgger** (Human-Gated DAgger), where the human operator only intervenes when the learned policy is about to err, yielding 25,877 demonstrations across 100 tasks plus 18,726 human videos for video-conditioned generalization.

## Section-ordered backbone

**1. Introduction.** The grand challenge is task-level generalization in vision-based robotic manipulation. Prior imitation work handles new objects or new goal configurations but not new tasks. BC-Z attacks this by combining (a) shared-autonomy data collection so corrective demonstrations are cheap to gather, and (b) flexible policy conditioning on a continuous task embedding (language string or human video) rather than a discrete one-hot identifier, since only the former can extrapolate to unseen tasks.

**2. Related Work.** Positions BC-Z against (a) one-shot imitation methods that condition on a single robot demo, (b) language-conditioned manipulation, (c) HG-DAgger / EIL interactive imitation, (d) sim-to-real generalization. BC-Z's novelty is *scale plus task-conditioning*: 100 training tasks, 7-DoF closed-loop control at 10 Hz, real hardware, and conditioning that supports zero-shot language or few-shot video.

**3. Problem Setup and Method Overview.** Defines the conditional policy $\mu : \mathcal{S} \times \mathcal{W} \to \mathcal{A}$ where $\mathcal{S}$ is RGB images, $\mathcal{W}$ is task commands (language or video), and $\mathcal{A}$ is 7-DoF end-effector control. The policy factorises as an **encoder** $q(z \mid w)$ that produces a task embedding, and a **control layer** $\pi(a \mid s, z)$ that maps image plus embedding to action. The decomposition lets pretrained language embeddings structure $\mathcal{Z}$.

**4. Data Collection and Workflow.** Oculus VR teleoperation, 10 Hz non-realtime loop, 6–15 household objects randomised on a table, 100 pre-specified tasks spanning 9 skills (pushing, pick-and-place, wiping, stacking, …). Shared-autonomy phase aggregates 11,108 expert-only demos with 14,769 HG-DAgger intervention demos over 16 deployment iterations. The intervention rate serves as a *live* proxy for policy performance, which the authors empirically validate.

**5. Learning Algorithm.** The language encoder is a frozen pretrained multilingual sentence encoder producing 512-dim vectors. The video encoder is a learned ResNet-18 trained end-to-end with an **auxiliary cosine-distance regression loss** that aligns the video embedding to the language embedding of the same task — without this auxiliary loss, video embeddings overfit to initial scenes and generalise poorly. Policy training uses **Huber loss** on XYZ and axis-angle, log-loss on the gripper. Two design choices are flagged: **open-loop auxiliary prediction** (predict the next 10 actions as a regularisation signal even though only the first is executed) and **state-differences as actions** (target $\Delta$ pose to $N>1$ steps in the future, adaptively chosen, to avoid dithering at 10 Hz).

**5.3 Network Architecture (the FiLM section).** RGB image → ResNet-18 torso → branched MLP action heads for $\Delta$XYZ, $\Delta$axis-angle, gripper. The 512-dim task embedding $z$ is injected via **FiLM layers** at each of the 4 ResNet blocks: $z$ is linearly projected into channel-wise scales $\gamma$ and shifts $\beta$, which modulate that block's activations. This is the load-bearing architectural choice.

**6. Experimental Results.** (6.1) Single-task validation: 3.4 picks/min on bin-emptying, 87 %–94 % on door opening. (6.2) Zero-shot/few-shot: language-conditioned BC-Z averages 32 % across 28 held-out tasks, 44 % over the 24 with non-zero success; video-conditioning averages only 4 %. A diagnostic Table 3 compares one-hot vs. language vs. video on *training* tasks — one-hot (42 %) and language (40 %) are nearly tied, which shows the language embedding is *not* the bottleneck; the control layer is. (6.3) Ablations: single-task baseline collapses to 5 % (need multi-task); HG-DAgger beats pure expert demos at fixed data budget; adaptive state-diff is critical (without it, success drops from 45 % to 3 %).

**7. Discussion.** 100 training tasks is sufficient to provoke task-level generalization; HG-DAgger is essential; frozen pretrained language embeddings beat learning task embeddings from scratch. Limitations: video-conditioning lags language; last-centimetre errors (gripper closure / object release) dominate failures; the "(verb) (noun)" command structure is rigid.

## Phase 1 — Undergraduate-level synthesis

**The setup.** Train one robot to do many household manipulation tasks, then have it perform brand-new tasks it has never been shown. The trick is to let the user *tell* the robot what to do, either with a sentence ("place the sponge in the tray") or by showing a video of a human doing it.

**The architecture.** Inside the robot's neural network there are two parts. (1) A **task encoder** turns the user's command into a 512-dimensional vector $z$ — this is the "task embedding". For language commands they use an off-the-shelf sentence encoder (pretrained on text data, never updated). For video they train a small CNN, but they add a trick: the video encoder has to *predict the language embedding* of the same task as an auxiliary objective. (2) A **visuomotor policy** is a ResNet-18 that processes the camera image; it outputs the next action (where to move the gripper, whether to open or close it). The task vector $z$ is injected into this ResNet using **FiLM layers**: at each block of the ResNet, $z$ is projected into a vector of scales and a vector of shifts, one number per feature channel, and the activations of that block are multiplied by the scales then added to the shifts. So the same network does different things for "pick up the apple" and "wipe the tray" — the FiLM scales/shifts re-purpose its features for the current task.

**Why FiLM?** Because the task command is fundamentally a *modulator* of perception and action, not a separate input to be concatenated. FiLM is a clean, low-parameter way to let one input "steer" a much larger feature stack. It is also the natural place to plug in a pretrained language encoder — the encoder doesn't have to know anything about robotics, it just produces a vector that the FiLM layers translate into "do this task right now".

**Headline finding.** Scaling demonstrations to 100 tasks plus a pretrained-language FiLM-conditioned policy generalises *zero-shot* to 24 of 29 new tasks (44 % success on those 24). The simplest version — frozen pretrained language encoder + FiLM-conditioned ResNet + HG-DAgger data — beats more complex alternatives.

## Phase 2 — Graduate-level deep dive

### 2.1 Conditional policy decomposition

The policy is the composition of an encoder and a control layer:

$$
\mu(a \mid s, w) \;=\; \pi(a \mid s, z), \qquad z \sim q(z \mid w),
$$

with $s \in \mathcal{S}$ an RGB image, $w \in \mathcal{W}$ the task command (language string $w_\ell$ or human video $w_h$), $z \in \mathcal{Z} = \mathbb{R}^{512}$, and $a \in \mathcal{A}$ a 7-DoF action (6-DoF end-effector pose + gripper). For language, $q$ is the frozen multilingual sentence encoder ("Universal Sentence Encoder", Yang et al. 2019). For video, $q$ is a learned ResNet-18.

### 2.2 Video-encoder objective with language regression

The video encoder is trained jointly with the policy, but to prevent it from overfitting to early-frame appearance the authors add a **language-regression auxiliary loss**:

$$
\min_{q,\pi} \;\; \mathbb{E}_{(s,a) \sim \mathcal{D}_e^i,\, w_h \sim \mathcal{D}_h^i}\!\Big[\underbrace{-\log \pi(a \mid s, z_h^i)}_{\text{behavior cloning}} \;+\; \underbrace{D_{\cos}\!\big(z_h^i,\, z_\ell^i\big)}_{\text{language regression}}\Big],
$$

with $z_h^i \sim q(\cdot \mid w_h^i)$ (video embedding) and $z_\ell^i \sim q(\cdot \mid w_\ell^i)$ (language embedding for the same task $i$), and $D_{\cos}(u,v) = 1 - \frac{u \cdot v}{\|u\|\,\|v\|}$ the cosine distance. Because robot demonstration videos are *also* videos of the task, they are co-trained as inputs to $q$ with the same auxiliary loss to anchor the embedding space.

### 2.3 FiLM conditioning on the policy network (the core "FiLM-on-policy" mechanism)

This is the section the curator should cite when comparing BC-Z to Perez et al. 2018 (`perez_2018_film.md`, B1) and Turkoglu et al. 2022 (`turkoglu_2022_film_ensemble.md`, B3).

Let $F^{(k)} \in \mathbb{R}^{H_k \times W_k \times C_k}$ denote the activation of the $k$-th of the four ResNet blocks of the visuomotor policy, indexed channel-wise as $F^{(k)}_{:, :, c}$ for $c = 1, \dots, C_k$. For each block $k$, BC-Z computes channel-wise scale and shift parameters from the 512-dim task embedding $z$ via two learned linear projections:

$$
\boldsymbol{\gamma}^{(k)}(z) \;=\; W_\gamma^{(k)} z + b_\gamma^{(k)}, \qquad
\boldsymbol{\beta}^{(k)}(z) \;=\; W_\beta^{(k)} z + b_\beta^{(k)},
$$

with $W_\gamma^{(k)}, W_\beta^{(k)} \in \mathbb{R}^{C_k \times 512}$ and $b_\gamma^{(k)}, b_\beta^{(k)} \in \mathbb{R}^{C_k}$. The FiLM modulation is then applied channel-wise, broadcasting across spatial dimensions:

$$
\widetilde F^{(k)}_{h, w, c} \;=\; \gamma^{(k)}_c(z) \cdot F^{(k)}_{h, w, c} \;+\; \beta^{(k)}_c(z), \qquad k = 1, \dots, 4.
$$

Following Perez et al. 2018, this is **affine modulation by a task-dependent gain and bias**, not by an entry-wise mask. The number of FiLM parameters per block is $2 C_k \times (512 + 1)$ — a tiny overhead relative to the ResNet itself. Crucially, the policy's visual feature extractor is *the same network* across all 100 tasks; the FiLM parameters are what redirect it task-by-task.

### 2.4 Joint policy loss

Letting $\hat a = (\hat a_{\text{xyz}}, \hat a_{\text{rot}}, \hat a_{\text{grip}})$ denote the policy's predicted action and $a^\star$ the expert label, with $a^\star_{\text{xyz}}$ a $\Delta$-pose target computed by the adaptive state-diff scheme over a horizon $N \ge 1$:

$$
\mathcal{L}_{\text{policy}} \;=\; \mathcal{L}_{\text{Huber}}\!\big(\hat a_{\text{xyz}}, a^\star_{\text{xyz}}\big) \;+\; \mathcal{L}_{\text{Huber}}\!\big(\hat a_{\text{rot}}, a^\star_{\text{rot}}\big) \;+\; \mathcal{L}_{\text{BCE}}\!\big(\hat a_{\text{grip}}, a^\star_{\text{grip}}\big) \;+\; \lambda\,\mathcal{L}_{\text{open-loop}},
$$

with the Huber loss

$$
\mathcal{L}_{\text{Huber}}(x, y) \;=\; \begin{cases} \tfrac{1}{2}(x-y)^2 & \text{if } |x - y| \le \delta \\ \delta\,|x-y| - \tfrac{1}{2}\delta^2 & \text{otherwise} \end{cases}
$$

and $\mathcal{L}_{\text{open-loop}}$ the analogous loss applied to the network's *auxiliary* 10-step rollout, predicted from the same image but never executed (it only serves as a training-time regulariser).

### 2.5 Why FiLM and not concat?

A naïve alternative is to concatenate $z$ to a flattened activation map before the action heads. BC-Z's design choice — FiLM at every ResNet block — is motivated by Perez et al.'s observation that **conditional features are most useful when injected throughout the depth** of a vision network, because the task should bias *what visual features get computed*, not merely *how the final features get read out*. BC-Z does not run an ablation isolating FiLM vs. concat, but the design follows Perez et al.'s prescription. The empirical fact that one-hot and language conditioning achieve near-identical *training-task* performance (42 % vs. 40 % in Table 3) is the cleanest evidence that the conditioning channel is *sufficient*: when language replaces one-hot, the policy continues to act task-appropriately, so FiLM is reading the embedding rather than ignoring it.

### 2.6 Adaptive state-difference as action target

At 10 Hz, raw expert actions are tiny and noisy. BC-Z relabels actions as differences to a future pose $N$ steps ahead:

$$
a^\star_t \;=\; \mathrm{pose}(s_{t+N}) \;-\; \mathrm{pose}(s_t), \qquad N \;=\; \min\bigl\{n \ge 1 : \|\mathrm{pose}(s_{t+n}) - \mathrm{pose}(s_t)\| \ge \tau\bigr\},
$$

choosing $N$ adaptively per timestep so that each label encodes a meaningful displacement (threshold $\tau$ set separately for arm motion and gripper). The ablation collapse from 45 % to 3 % when $N = 1$ underscores how brittle behavior cloning is to action-scale mismatch.

### 2.7 Empirical signature for the project's own modulation work

For our project's NMN-as-hyperparameter-modulation line, BC-Z's two cleanest findings are: (i) **a 512-dim task embedding produced by an external pretrained encoder is enough to redirect a visuomotor ResNet across 100+ tasks via FiLM**, and (ii) **the modulation channel is bottlenecked by the controller, not by the embedding** (Table 3: one-hot ≈ language on training tasks; large drop only on held-out tasks). For a modulator-as-controller-of-other-network analog, BC-Z is the empirical proof-of-concept that *low-dim continuous modulators can steer a much larger feature stack without retraining the base network.*

## Connections

- **`perez_2018_film.md` (B1, canonical FiLM)** — BC-Z is the load-bearing real-world application of Perez et al.'s FiLM mechanism, scaling it from the CLEVR visual-reasoning benchmark to a 7-DoF real robot at 10 Hz. The FiLM equations in §2.3 above are *the same affine $\gamma, \beta$ modulation* introduced in `perez_2018_film.md`; only the conditioner has changed (sentence/video embedding instead of question embedding).
- **`turkoglu_2022_film_ensemble.md` (B3, FiLM-Ensemble)** — Where BC-Z uses *one* FiLM per task to redirect a single network, FiLM-Ensemble uses *$M$ FiLM channels* to instantiate $M$ implicit ensemble members in the same backbone. BC-Z thus sits "upstream" of FiLM-Ensemble: it shows that one FiLM channel suffices to redirect a visuomotor backbone, which is the precondition for using FiLM channels as ensemble heads.
- **Multimodal meta-learning cluster (Abdollahzadeh et al. 2021, Takeda et al. 2021)** — BC-Z's "one shared backbone, FiLM-conditioned per task" architecture is the workhorse version of the multi-task / mixed-task FiLM setup these meta-learning papers analyse. The auxiliary cosine-regression loss for video embeddings is conceptually related to multi-task FiLM training when one task modality is much harder to learn end-to-end than the other.
- **Project-side relevance** — For NMN-as-modulator: BC-Z is the cleanest evidence in the corpus that *low-bandwidth task signals injected via FiLM at every layer* can re-purpose a fixed feature extractor across many behaviours. The 512→2$C_k$ projection per block is the architectural template for "an interoceptive modulator that re-conditions a control policy on internal state". Follow-up review hand-off would go to `senior-developer` if a FiLM-in-recurrent-PPO implementation is desired.
