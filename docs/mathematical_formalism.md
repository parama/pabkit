# Mathematical Formalism of Process-Aware Benchmarking (PAB)

This document provides a detailed mathematical formulation of the key metrics and concepts used in Process-Aware Benchmarking (PAB).

## Learning Trajectory

At the core of PAB is the concept of a learning trajectory, defined as the sequence of model states (hypotheses) over time:

$$\mathcal{P} = \{h_{\theta_t}\}_{t=1}^T$$

where $h_{\theta_t}$ represents the model hypothesis at step $t$ during training, and $T$ is the total number of training steps.

## Key PAB Metrics

### 1. Learning Trajectory Stability

A robust model should demonstrate a gradual and structured learning process. We define stability as the smoothness of the loss trajectory:

$$S(\mathcal{P}) = \frac{1}{T-1} \sum_{t=1}^{T-1} \left| R(h_{\theta_t}) - R(h_{\theta_{t+1}}) \right|$$

where $R(h_{\theta_t})$ is the validation loss at step $t$. Smaller $S(\mathcal{P})$ values indicate a more stable learning trajectory.

### 2. Generalization Efficiency

Instead of evaluating generalization only at convergence, we define instantaneous generalization efficiency $G(t)$ as:

$$G(t) = R_{\text{train}}(h_{\theta_t}) - R_{\text{test}}(h_{\theta_t})$$

A well-trained model should maintain low $G(t)$ throughout training. The average generalization efficiency over the entire training process is given by:

$$\bar{G} = \frac{1}{T} \sum_{t=1}^T G(t)$$

### 3. Rule Evolution

Learning should involve the refinement of structured abstractions rather than abrupt shifts in representation. We measure rule formation divergence as:

$$R_{\text{evo}} = \frac{1}{T} \sum_{t=1}^T \|{\phi_t - \phi_{t-1}}\|_2$$

where $\phi_t$ is the feature representation at step $t$. A model that learns structured rules should exhibit smooth, interpretable rule evolution with gradually decreasing $R_{\text{evo}}$ values.

### 4. Learning Curve Predictability

In human learning, expertise develops through structured exposure and refinement. We define learning curve predictability as:

$$\mathcal{P}_{\text{learn}} = \mathbb{E}\left[ \left(R(h_{\theta_t}) - R(h_{\theta_{t-1}})\right)^2 \right]$$

Lower values indicate smoother and more predictable learning, akin to structured human learning.

## Class-Wise Learning Analysis

To analyze how different classes are learned, we define:

### Early-Learning Classes

Classes $c$ whose accuracy exceeds threshold $\tau$ within the first third of training:

$$\text{Early}(\tau) = \{c \mid \exists t \leq T/3 : \text{Acc}_c(t) \geq \tau\}$$

### Late-Learning Classes

Classes $c$ whose accuracy only exceeds threshold $\tau$ in the last third of training:

$$\text{Late}(\tau) = \{c \mid \forall t \leq 2T/3 : \text{Acc}_c(t) < \tau \land \exists t > 2T/3 : \text{Acc}_c(t) \geq \tau\}$$

### Unstable Classes

Classes $c$ that show inconsistent learning (accuracy rises above threshold $\tau$ but later drops below it):

$$\text{Unstable}(\tau) = \{c \mid \exists t_1 < t_2 : \text{Acc}_c(t_1) \geq \tau \land \text{Acc}_c(t_2) < \tau\}$$

## Adversarial Robustness Evolution

To evaluate robustness over time, we define robustness at step $t$ as:

$$\text{Rob}(t) = \frac{\text{Acc}_{\text{adv}}(t)}{\text{Acc}_{\text{clean}}(t)}$$

where $\text{Acc}_{\text{adv}}(t)$ and $\text{Acc}_{\text{clean}}(t)$ are the model accuracies on adversarially perturbed and clean data, respectively.

We calculate robustness degradation as:

$$\text{RobDeg} = \frac{\max_t \text{Acc}_{\text{adv}}(t) - \text{Acc}_{\text{adv}}(T)}{\max_t \text{Acc}_{\text{adv}}(t)}$$

A high value indicates significant deterioration in robustness during later stages of training.

## Representation Similarity Analysis

To analyze how representations evolve during training, we define similarity between representations at different time steps using cosine similarity:

$$\text{Sim}(t_1, t_2) = \frac{\phi_{t_1} \cdot \phi_{t_2}}{\|\phi_{t_1}\| \|\phi_{t_2}\|}$$

This allows us to create a similarity matrix that visualizes how representations change over time.

## Process-Aware Evaluation

Unlike traditional benchmarks that evaluate models at a single endpoint $T$, PAB considers the entire learning trajectory:

$$\mathcal{E}_{\text{PAB}} = \sum_{t=1}^T w_t \mathbb{E}_{(x,y) \sim \mathcal{D}}[L(f_t(x), y)]$$

where $w_t$ is a weight function that emphasizes critical learning phases and $L$ is a loss function.

## Comparing Models Using PAB

Given two models with learning trajectories $\mathcal{P}_A$ and $\mathcal{P}_B$, PAB allows us to compare them based on:

1. Overall stability: $S(\mathcal{P}_A)$ vs. $S(\mathcal{P}_B)$
2. Generalization efficiency: $\bar{G}_A$ vs. $\bar{G}_B$
3. Class learning patterns: Early, late, and unstable classes
4. Robustness evolution: $\text{RobDeg}_A$ vs. $\text{RobDeg}_B$

## PAB Alignment with Learning Theory

PAB is aligned with Probably Approximately Correct (PAC) learning theory, which defines learning in terms of generalization probability rather than deterministic correctness.

In PAC learning, a hypothesis $h$ is considered successful if:

$$P_{\mathcal{D}}[h(x) \neq c(x)] \leq \epsilon$$

where $c$ is the true concept and $\epsilon$ is the error parameter.

PAB extends this by evaluating not just whether a model achieves this bound, but how efficiently it does so during training, and how stable its learning process is.

## Mathematical Basis for PAB Recommendations

PAB provides recommendations based on thresholds for various metrics:

1. **Overfitting Detection**: If $G(T) > G(T/2)$ and the validation loss increases after reaching a minimum.
2. **Robustness Degradation Warning**: If $\text{RobDeg} > \tau_{\text{rob}}$ (typically 0.1 or 10%).
3. **Training Instability Alert**: If $\text{Std}(S(\mathcal{P})) > \tau_{\text{stab}}$ (typically 0.1).
4. **Early Stopping Recommendation**: At epoch $t^*$ where $R_{\text{test}}(h_{\theta_{t^*}})$ is minimized.

These mathematically grounded criteria help identify specific issues in model training that might not be apparent from final accuracy alone.
