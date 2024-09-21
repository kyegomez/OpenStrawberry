[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Open Strawberry

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

Algorithm: RLHF with DPO and Monte Carlo Tree of Thoughts

Initialize:
- Policy model \( \pi_{\theta} \)
- Reference model \( \pi_{\theta_{\text{ref}}} \leftarrow \pi_{\theta} \) (copy of policy model)
- Freeze parameters of \( \pi_{\theta_{\text{ref}}} \)
- Reward model \( r_{\phi} \)
- Hyperparameters:
  - Batch size \( N \)
  - Number of thought branches \( B \)
  - Number of preferred/unpreferred sequences per prompt \( K \)
  - DPO beta parameter \( \beta \)

Training Loop:
Repeat until convergence:

1. **Sample Prompts:**
   - For \( i = 1 \) to \( N \):
     - Sample prompt \( x_i \) from the training data.

2. **Generate Thought Sequences:**
   - For each prompt \( x_i \):
     - For \( j = 1 \) to \( B \):
       - Generate sequence \( y_{i,j} \) by sampling from the policy model:
         \[
         y_{i,j} \sim \pi_{\theta}(\cdot \mid x_i)
         \]

3. **Evaluate Sequences with Reward Model:**
   - For each sequence \( y_{i,j} \):
     - Compute reward:
       \[
       r_{i,j} = r_{\phi}(x_i, y_{i,j})
       \]

4. **Rank Sequences:**
   - For each prompt \( x_i \):
     - Sort sequences \( \{ y_{i,j} \} \) based on rewards \( \{ r_{i,j} \} \) in descending order.

5. **Select Preferred and Unpreferred Sequences:**
   - For each prompt \( x_i \):
     - **Preferred sequences:** Top \( K \) sequences \( \{ y_{i,k}^{\text{pref}} \} \) where \( k = 1, \ldots, K \).
     - **Unpreferred sequences:** Bottom \( K \) sequences \( \{ y_{i,k}^{\text{unpref}} \} \) where \( k = 1, \ldots, K \).

6. **Create Sequence Pairs:**
   - For each prompt \( x_i \) and \( k = 1 \) to \( K \):
     - Pair preferred and unpreferred sequences:
       \[
       \left( y_{i,k}^{\text{pref}}, y_{i,k}^{\text{unpref}} \right)
       \]

7. **Compute Log Probabilities:**
   - For each pair \( (x_i, y_{i,k}^{\text{pref}}, y_{i,k}^{\text{unpref}}) \):
     - **Policy Model:**
       \[
       \begin{align*}
       \log \pi_{\theta}^{\text{pref}} &= \log \pi_{\theta}(y_{i,k}^{\text{pref}} \mid x_i) \\
       \log \pi_{\theta}^{\text{unpref}} &= \log \pi_{\theta}(y_{i,k}^{\text{unpref}} \mid x_i)
       \end{align*}
       \]
     - **Reference Model:**
       \[
       \begin{align*}
       \log \pi_{\theta_{\text{ref}}}^{\text{pref}} &= \log \pi_{\theta_{\text{ref}}}(y_{i,k}^{\text{pref}} \mid x_i) \\
       \log \pi_{\theta_{\text{ref}}}^{\text{unpref}} &= \log \pi_{\theta_{\text{ref}}}(y_{i,k}^{\text{unpref}} \mid x_i)
       \end{align*}
       \]

8. **Compute Log-Ratios:**
   - **Policy Log-Ratio:**
     \[
     \Delta \log \pi_{\theta} = \log \pi_{\theta}^{\text{pref}} - \log \pi_{\theta}^{\text{unpref}}
     \]
   - **Reference Log-Ratio:**
     \[
     \Delta \log \pi_{\theta_{\text{ref}}} = \log \pi_{\theta_{\text{ref}}}^{\text{pref}} - \log \pi_{\theta_{\text{ref}}}^{\text{unpref}}
     \]

9. **Compute DPO Loss:**
   - For each pair \( (x_i, y_{i,k}^{\text{pref}}, y_{i,k}^{\text{unpref}}) \):
     \[
     \mathcal{L}_{i,k} = -\log \sigma\left( \beta \left[ \Delta \log \pi_{\theta} - \Delta \log \pi_{\theta_{\text{ref}}} \right] \right)
     \]
     where \( \sigma \) is the sigmoid function.

10. **Aggregate Loss:**
    - Compute average loss over all pairs:
      \[
      \mathcal{L} = \frac{1}{N \times K} \sum_{i=1}^{N} \sum_{k=1}^{K} \mathcal{L}_{i,k}
      \]

11. **Backpropagation and Update:**
    - Backpropagate the loss \( \mathcal{L} \) to compute gradients with respect to \( \theta \).
    - Update policy model parameters \( \theta \) using an optimizer (e.g., SGD, Adam).

12. **Optional Reference Model Update:**
    - Optionally update \( \pi_{\theta_{\text{ref}}} \) periodically or according to a defined schedule.

Repeat the training loop until convergence criteria are met (e.g., loss below a threshold, maximum number of iterations).

**Notes:**
- **Monte Carlo Sampling:** Used to explore multiple branches of thoughts by generating diverse sequences.
- **Reward Model \( r_{\phi} \):** Trained separately to evaluate the quality of generated sequences.
- **DPO Beta Parameter \( \beta \):** Controls the trade-off between the policy and reference models.
- **Sequence Padding and Masks:** Ensure proper handling of variable-length sequences during computation (omitted here for brevity but should be handled in implementation).

**Key Functions:**
- **Log Probability Computation:** Calculate the log probability of a sequence given a model and prompt.
- **Sigmoid Function \( \sigma(z) \):** Defined as \( \sigma(z) = \frac{1}{1 + e^{-z}} \).

**Mathematical Functions:**
- **\(\log\):** Natural logarithm.
- **\(-\log \sigma(z)\):** Negative log-sigmoid function used in DPO loss.

**Final Output:**
- An optimized policy model \( \pi_{\theta} \) capable of generating high-quality sequences as rated by the reward model.

