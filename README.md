[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Open Strawberry

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


```latex

\documentclass{article}
\usepackage{amsmath, amssymb, algorithm, algorithmicx, algpseudocode, hyperref}

\begin{document}

\title{RLHF Algorithm with PPO and Monte Carlo Tree Search}
\author{}
\date{}
\maketitle

\begin{algorithm}
\caption{RLHF Algorithm using PPO and Monte Carlo Tree Search}
\begin{algorithmic}[1]

\State \textbf{Input:} 
\begin{itemize}
    \item Initial policy $\pi_\theta$ with parameters $\theta$
    \item Reward model $R$
    \item Thought space $\mathcal{T}$
    \item Step size $\alpha$
    \item Entropy coefficient $\beta$
    \item PPO clip parameter $\epsilon$
    \item Horizon $T$
    \item Monte Carlo branches $B$
    \item Monte Carlo rollouts per branch $M$
\end{itemize}

\State Initialize PPO policy $\pi_\theta$ with random weights $\theta$
\State Initialize thought tree $\mathcal{T} \gets \{\text{root node}\}$

\While{not converged}
    \State \textbf{Step 1: Thought Generation}
    \For{each thought trajectory $t_i \in \mathcal{T}$, $i = 1, \ldots, B$}
        \State Generate branches of thoughts from policy $\pi_\theta$
        \State Perform $M$ rollouts per branch for trajectory $t_i$
    \EndFor
    
    \State \textbf{Step 2: Thought Evaluation using Reward Model}
    \For{each trajectory $t_i \in \mathcal{T}$}
        \State Evaluate trajectory with reward model: $r_i \gets R(t_i)$
    \EndFor
    
    \State \textbf{Step 3: Monte Carlo Tree Search (MCTS) Expansion}
    \For{each thought trajectory $t_i \in \mathcal{T}$}
        \State Expand tree using thought branches based on $r_i$ and exploration strategy
    \EndFor
    
    \State \textbf{Step 4: Policy Optimization with PPO}
    \For{each thought trajectory $t_i \in \mathcal{T}$}
        \State Compute advantage estimate: 
        \[
        A_i = r_i - V(t_i)
        \]
        \State Update policy using PPO objective:
        \[
        \mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[\min \left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t, \text{clip}\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon \right) A_t \right) \right]
        \]
    \EndFor
    
    \State Update policy parameters:
    \[
    \theta \gets \theta + \alpha \nabla_\theta \mathcal{L}^{\text{PPO}}(\theta)
    \]
    
    \State \textbf{Step 5: Entropy Regularization}
    \State Add entropy term to encourage exploration:
    \[
    \mathcal{L}^{\text{entropy}}(\theta) = \beta \sum_{t} \pi_\theta(a_t|s_t) \log \pi_\theta(a_t|s_t)
    \]
    \State Update total loss:
    \[
    \mathcal{L}^{\text{total}}(\theta) = \mathcal{L}^{\text{PPO}}(\theta) + \mathcal{L}^{\text{entropy}}(\theta)
    \]

\EndWhile

\State \textbf{Return:} Optimal policy $\pi_\theta$

\end{algorithmic}
\end{algorithm}

\end{document}


```