⚡ Engineered by Kiliandiama | The Diama Protocol [10/10] | All rights reserved.

MentalAgent V3+

A neural architecture for emotionally-aware reinforcement learning agents combining a Transformer encoder, a recurrent state-space model (RSSM), emotion modeling, and a safe policy/value RL head.

This project explores how latent emotional dynamics can be integrated into sequential decision-making systems.

🧠 Overview

MentalAgent V3+ is composed of four main components:

Encoder (Transformer): encodes sequential observations (context + emotions)
RSSM (Recurrent State-Space Model): learns a stochastic latent memory of emotional dynamics
Emotion Head: predicts a distribution over discrete emotional states
Safe Policy + Value Heads: reinforcement learning module with PPO-style training
🏗️ Architecture
1. Encoder

Transforms input sequences into contextual embeddings using a Transformer encoder.

Input: [B, T, D]
Output: [B, T, D]
2. RSSM (Emotional Memory)

A stochastic recurrent latent model that learns temporal emotional dynamics.

Learns prior: p(z_t | z_{t-1})
Learns posterior: q(z_t | z_{t-1}, h_t)
Uses GRU-based state transition

Outputs:

Latent states z
Prior/posterior distributions
3. Emotion Head

Maps latent state to a categorical emotion distribution.

Input: [B, T, Z]
Output:
Emotion probabilities
Emotion logits
4. Safe Policy (PPO-style)

A stochastic policy with tanh-squashed Gaussian actions.

Features:

Log-normal action distribution
Entropy regularization
Designed for stable and safe exploration
5. Value Network

Estimates state value for PPO training stability.

Input: latent state z
Output: scalar value
