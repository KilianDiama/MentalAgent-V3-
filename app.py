import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# Encoder (émotions + contexte)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(d_model, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: [B, T, D]
        src_key_padding_mask: [B, T] (True = pad)
        """
        x = self.input_proj(x)
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(h)


# ============================================================
# RSSM (mémoire émotionnelle)
# ============================================================
class RSSM(nn.Module):
    def __init__(self, d_model=256, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.gru = nn.GRUCell(latent_dim + d_model, latent_dim)

        self.prior = nn.Linear(latent_dim, 2 * latent_dim)
        self.posterior = nn.Linear(latent_dim + d_model, 2 * latent_dim)

        # état initial appris (plus stable que zéro fixe)
        self.z0 = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, h_seq, mask=None):
        """
        h_seq: [B, T, D]
        mask: [B, T] (1 = valide, 0 = pad) ou None
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        if mask is None:
            mask = torch.ones(B, T, device=device)

        z = self.z0.unsqueeze(0).expand(B, -1)  # [B, latent_dim]

        zs, prior_mus, prior_logvars = [], [], []
        post_mus, post_logvars = [], []

        for t in range(T):
            h = h_seq[:, t]  # [B, D]
            m = mask[:, t].unsqueeze(-1)  # [B, 1]

            # prior p(z_t | z_{t-1})
            prior_mu, prior_logvar = self.prior(z).chunk(2, -1)

            # posterior q(z_t | z_{t-1}, h_t)
            post_in = torch.cat([z, h], dim=-1)
            post_mu, post_logvar = self.posterior(post_in).chunk(2, -1)

            std = torch.exp(0.5 * post_logvar)
            eps = torch.randn_like(std)

            z_sample = post_mu + std * eps

            # GRU update
            gru_in = torch.cat([z_sample, h], dim=-1)
            z_next = self.gru(gru_in, z)

            # masking (pour séquences de longueurs différentes)
            z = m * z_next + (1 - m) * z

            zs.append(z)
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)
            post_mus.append(post_mu)
            post_logvars.append(post_logvar)

        return (
            torch.stack(zs, 1),           # [B, T, Z]
            torch.stack(prior_mus, 1),    # [B, T, Z]
            torch.stack(prior_logvars, 1),
            torch.stack(post_mus, 1),
            torch.stack(post_logvars, 1),
        )


# ============================================================
# Emotion Head (clé pour santé mentale)
# ============================================================
class EmotionHead(nn.Module):
    def __init__(self, latent_dim=128, n_emotions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, n_emotions)
        )

    def forward(self, z):
        """
        z: [B, T, Z] ou [B, Z]
        """
        logits = self.net(z)
        return torch.softmax(logits, dim=-1), logits


# ============================================================
# Policy SAFE (pas toxique)
# ============================================================
class SafePolicy(nn.Module):
    def __init__(self, latent_dim=128, action_dim=32, log_std_init=-0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.SiLU(),
        )

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, z):
        """
        z: [B, T, Z] ou [B, Z]
        Retourne:
            action: même shape que mean, tanh-squashed
            logp: log prob par sample (sans somme temporelle)
            entropy: entropie de la dist gaussienne avant tanh
        """
        h = self.net(z)

        mean = self.mean(h)
        std = torch.exp(self.log_std)

        # broadcast std si besoin
        while std.dim() < mean.dim():
            std = std.unsqueeze(0)

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # log prob avec correction tanh
        log_prob = dist.log_prob(raw_action) - torch.log(
            1 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(-1)

        # entropie de la gaussienne (avant tanh)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy


# ============================================================
# Value (stabilité émotionnelle)
# ============================================================
class Value(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, z):
        """
        z: [B, T, Z] ou [B, Z]
        """
        v = self.net(z).squeeze(-1)
        return v


# ============================================================
# MentalAgent V3+ (version robuste 10/10)
# ============================================================
class MentalAgentV3Plus(nn.Module):
    def __init__(self, d_model=256, latent_dim=128, n_emotions=6, action_dim=32):
        super().__init__()

        self.encoder = Encoder(d_model=d_model)
        self.rssm = RSSM(d_model=d_model, latent_dim=latent_dim)
        self.emotion = EmotionHead(latent_dim=latent_dim, n_emotions=n_emotions)
        self.policy = SafePolicy(latent_dim=latent_dim, action_dim=action_dim)
        self.value = Value(latent_dim=latent_dim)

    def forward(self, obs, mask=None, detach_policy=True):
        """
        obs: [B, T, D_model]
        mask: [B, T] (1 = valide, 0 = pad) ou None
        """
        h = self.encoder(obs, src_key_padding_mask=(mask == 0) if mask is not None else None)

        z, prior_mu, prior_logvar, post_mu, post_logvar = self.rssm(h, mask=mask)

        emotions, emotion_logits = self.emotion(z)

        policy_input = z.detach() if detach_policy else z
        action, logp, policy_entropy = self.policy(policy_input)
        value = self.value(z)

        return {
            "z": z,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
            "emotions": emotions,
            "emotion_logits": emotion_logits,
            "action": action,
            "logp": logp,
            "policy_entropy": policy_entropy,
            "value": value,
        }


# ============================================================
# KL Gaussien (q || p) correct
# ============================================================
def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL(q || p) pour gaussiennes diagonales.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl = 0.5 * (
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8) - 1
    )
    return kl


# ============================================================
# Loss mentale (stable + safe)
# ============================================================
def compute_loss(
    agent,
    batch,
    old_logp,
    advantages,
    returns,
    mask=None,
    kl_coef=0.01,
    value_coef=0.5,
    policy_entropy_coef=0.01,
    emotion_entropy_coef=0.01,
):
    """
    batch["obs"]: [B, T, D]
    old_logp: [B, T]
    advantages: [B, T]
    returns: [B, T]
    mask: [B, T] (1 = valide, 0 = pad) ou None
    """

    out = agent(batch["obs"], mask=mask)

    z = out["z"]
    prior_mu = out["prior_mu"]
    prior_logvar = out["prior_logvar"]
    post_mu = out["post_mu"]
    post_logvar = out["post_logvar"]
    emotions = out["emotions"]
    action = out["action"]
    logp = out["logp"]
    value = out["value"]
    policy_entropy = out["policy_entropy"]

    if mask is None:
        mask = torch.ones_like(logp)

    # KL correct (posterior vs prior) : KL(q(z|h) || p(z))
    kl_t = gaussian_kl(post_mu, post_logvar, prior_mu, prior_logvar).sum(-1)  # [B, T]
    kl = (kl_t * mask).sum() / (mask.sum() + 1e-8)

    # PPO
    ratio = torch.exp(logp - old_logp)
    clip = torch.clamp(ratio, 0.8, 1.2)

    policy_loss_t = -torch.min(
        ratio * advantages,
        clip * advantages
    )  # [B, T]

    policy_loss = (policy_loss_t * mask).sum() / (mask.sum() + 1e-8)

    # Value loss
    value_loss_t = F.mse_loss(value, returns, reduction="none")
    value_loss = (value_loss_t * mask).sum() / (mask.sum() + 1e-8)

    # Entropie des émotions (stabilité, éviter émotions trop "rigides")
    emotion_entropy_t = -(emotions * torch.log(emotions + 1e-8)).sum(-1)  # [B, T]
    emotion_entropy = (emotion_entropy_t * mask).sum() / (mask.sum() + 1e-8)

    # Entropie de la policy (exploration safe)
    policy_entropy_mean = (policy_entropy * mask).sum() / (mask.sum() + 1e-8)

    loss = (
        policy_loss
        + value_coef * value_loss
        + kl_coef * kl
        - policy_entropy_coef * policy_entropy_mean
        - emotion_entropy_coef * emotion_entropy
    )

    info = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl": kl.item(),
        "policy_entropy": policy_entropy_mean.item(),
        "emotion_entropy": emotion_entropy.item(),
    }

    return loss, info
