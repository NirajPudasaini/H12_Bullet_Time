"""SIM-* future observations from the perfect-WM environment's Future_Projectile.

SIM-RAW: future ToF image, SIM-MINDIST: per-sensor minimum of it, SIM-LATENT: PWM encoder latent of it.
They replace the PWM future (dynamics rollout) and the optional contact flag comes from the sim propagation.
"""

import torch

SIM_FUTURE_OBS_TYPES = ("sim-raw", "sim-mindist", "sim-latent")


def sim_future_features(world_model, kind: str, frame: torch.Tensor, contact: torch.Tensor | None = None):
    """(B, D) future observation from a future ToF frame (B, 1, H, W, S); contact (B,) is appended when given."""
    if kind == "sim-raw":
        out = frame.flatten(1)
    elif kind == "sim-mindist":
        out = frame.amin(dim=(2, 3)).flatten(1)
    elif kind == "sim-latent":
        out = world_model._as_obs("latent", latent=world_model._encode(world_model._validate_frame(frame)))
    else:
        raise ValueError(f"Unknown SIM future type {kind!r}; expected one of {SIM_FUTURE_OBS_TYPES}")
    out = out.float()
    return out if contact is None else torch.cat((out, contact.float()[:, None]), dim=1)


def require_future_projectile(env):
    if not hasattr(env, "_future_projectile_contact"):
        raise RuntimeError("SIM-* future observations need --task Template-H12-Survive-Time-Perfect-WM")
