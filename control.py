"""MiniMax-H3 Fun ControlNet, built on ComfyUI's own H3 blocks.

WHY THIS IS SHORT
-----------------
The first attempt at this port reimplemented the H3 transformer tower by hand,
because the official Fun-ControlNet weights are full-width AdaLN
(adaln_proj.linear = [96768, 2688]) while the pruned checkpoints everyone
actually runs are curve-form ([96768, 8]). Nothing lined up, the mismatch forced
the 34 GB non-pruned base, and adherence never came right at strength 1.0.

Kijai then published a re-derived ControlNet in the curve-form basis
(metadata: {"minimax_h3_fun_controlnet": "adaln_basis"}), using ComfyUI's own
module naming -- qkv_proj, out_proj, q_norm/k_norm, mlp.fc1/fc2. Measured
against comfy.ldm.minimax.model.DiTBlock, all ten tensors match by name and
shape and load with strict=True.

So this file implements almost nothing:

    a control block          = comfy's DiTBlock + before_proj/after_proj
    AdaLN curve form         = comfy's AdalnProj, unmodified
    the injection hook       = transformer_options["patches_replace"]["dit"]

THE ARCHITECTURE, ALL SOURCED
-----------------------------
    control_in_dim   49 channels -> 196 packed        (model card + VideoX-Fun)
                     96 control latent | 4 mask | 96 masked latent
                     each patchified (1,2,2) over 24 VAE channels
    blocks           5, injected at base layers 0/10/20/30/40 of 50
    skip             zero-gated: after_proj is zero-init in training, so an
                     untrained/zeroed controlnet is a no-op rather than noise

THE SHAPES, WHICH ARE NOT NEGOTIABLE
------------------------------------
H3 is batch-free: the forward builds its stream as
torch.empty(layout.seq_len, hidden_size) and Attention reads s = x.shape[0] as
the token count. So control rows are 2-D, [tokens, 196] -- a leading batch dim
of 1 does not broadcast away, it makes s = 1 and the qkv view collapses.

And the control STREAM (built in nodes.py) is full seq_len, seeded from the
base stream, not a video-only strip: DiTBlock modulates through
_mod_scale_shift / _mod_gate, which index mod_segments -- absolute offsets over
[text | cond | audio | video] -- into whatever tensor they are handed.

WHAT IS STILL UNVERIFIED
------------------------
H3 packs video and audio rows into ONE token stream (see PackedLayout in
comfy/ldm/minimax/model.py). The control skip must be added to the VIDEO
segment only. Adding it across the whole stream would corrupt audio silently --
no error, just wrong sound -- so the segment slice in nodes.py is the part to
scrutinise first if results are strange.
"""

import os

import torch
from torch import nn

import comfy.ops
from comfy.ldm.minimax.model import DiTBlock

# From MiniMaxH3Model.__init__ defaults. These are not guesses: hidden 5376,
# 56 heads x 128 = 7168 (so qkv is 21504), ffn 14336, and t_dim 8 for the
# curve-form basis that the pruned checkpoints use.
H3 = dict(hidden=5376, heads=56, head_dim=128, ffn=14336, t_dim=8,
          eps=1e-5, qk_eps=1e-5, apply_silu=False)

# 24 latent channels x prod(patch_size) for the control latent, the same again
# for the masked latent, and 1 mask channel x prod(patch_size).
LATENT_DIM, PATCH = 24, (1, 2, 2)
PATCH_PROD = PATCH[0] * PATCH[1] * PATCH[2]
CONTROL_IN = (LATENT_DIM + LATENT_DIM + 1) * PATCH_PROD      # 49 * 4 = 196

DEFAULT_LAYERS = (0, 10, 20, 30, 40)

# The mask value to send when nothing is being inpainted -- every shot in this
# pipeline. MEASURED, not reasoned about, because the wrong value does not
# raise, it degrades:
#
#   mask = 1   strength 1.0 locks to the control but the picture collapses to
#              a silhouette on a flat neutral field, style gone by frame 1.
#              Backing off to 0.3 restores the style and loses the lock. No
#              value in 0.3-1.0 gives both.
#   mask = 0   style and frame-lock together at strength 1.0. Same seed, same
#              steps, same plate; the only change was this number.
#
# So 1 does not mean "generate here", it means "this row's masked_latent is
# valid, reproduce it" -- and since the masked latent is zeros, that instructed
# the model to reproduce black everywhere. The depth structure was the only
# thing left, which is exactly what the collapsed takes looked like.
NO_INPAINT_MASK = float(os.environ.get("H3_NO_INPAINT_MASK", "0.0"))


class ControlBlock(nn.Module):
    """One control block: a stock DiTBlock plus its skip projection.

    ONLY THE FIRST BLOCK HAS before_proj, and the checkpoint says so: block 0
    carries 14 tensors, blocks 1-4 carry 12. before_proj is the ENTRY into the
    control stream, not a per-block adapter -- the stream is projected in once
    and then flows through all five blocks. after_proj is the zero-gated skip
    each block contributes back to the main branch.

    Assuming before_proj on every block cost one failed load, which is the
    cheap way to find out: the checkpoint is the specification.
    """

    def __init__(self, first=False, dtype=None, device=None,
                 operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.before_proj = (
            operations.Linear(H3["hidden"], H3["hidden"], bias=True,
                              dtype=dtype, device=device) if first else None)
        self.block = DiTBlock(**H3, adaln_dtype=torch.float32, dtype=dtype,
                              device=device, operations=operations)
        self.after_proj = operations.Linear(H3["hidden"], H3["hidden"], bias=True,
                                            dtype=dtype, device=device)


class H3FunControlNet(nn.Module):
    """The 5-block control tower plus its input projection."""

    def __init__(self, num_blocks=5, dtype=None, device=None,
                 operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.control_proj_in = operations.Linear(CONTROL_IN, H3["hidden"], bias=True,
                                                 dtype=dtype, device=device)
        self.control_blocks = nn.ModuleList([
            ControlBlock(first=(i == 0), dtype=dtype, device=device,
                         operations=operations)
            for i in range(num_blocks)])

    @classmethod
    def from_state_dict(cls, sd, dtype=torch.bfloat16, device="cpu"):
        """Load a checkpoint, mapping its flat names onto this module tree.

        The checkpoint stores control_blocks.N.<tensor> with the DiTBlock's own
        tensors at the top level of the block. This module nests them under
        .block so the stock class can own them, so the only remapping is
        inserting that one path component.
        """
        count = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("control_blocks."))
        model = cls(num_blocks=count, dtype=dtype, device=device)

        remapped = {}
        for key, value in sd.items():
            if key.startswith("control_blocks."):
                _, index, rest = key.split(".", 2)
                if rest.startswith(("before_proj", "after_proj")):
                    remapped[f"control_blocks.{index}.{rest}"] = value
                else:
                    remapped[f"control_blocks.{index}.block.{rest}"] = value
            else:
                remapped[key] = value

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"controlnet does not match this implementation.\n"
                f"  missing: {sorted(missing)[:6]}\n"
                f"  unexpected: {sorted(unexpected)[:6]}")

        # INFERENCE-ONLY, SAID OUT LOUD.
        # nn.Parameter defaults to requires_grad=True, and ComfyUI's own
        # weights get marked otherwise by the loader path this tower does not
        # go through. Leaving it on made every control activation carry
        # requires_grad, and H3's attention takes the fused IN-PLACE rope
        # kernel, which refuses outright: "in-place RoPE operations are
        # inference-only and do not support autograd". The base blocks never
        # hit it because their weights are already frozen.
        model.eval()
        model.requires_grad_(False)
        return model


def pack_control_latent(control_latent, mask=None, masked_latent=None):
    """Build the 196-wide control rows the ControlNet's input projection wants.

    Order is not a choice -- it is VideoX-Fun's, from
    pipeline_minimax_h3_control.py:

        control_rows = cat([control, cat([mask_condition, masked_latent])])

    which is [control 96 | mask 4 | masked 96]. Patchify happens BEFORE the
    concatenation, so each group is contiguous rather than interleaved.

    With no inpainting, the mask is all-visible and the masked latent is zeros:
    the ControlNet still expects all 196 columns, and feeding it 96 would fail
    at the projection rather than degrade.

    The result is 2-D, [tokens, 196]. H3 has no batch axis anywhere in the
    tower -- the forward builds its stream as
    ``torch.empty(layout.seq_len, hidden_size)`` and Attention reads
    ``s = x.shape[0]`` as the token count. A leading batch dim of 1 therefore
    does not broadcast harmlessly; it makes s = 1 and the qkv view collapses
    with "shape [1, 56, 128] is invalid for input of size 170311680".
    """
    rows = patchify(control_latent)                       # [T, 96]
    t, _ = rows.shape
    if mask is None:
        mask_rows = torch.full((t, 1 * PATCH_PROD), float(NO_INPAINT_MASK),
                               dtype=rows.dtype, device=rows.device)
    else:
        mask_rows = patchify(mask, channels=1)
    if masked_latent is None:
        masked_rows = torch.zeros_like(rows)
    else:
        masked_rows = patchify(masked_latent)
    return torch.cat([rows, mask_rows, masked_rows], dim=-1)


def patchify(latent, channels=LATENT_DIM):
    """[1, C, T, H, W] -> [T*(H/2)*(W/2), C*prod(patch)] -- comfy's layout.

    Batch-free on purpose, matching patchify_video in comfy/ldm/minimax:
    the tower is 2-D throughout. A batch greater than 1 has nowhere to go in
    this architecture, so say so rather than folding it into the token axis.
    """
    b, c, t, h, w = latent.shape
    if b != 1:
        raise ValueError(f"H3 control latents are single-batch; got batch {b}")
    pt, ph, pw = PATCH
    latent = latent.view(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    latent = latent.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return latent.view((t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)
