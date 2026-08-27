# ComfyUI-H3-FunControl

A ComfyUI ControlNet for **MiniMax-H3**: depth, canny, pose, HED or MLSD video
driving H3's generation, frame by frame.

As far as we can tell this is the first working ComfyUI implementation. We
checked ComfyUI core, KJNodes, Kijai's README, and
[awesome-minimax-H3](https://github.com/wildminder/awesome-minimax-H3), and none
of them wire up the Fun-ControlNet. Kijai's re-derived weights exist, but
nothing consumes them.

This README documents **what went wrong and how we fixed it**, not just how to
use it. Most of the bugs below failed *silently*. They produced a plausible
video rather than an error, so they are worth writing down.

---

## What it's for

Unreal Engine blocking to AI video, frame-locked. You render a control pass
(depth, pose) from your 3D blocking, and H3 renders your look on top of it
without re-timing or re-framing the action. It is WYSIWYG: what you block is
what you get.

That is the trade against a model like Seedance, which produces a better picture
but reframes and re-times freely. Frame-lock is the whole point here.

---

## Install

```
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-H3-FunControl
```

No dependencies beyond ComfyUI itself. The node is built out of ComfyUI's own H3
modules.

**Restart ComfyUI after any edit.** Custom nodes are imported once at startup,
so an edited file is inert until then. This cost us a whole experiment (see
[Trap 1](#trap-1-a-node-edit-is-inert-until-restart)).

### Weights

You need a **curve-form** ControlNet checkpoint, matching the pruned H3 bases
everyone actually runs:

| | `adaln_proj.linear` | works here |
|---|---|---|
| `MiniMax-H3-Fun-Controlnet-Union.safetensors` (original) | `[96768, 2688]` | no |
| `minimax_h3_fun_controlnet_union_pruned_bf16.safetensors` (Kijai) | `[96768, 8]` | yes |

The original release is full-width AdaLN; the pruned H3 checkpoints are
curve-form. Kijai re-derived the ControlNet in the curve-form basis (metadata
`{"minimax_h3_fun_controlnet": "adaln_basis"}`) using ComfyUI's own module
naming, which is what makes this node short. See
[Why this is small](#why-this-is-small). The loader checks, and refuses the
wrong one with a clear message rather than loading garbage.

---

## Nodes

### `H3 Fun ControlNet Loader`
Loads the checkpoint from `models/controlnet`.

### `H3 Fun ControlNet Apply`

| input | notes |
|---|---|
| `model` | MODEL in, MODEL out. **It patches the model, not the conditioning.** See [Composing](#composing-with-other-nodes). |
| `control_net` | from the loader |
| `vae` | the H3 **video** VAE |
| `control_video` | IMAGE batch, same length, width and height as the generation |
| `strength` | scales every skip. **When chaining, this is a shared budget.** See [Trap 3](#trap-3-chained-towers-sum). |
| `start_percent` / `end_percent` | the slice of the sampling schedule where control applies. Not cosmetic. See [Releasing control early](#releasing-control-early). |

---

## The architecture, all sourced

```
control_in_dim  49 channels -> 196 packed
                [ control latent 96 | mask 4 | masked latent 96 ]
                each patchified (1,2,2) over 24 VAE channels
blocks          5, injected at base layers 0 / 10 / 20 / 30 / 40 of 50
entry           before_proj on block 0 ONLY. It is the stream entry,
                not a per-block adapter.
skip            after_proj on every block, zero-gated
```

Sources: the [model card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union),
[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
(`videox_fun/pipeline/pipeline_minimax_h3_control.py` for the `torch.cat`
order), and ComfyUI's own `comfy/ldm/minimax/model.py`.

The checkpoint is the specification: block 0 carries 14 tensors, blocks 1 to 4
carry 12. That difference *is* `before_proj`, and assuming otherwise cost one
failed load.

### Why this is small

Because it implements almost nothing:

| piece | what it actually is |
|---|---|
| a control block | `comfy.ldm.minimax.model.DiTBlock` plus `before_proj` / `after_proj` |
| AdaLN curve form | comfy's `AdalnProj`, unmodified |
| the injection hook | `transformer_options["patches_replace"]["dit"]`, a supported extension point |

All ten DiTBlock tensors load `strict=True`. Nothing is monkey-patched and
nothing in ComfyUI is modified.

---

## The shapes, which are not negotiable

H3 is **batch-free and single-stream**. Its forward builds

```python
h = torch.empty(layout.seq_len, self.hidden_size)   # 2-D. No batch axis.
```

over a packed sequence of `[text | cond | audio | video]`, and every block
modulates it through `_mod_scale_shift` / `_mod_gate`, which index
`mod_segments` (absolute offsets running to `seq_len`) into whatever tensor they
are handed.

Two consequences, both load-bearing.

**1. Control rows are 2-D `[tokens, 196]`.** A leading batch dim of 1 does not
broadcast away. `Attention` reads `s = x.shape[0]` as the token count, so a
`[1, 23760, 5376]` stream makes `s = 1` and the qkv view collapses with:

```
shape '[1, 56, 128]' is invalid for input of size 170311680
```

**2. The control stream is a full-length sibling of the packed stream**, seeded
from it, not a video-only strip. VideoX-Fun's tower does
`control = before_proj(proj_in(rows)) + hidden_states` once, then runs five
blocks off that. A video-length control stream would make `_mod_gate` read
`mod_segments`' offsets off the end of the tensor.

The control *rows* still cover only the video segment, so they are added into
that span of the seed.

### The video-segment slice

The skip belongs to the **video rows only**. Adding it across the whole stream
would corrupt audio silently: no error, just wrong sound.

`transformer_options` carries no layout, so the span is read from
`mod_segments[-1]`. `PackedLayout` appends `("audio", ...)` then
`("video", n_video)`, so video is always last. Text can split into several tag
runs, so the count varies at the front, but the tail does not.

We verified this offline before trusting it: a synthetic packed stream, five
blocks, then asserting the non-video rows changed by exactly zero.

---

## Traps

Every one of these produced a plausible result rather than an error.

### Trap 1: a node edit is inert until restart

We added tower chaining, ran a "depth + pose" test, and got a result that looked
reasonable. It was **bit-identical to pose alone**, because ComfyUI was still
running the pre-edit module and the second Apply had silently overwritten the
first.

A thirty-second PSNR check caught it:

```bash
ffmpeg -i a.mp4 -i b.mp4 -lavfi "[0:v][1:v]psnr" -f null -
# average:inf  ->  identical frames. One of your runs did not happen.
```

**"They look nearly the same" and "one of them silently didn't run" are
indistinguishable by eye.** Measure.

### Trap 2: ComfyUI's execution cache looks like a speed-up

We timed a run, changed only the output filename prefix, and the second run
finished in 10 seconds. That was not speed. Nothing upstream of `SaveVideo` had
changed, so ComfyUI returned cached node outputs. It even wrote a new file with
a different md5, because the container is re-encoded.

**Change the seed to force execution.** PSNR `inf` again tells you the frames
are identical.

### Trap 3: chained towers *sum*

`set_model_patch_replace` assigns by key:

```python
to["patches_replace"][name][block] = patch   # overwrites
```

So a second `H3FunControlApply` used to drop the first with no error. This node
now reads the existing patch and calls it from inside its own hook, so towers
compose the way ComfyUI chains ControlNets: each runs its own control stream and
adds its own zero-gated skip.

**Their skips sum.** Depth at 1.0 plus pose at 1.0 behaves like a single tower
at 2.0 and saturates: washed out, figure smeared, detail gone. Treat strength as
a budget across the chain. Depth 0.3 with pose 0.7 works well; 1.0 with 1.0 does
not.

### Trap 4: the no-inpaint mask value

The packing is `[control latent 96 | mask 4 | masked latent 96]`. With no
inpainting the masked latent is zeros, and the mask value decides what that
means.

| mask | result |
|---|---|
| `1` | at strength 1.0 the picture **collapses** to a silhouette on a flat neutral field, style gone by frame 1. Backing off to 0.3 restores the style and loses the lock. No value in 0.3 to 1.0 gives both. |
| `0` | style and frame-lock together at strength 1.0. |

`1` does not mean "generate here". It means "this row's masked latent is valid,
reproduce it", and since the masked latent is zeros, that instructs the model to
reproduce black everywhere. The depth structure was the only thing left, which
is exactly what the collapsed takes looked like.

It is `NO_INPAINT_MASK = 0.0` in `control.py`, overridable by the
`H3_NO_INPAINT_MASK` env var so the A/B can be re-run without editing code.

### Trap 5: inference-only, said out loud

`nn.Parameter` defaults to `requires_grad=True`, and ComfyUI's own weights are
frozen by a loader path this tower does not go through. Leaving it on made every
control activation carry `requires_grad`, and H3's attention takes a fused
**in-place** RoPE kernel which refuses those outright:

```
in-place RoPE operations are inference-only and do not support autograd
```

Fixed with `model.eval(); model.requires_grad_(False)` at load.

### Trap 6: dtypes, per role

Casting the whole tower to the sampling dtype breaks AdaLN, because `DiTBlock`
keeps it float32: `t_emb` and the curve table are float32. Moving device only
leaves the checkpoint's own mix (norms and `control_proj_in` stored F32,
everything else BF16), which still mismatches inside the matmuls.

So the node states it outright rather than inferring it from what the file
happened to contain: **compute weights take the sampler's dtype, adaln stays
float32.**

### Trap 7: `_mod_gate` mutates in place

```python
x[a:b].addcmul_(other[a:b], ...)   # returns the SAME tensor
```

Once the original block has run, `args["img"]` *is* the output and the tower's
entry state is gone. The seed must be cloned before the base block runs, at the
entry layer only.

### Trap 8: sparse-attention backends silently capture the control tower

Sparse-attention packs install an `optimized_attention_override` into
`transformer_options`, alongside routing state that describes the **base** packed
stream: the running block index, the video span, a compose table. comfy's
`Attention` passes `transformer_options` straight through to
`optimized_attention`, so if you hand the control tower the sampler's dict, its
blocks go sparse too, routed with another tensor's bookkeeping.

Measured with [Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton) on the
test shot: **1.39x faster and the control was gone entirely.** Not degraded,
gone. The figure stood centred and near-motionless for the whole clip while the
same seed without Sol-Attn followed the blocking exactly. Nothing errored.

This node now strips those keys before calling its own blocks (`_dense_options`
in `nodes.py`), so the tower stays dense whatever the sampler is using. The
tower is 5 blocks against the base model's 50, so running it dense costs little
and keeps the control exact.

If you write another ControlNet for H3, this is the trap that will get you: your
blocks are not the model's blocks, but they receive the model's
`transformer_options`.

---

## Practical findings

These are measurements on one shot (90 frames, 1280x704, a figure crossing a
wide locked-off frame), not universal laws. Your mileage will vary with how big
the subject is in frame, which turns out to matter a lot.

### Pose holds the figure; depth does not

We expected the opposite. A DWPose skeleton is thin lines and depth fills the
whole silhouette, so depth "should" survive downsampling better.

It doesn't. With **depth alone**, the figure drifted in *distance from camera*,
walking diagonally toward the lens instead of laterally, and grew steadily
across the shot. With **pose**, her size stayed constant and matched the
blocking.

The likely reason is token resolution. H3's video VAE downsamples 16x, then
patches 2x2, so 1280x704 is a token grid of **40 by 22 per latent frame**. A
figure roughly 100px tall occupies about **3 of 22 vertical tokens**. A depth
ramp across three tokens is nearly flat and says almost nothing, while a
skeleton's joint positions stay unambiguous even when coarse.

Practical consequence: **the subject's size in frame is the binding constraint
on control authority.** If you need tight control, block the shot closer. That
is a camera decision, not a node setting.

### Strength above 1.0 is not the answer

At 1.6 the drift was unchanged and the picture degraded badly, with the same
saturation signature as Trap 3 and Trap 4. Don't reach for it.

### Releasing control early

`end_percent` defaults to 1.0, which means control asserts itself through every
step including the last, where texture forms.

If your control pass has featureless regions (ours had a bare Unreal ground
plane, so the depth pass's ground is a smooth gradient) the ControlNet keeps
insisting that surface is smooth, and suppresses any texture the prompt asks
for. Ending control around `0.6` pins structure while the noise is coarse, and
lets the late steps put in detail the control cannot describe.

### CFG

Community guidance for H3 is "keep guidance at 1, it's baked into the weights".
For a *detailed style brief* at 28 steps we measured the opposite: CFG 4.0
produced markedly better adherence to the described technique.

Note also that `BasicGuider` has **no negative input at all**, so every negative
prompt is inert until you switch to `CFGGuider`. Cost is about 1.9x, since the
tower runs twice per step.

---

## Composing with other nodes

`H3FunControlApply` takes a MODEL and returns a MODEL. It never touches the
conditioning branch, so conditioning can come from anywhere:

```
UNETLoader --> H3FunControlApply --> MiniMaxH3SigmaShift --> Guider --> Sampler
                     ^ control video                          ^
               H3FunControlLoader                  any H3 conditioning node
```

This matters. `MiniMaxH3ImageToVideo` only accepts `first_frame` /`last_frame`,
which means a character can only reach the model baked into a styled first
frame. If your first frame is derived from a featureless 3D blocking render, the
character comes out featureless too, and no prompt wording fixes it.

Swapping in **`MiniMaxH3ReferenceToVideo`** puts a character sheet in as
`ref_image_0`, present on every frame, while the ControlNet still drives the
motion. That single change is what took our output from a nude mannequin to the
actual character.

Chain a second `H3FunControlApply` for multi-control, remembering Trap 3.

---

## Status

Working and in use. Verified: loads `strict=True`; `strength=0.0` is a true
bypass; audio survives (stereo, correct duration) with the video-segment slice
in place; the control token count is checked against the video segment and
raises with both numbers if they diverge.

Untested or unknown:

- The ControlNet was trained against the **fl2va** base, and we mostly run it on
  **ref2va**. Same architecture, loads and runs, adherence appears to hold, but
  it is not the trained pairing.
- Sparse attention on the BASE model. The control tower is kept dense (Trap 8),
  but we have not yet checked how much sparsifying the base costs in adherence,
  only that the tower must be excluded. Sol-Attn measured 1.39x on our shot.
- Only depth and pose have been exercised. Canny, HED and MLSD are in the union
  model's training set but we have not run them.

Issues and PRs welcome, especially from anyone who has the Fun-ControlNet
working another way, or who can improve on the pose-versus-depth finding.

---

## Credits

- **MiniMax** for H3, **alibaba-pai / VideoX-Fun** for the Fun-ControlNet.
- **[Kijai](https://github.com/kijai)** for the curve-form re-derivation, without
  which this node would need the 34 GB non-pruned base and would be a great deal
  longer.
- The ComfyUI team for `patches_replace`, a genuinely well-designed extension
  point.

---

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
