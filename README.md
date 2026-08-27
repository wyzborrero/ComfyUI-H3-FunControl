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

## What it is

A ControlNet for [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), so you
can drive an H3 generation with a control
video instead of only a prompt and a first frame. Feed it depth, canny, pose,
HED or MLSD and the output follows that structure frame by frame, without
re-timing or re-composing the action.

Where the control video comes from is entirely up to you. A depth or pose pass
out of a 3D package, an estimator run over reference footage, something drawn by
hand, or preprocessors inside ComfyUI itself. The node takes an IMAGE batch and
does not care how you made it.

## Why it exists

The weights were already out there and nothing could load them.

[Alibaba PAI](https://huggingface.co/alibaba-pai) released
**[MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)**
with the [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) pipeline, but that release is full-width AdaLN, while the pruned H3 checkpoints
almost everyone actually runs are curve-form. The two do not line up. Using the
official ControlNet meant the 34 GB non-pruned base, and adherence still would
not come right.

**[Kijai](https://github.com/kijai) then re-derived the ControlNet in the
curve-form basis**, using
ComfyUI's own module naming. That is the piece that made a small implementation
possible: with those weights, a control block *is* comfy's `DiTBlock`, and the
AdaLN is comfy's `AdalnProj` unmodified.

What was still missing was anything that consumed them. We checked ComfyUI core,
KJNodes, Kijai's own README, and
[awesome-minimax-H3](https://github.com/wildminder/awesome-minimax-H3): no
ControlNet node for H3 anywhere. As far as we can tell this is the first one.

So this is a small piece of glue over other people's work, filling a hole rather
than inventing anything. The interesting part is not the code, which is short,
but the list of ways it can be wrong while still producing a perfectly plausible
video. That is what most of this README is.

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
| `minimax_h3_fun_controlnet_union_pruned_bf16.safetensors` ([Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental)) | `[96768, 8]` | yes |

The original release is full-width AdaLN; the pruned H3 checkpoints are
curve-form. Kijai re-derived the ControlNet in the curve-form basis (metadata
`{"minimax_h3_fun_controlnet": "adaln_basis"}`) using ComfyUI's own module
naming, which is what makes this node short. See
[Why this is small](#why-this-is-small). The loader checks, and refuses the
wrong one with a clear message rather than loading garbage.

---

## Example workflows

In [`workflows/`](workflows/), as ordinary ComfyUI graphs. Drag one onto the
canvas.

| file | what it shows | needs |
|---|---|---|
| `01_single_control.json` | the smallest thing that works: prompt, optional first frame, one control video | ComfyUI core + this node |
| `02_depth_plus_pose_reference.json` | two chained controls, a character reference on every frame, and CFG | ComfyUI core + this node |
| `03_optional_solattn_speedup.json` | 02 with sparse attention added, for a speed-up | also [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) and Triton |

**01 and 02 need nothing but ComfyUI and this node.** 03 is kept separate
precisely so the first two stay dependency-free: it adds a third-party node pack
that most people will not have installed, and nothing else depends on it.

All three are laid out in labelled groups, every node is titled, and each
carries notes on the canvas explaining the settings that are easy to get wrong.
You still have to point the loaders at your own models and control videos.

### About workflow 03

Sol-Attn is a training-free sparse attention method. It is a real speed-up here,
but it comes with a trap that costs nothing to avoid and everything to miss:
**`morton` must stay off.** It reorders video tokens, and this ControlNet adds
its contribution at raster-order row offsets, so a reordered stream puts the
control on the wrong tokens. With it on, the control disappears completely and
nothing errors. Trap 8 below has the detail.

Measured on an **RTX 5090**, 90 frames at 1280x704, 28 steps, depth 0.3 with
pose 0.7, same seed throughout:

| | time | control |
|---|---|---|
| no Sol-Attn | 332s | follows the control |
| Sol-Attn, `morton: true` | 240s | **gone** |
| Sol-Attn, `morton: false` | **220s, 1.51x** | follows the control |

Triton compiles its kernels on first use, so the first run is a compile rather
than a measurement. Run it twice before believing a number.

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

### Trap 8: token reordering moves the rows your skip lands on

Sparse-attention backends work with this node, but **not with token reordering
switched on.**

Measured with [Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton) on our
test shot, same seed and settings throughout:

| run | time | control |
|---|---|---|
| no Sol-Attn (baseline) | 332s | follows the control |
| Sol-Attn, `morton: true` | 240s | **gone.** Figure static and centred for the whole clip |
| Sol-Attn, `morton: false` | **220s (1.51x)** | follows the control |

Nothing errored in the broken case. The clip looked like a normal generation
that had simply ignored its control input.

`morton` permutes video tokens into Z-order so each attention block is a compact
3D neighbourhood instead of a 2-row strip. Our skip is added at **raster-order**
rows `[a:b]` read from `mod_segments`, so under a permuted stream the control
lands on the wrong tokens. That is the whole failure.

A skip applied by row offset assumes nobody has reordered the rows. Any backend
that permutes tokens invalidates it.

**Also worth keeping** (though it turned out not to be the cause here): sparse
packs install an `optimized_attention_override` into `transformer_options`, and
comfy's `Attention` forwards `transformer_options` straight to
`optimized_attention`, so the control tower's own `DiTBlock`s inherit it and go
sparse, routed with the base stream's bookkeeping. `_dense_options` in
`nodes.py` strips those keys so the tower stays dense whatever the sampler uses.
We disproved this as the cause of the Morton failure by testing it on its own,
and it did not restore control. It is still the right thing to do, and the
confirmed-good configuration has both it and `morton: false`. We have not
isolated whether `_dense_options` is strictly necessary once Morton is off.

If you are writing another ControlNet for H3: **your blocks are not the model's
blocks, but they receive the model's `transformer_options`, and any backend that
reorders tokens invalidates the row offsets your skip depends on.**

---

## Practical findings

These are measurements on one shot: 90 frames at 1280x704, a single figure
walking, jumping and running across a locked-off frame, controlled from a depth
pass and a DWPose pass of the same action. They are not universal laws. How big
the subject is in frame turns out to matter a lot, so expect different results
at a different scale.

### Pose holds the figure; depth does not

We expected the opposite. A DWPose skeleton is thin lines and depth fills the
whole silhouette, so depth "should" survive downsampling better.

It doesn't. With **depth alone**, the figure drifted in *distance from camera*,
walking diagonally toward the lens instead of laterally, and grew steadily
across the shot. With **pose**, the figure's size stayed constant and matched
the control.

The likely reason is token resolution. H3's video VAE downsamples 16x, then
patches 2x2, so 1280x704 is a token grid of **40 by 22 per latent frame**. A
figure roughly 100px tall occupies about **3 of 22 vertical tokens**. A depth
ramp across three tokens is nearly flat and says almost nothing, while a
skeleton's joint positions stay unambiguous even when coarse.

Practical consequence: **the subject's size in frame is the binding constraint
on control authority.** If you need tight control over a figure, frame it larger
or generate at a higher resolution. That is a decision about the shot, not a
node setting.

### Strength above 1.0 is not the answer

At 1.6 the drift was unchanged and the picture degraded badly, with the same
saturation signature as Trap 3 and Trap 4. Don't reach for it.

### Releasing control early

`end_percent` defaults to 1.0, which means control asserts itself through every
step including the last, where texture forms.

If your control pass has featureless regions the ControlNet keeps insisting
those surfaces are smooth, and suppresses any texture the prompt asks for. Our
test case was a depth pass of a figure on an empty ground plane: the ground is a
smooth gradient carrying no detail, so a prompt asking for drawn texture there
lost every time. Ending control around `0.6` pins structure while the noise is
coarse, and lets the late steps put in detail the control cannot describe.

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
frame. If that frame comes from an untextured source, an ordinary 3D render
without materials for instance, the character comes out untextured too, and no
prompt wording fixes it.

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
- How much sparsifying the base model costs in adherence. Sol-Attn with
  `morton: false` measured 1.51x on our shot with the control still following
  the control, but we have not graded the picture against the dense baseline
  beyond confirming the control survives.
- Only depth and pose have been exercised. Canny, HED and MLSD are in the union
  model's training set but we have not run them.

Issues and PRs welcome, especially from anyone who has the Fun-ControlNet
working another way, or who can improve on the pose-versus-depth finding.

---

## Thanks

This node is a small piece of glue on top of a lot of other people's work. In
order of how much of it they did:

**[MiniMax](https://www.minimax.io/blog/minimax-h3)** for building and openly
releasing **[MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)**, an
omni-modal generator that does video with native stereo audio at up to 2K. None
of this exists without that release, and releasing it open was a choice they did
not have to make.

**[Alibaba PAI](https://huggingface.co/alibaba-pai)** for the Fun-ControlNet
itself: **[MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)**,
a single unified adapter covering canny, depth, HED, MLSD and pose, plus video
inpainting. The architecture this node loads is theirs, and
**[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)** is where we read the
packing order and the pipeline that told us what the 196 columns are. They also
publish the [Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).

**[Kijai](https://github.com/kijai)** twice over. First for
**[MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental)**,
the pruned and quantised H3 checkpoints most people actually run, where the
modulation weights (roughly 40% of the parameters) are replaced by a lookup
table built from a curve fitted to them. Then for re-deriving the Fun-ControlNet
into that same curve-form basis, using ComfyUI's own module naming. That second
piece is the reason this node is a few hundred lines instead of a few thousand:
with those weights a control block simply *is* comfy's `DiTBlock`. Also for
**[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)**,
which we tested against and which produced Trap 8.

**The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team** for
`comfy/ldm/minimax/`, which is clear enough to read as a specification, and for
`transformer_options["patches_replace"]`, a genuinely well-designed extension
point that let this be written without monkey-patching a single thing.

**[NVlabs](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn)**
for Sol-Attn itself ([paper](https://arxiv.org/abs/2607.24027)).

**[wildminder](https://github.com/wildminder/awesome-minimax-H3)** for
awesome-minimax-H3, which is how we established that nobody had done this yet.

If we have misattributed anything, open an issue and we will fix it.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
