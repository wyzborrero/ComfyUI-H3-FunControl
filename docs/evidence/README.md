# Evidence for the upstream sparse-attention PR

`comfyui-0.35.2-block-sparse-attention-order.png` is a filmstrip of ten clips
against the pose skeleton that drove them, frames 0 / 25 / 51 / 64 / 77, seed
777, ComfyUI v0.35.2, RTX 5090, 90 frames at 1280x704, 28 steps.

Rows, for both ComfyUI's native `MiniMaxH3FunControlNetApply` and this node:
baseline (no sparse attention), `BlockSparseAttention` applied before the
ControlNet, applied after the ControlNet on unfixed core, and applied after
with the fix. On unfixed core the "after" rows lose the control completely:
a centred figure running in place with the same framing in every frame.

This branch exists only to give the pull request a stable URL for the image.
It is not merged into the main line and has no code.
