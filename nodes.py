"""ComfyUI nodes for the MiniMax-H3 Fun ControlNet.

    H3FunControlLoader   loads the controlnet from models/controlnet
    H3FunControlApply    encodes a control video and patches it into the model

HOW THE INJECTION WORKS
-----------------------
comfy/ldm/minimax/model.py exposes a supported extension point:

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace  = patches_replace.get("dit", {})
    if ("double_block", i) in blocks_replace:
        h = blocks_replace[("double_block", i)]({...}, {"original_block": ...})["img"]

So for each of the five injection layers we register a callback that runs the
ORIGINAL block first, then adds that control block's zero-gated skip. Nothing
is monkey-patched and nothing in ComfyUI is modified.

THE SHAPE THE TOWER INSISTS ON
------------------------------
H3 is batch-free and single-stream. The forward builds

    h = torch.empty(layout.seq_len, hidden_size)

over [text | cond | audio | video] and every block modulates it through
_mod_scale_shift / _mod_gate, which index `mod_segments` -- ABSOLUTE offsets
running to seq_len -- into whatever tensor they are given. Two consequences,
both load-bearing:

  * control rows are 2-D, [tokens, hidden]. A leading batch dim of 1 does not
    broadcast away; Attention reads s = x.shape[0] as the token count.
  * the control stream is a FULL-LENGTH sibling of the packed stream, seeded
    from it (VideoX-Fun: control = before_proj(proj_in(rows)) + hidden_states).
    A video-only strip would read mod_segments' offsets off the end.

THE PART THAT NEEDS WATCHING
----------------------------
The stream is full length, but the SKIP belongs to the video rows only. This
reads the video segment out of the layout the model itself computed rather than
assuming an offset -- assuming would corrupt audio silently, which is the worst
kind of wrong.
"""

import torch

import comfy.model_management
import comfy.utils
import folder_paths

from .control import H3FunControlNet, DEFAULT_LAYERS, pack_control_latent


class H3FunControlLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "control_net_name": (folder_paths.get_filename_list("controlnet"),),
        }}

    RETURN_TYPES = ("H3_FUN_CONTROL",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load"
    CATEGORY = "previz/h3"
    DESCRIPTION = ("Load a MiniMax-H3 Fun ControlNet. Needs a CURVE-FORM "
                   "checkpoint (adaln_proj.linear = [96768, 8]) matching the "
                   "pruned H3 bases -- the original full-width release "
                   "([96768, 2688]) will not load.")

    def load(self, control_net_name):
        path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)

        first = sd.get("control_blocks.0.adaln_proj.linear.weight")
        if first is not None and first.shape[-1] != 8:
            raise RuntimeError(
                f"'{control_net_name}' has full-width AdaLN "
                f"(t_dim={first.shape[-1]}), which does not match the "
                f"curve-form pruned H3 checkpoints this node targets. Use a "
                f"pruned/adaln_basis controlnet instead.")

        net = H3FunControlNet.from_state_dict(
            sd, dtype=comfy.model_management.unet_dtype(),
            device=comfy.model_management.unet_offload_device())
        return (net,)


class H3FunControlApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "control_net": ("H3_FUN_CONTROL",),
            "vae": ("VAE",),
            "control_video": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                   "step": 0.05,
                                   "tooltip": "Scales every control skip. 0 is "
                                              "a true bypass."}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                        "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "previz/h3"
    DESCRIPTION = ("Condition MiniMax-H3 on a control video (depth, canny, "
                   "pose, HED or MLSD). The video is VAE-encoded, packed to "
                   "196 columns and injected at base layers 0/10/20/30/40.")

    def apply(self, model, control_net, vae, control_video, strength,
              start_percent, end_percent):
        patched = model.clone()

        # Encode once, here, rather than per sampling step.
        latent = vae.encode(control_video[:, :, :, :3])
        if latent.ndim == 4:                       # [B,C,H,W] -> [B,C,1,H,W]
            latent = latent.unsqueeze(2)
        rows = pack_control_latent(latent)         # [control tokens, 196]

        sigmas = getattr(model.model, "model_sampling", None)
        start_t = sigmas.percent_to_sigma(start_percent) if sigmas else 1.0
        end_t = sigmas.percent_to_sigma(end_percent) if sigmas else 0.0

        entry_layer = DEFAULT_LAYERS[0]
        state = {"rows": rows, "stream": None, "key": None}

        def in_window(options):
            sigma = options.get("sigmas")
            if sigma is None:
                return True
            return end_t <= float(sigma[0]) <= start_t

        def make_resident(net, device, dtype):
            """Put the tower where the sampling is, once, with per-role dtypes.

            The loader leaves it on the offload device so it does not hold
            4.2 GB of VRAM while idle, but the hook runs on the compute device
            -- loading it and never moving it produced "mat1 is on cuda:0,
            different from other tensors on cpu" on the first sampler step.

            Lazily here rather than eagerly in apply(): the tower only needs to
            be resident while this model is actually sampling, so a graph that
            never reaches the sampler never pays for it.

            Two attempts got the dtypes wrong. Casting the whole tower to the
            sampling dtype broke adaln (DiTBlock keeps it float32 because t_emb
            and the curve table are float32). Moving device only left the
            checkpoint's own mix -- its norms and control_proj_in stored F32,
            everything else BF16 -- which still mismatched inside the matmuls.
            So state it outright rather than inferring it from what the file
            happened to contain: compute weights take the sampler's dtype,
            adaln stays float32.
            """
            if state["key"] == (device, dtype):
                return
            net.to(device=device)
            for name, module in net.named_modules():
                if getattr(module, "weight", None) is None:
                    continue
                module.to(torch.float32 if "adaln" in name else dtype)
            state["key"] = (device, dtype)

        def enter(net, seed, a, b):
            """Open the control stream: projected control added into the base.

            before_proj lives on block 0 alone because it is the ENTRY, not a
            per-block adapter -- VideoX-Fun's tower does

                control = before_proj(control_proj_in(rows)) + hidden_states

            once, then runs its five blocks off that. The control rows cover
            the video segment only, so they land in that span of the seed while
            the rest of the stream carries through unchanged.
            """
            proj = net.control_proj_in
            control = proj(state["rows"].to(device=seed.device,
                                            dtype=proj.weight.dtype))
            control = net.control_blocks[0].before_proj(control)
            if control.shape[0] != b - a:
                raise RuntimeError(
                    "H3FunControl: the control video does not match the "
                    "generation. It packs to %d tokens but the video segment "
                    "of the stream is %d. The control pass must have the same "
                    "length, width and height as the shot -- check "
                    "LENGTH/WIDTH/HEIGHT against control.mp4."
                    % (control.shape[0], b - a))
            seed[a:b] += control.to(seed.dtype)
            return seed

        def make_hook(index, inner):
            # `inner` is whatever patch already owned this layer -- another
            # H3FunControlApply upstream, chained the way ComfyUI chains
            # multiple ControlNets. Calling it instead of the raw block lets
            # the towers COMPOSE: each runs its own control stream and adds
            # its own zero-gated skip to the video rows, so depth and pose sum
            # rather than the second one silently replacing the first.
            def hook(args, extra):
                options = args.get("transformer_options", {}) or {}
                active = strength != 0.0 and in_window(options)
                span = _video_span(args.get("mod_segments")) if active else None

                # SEED THE STREAM BEFORE THE BASE BLOCK RUNS.
                # DiTBlock's _mod_gate does x[a:b].addcmul_(...) -- it mutates
                # its input and returns that same tensor, so once the original
                # block has run, args["img"] IS the output and the tower's
                # entry state is gone. Clone it here, at the entry layer only.
                seed = (args["img"].clone()
                        if span is not None and index == entry_layer else None)

                out = (inner(args, extra) if inner is not None
                       else extra["original_block"](args))
                img = out["img"]
                if span is None:
                    return out
                a, b = span

                net = control_net
                make_resident(net, img.device, img.dtype)

                if seed is not None:
                    state["stream"] = enter(net, seed, a, b)
                control = state["stream"]
                if control is None:
                    return out

                position = DEFAULT_LAYERS.index(index)
                cb = net.control_blocks[position]
                try:
                    control = cb.block(control, args["t_emb"],
                                       args["mod_segments"],
                                       args["rope_freqs"],
                                       transformer_options=options)
                except RuntimeError as exc:
                    # A bare "mat1 and mat2 must have the same dtype"
                    # names no tensor. Say which one.
                    raise RuntimeError(
                        "H3FunControl block %d (base layer %d) failed: %s\n"
                        "  control  %s %s\n"
                        "  t_emb    %s %s\n"
                        "  qkv_proj %s\n"
                        "  adaln    %s\n"
                        "  img      %s %s\n"
                        "  video    rows %d..%d" % (
                            position, index, exc,
                            tuple(control.shape), control.dtype,
                            tuple(args["t_emb"].shape), args["t_emb"].dtype,
                            cb.block.attn.qkv_proj.weight.dtype,
                            cb.block.adaln_proj.linear.weight.dtype,
                            tuple(img.shape), img.dtype,
                            a, b)) from exc
                state["stream"] = control

                # VIDEO ROWS ONLY. after_proj is row-wise, so projecting the
                # span rather than the whole stream is identical arithmetic and
                # skips a 5376x5376 matmul over the text and audio rows.
                skip = cb.after_proj(control[a:b]) * strength
                img[a:b] += skip.to(img.dtype)
                return {"img": img}
            return hook

        # READ BEFORE OVERWRITING.
        # set_model_options_patch_replace assigns to
        # patches_replace["dit"][("double_block", n)] outright, so registering a
        # second control tower on the same layers would drop the first without
        # any error. Capture what is there and chain it.
        existing = (patched.model_options.get("transformer_options", {})
                    .get("patches_replace", {}).get("dit", {}))
        for layer in DEFAULT_LAYERS:
            inner = existing.get(("double_block", layer))
            patched.set_model_patch_replace(make_hook(layer, inner), "dit",
                                            "double_block", layer)
        return (patched,)


def _video_span(mod_segments):
    """(start, stop) of the video rows in H3's packed token stream.

    NOT read from transformer_options -- the H3 forward puts nothing there.
    Checking that was worth it: the first version of this looked up a
    "minimax_layout" key that does not exist, so it would have returned None on
    every call and the controlnet would have done nothing at all, quietly.

    The real source is mod_segments, which the block hook already receives.
    PackedLayout appends its segments in a fixed order and video is ALWAYS
    last (model.py: segments.append(("audio", ...)) then
    segments.append(("video", n_video))), and mod_segments is built by
    iterating those segments in order. Text can split into several tag runs, so
    the count varies at the front -- but the tail does not.

    Returning None adds nothing, which fails visibly as "the controlnet did
    nothing" rather than writing control data into the audio rows.
    """
    if not mod_segments:
        return None
    a, b = mod_segments[-1][0], mod_segments[-1][1]
    return (int(a), int(b)) if b > a else None


NODE_CLASS_MAPPINGS = {
    "H3FunControlLoader": H3FunControlLoader,
    "H3FunControlApply": H3FunControlApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FunControlLoader": "H3 Fun ControlNet Loader",
    "H3FunControlApply": "H3 Fun ControlNet Apply",
}
