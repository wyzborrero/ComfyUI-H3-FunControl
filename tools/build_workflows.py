"""Generate the example workflows in ../workflows/ as loadable UI graphs.

Run it against a live ComfyUI that has this node installed:

    python tools/build_workflows.py [--server http://127.0.0.1:8188]

WHY GENERATE THEM
-----------------
A ComfyUI workflow file has to list every input of every node, in the right
order, with the right type, and split correctly between "this is a widget" and
"this is a socket". Hand-writing that is how you get a graph that loads with
red nodes and missing links. So the shape of each node is read from the live
server's /object_info, and only the values and the wiring are written here.

The layout, titles, groups and notes are the point: someone opening these
should be able to see what connects to what and read why, without going
looking for documentation.
"""

import argparse
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "workflows"

# Types that are sockets rather than widgets. Anything else in a node's
# required/optional dict is a widget, including every COMBO.
LINK_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "AUDIO", "VIDEO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS",
    "H3_FUN_CONTROL",
}


# Sol-Attn settings used by workflow 03. Stated explicitly rather than relying
# on the node's defaults, because one of them is load-bearing: morton MUST be
# False or this ControlNet's contribution lands on the wrong tokens.
SOLATTN_WIDGETS = {
    "tau": 1.3,
    "start_percent": 0.2,
    "end_percent": 0.9,
    "min_tokens": 4096,
    "int8_qk": True,
    "sink_conditioning": "exact_kv_and_rows",
    "morton": False,
    "morton_curve": "2d_frame",
    "int8_pv": True,
    "verbose": False,
    "use_tma": False,
    "dense_blocks": "0,-1",
}


def fetch(server):
    with urllib.request.urlopen(f"{server}/object_info", timeout=180) as r:
        return json.load(r)


# How many slots of an autogrow input to lay out. The node grows them on demand
# in the editor; a saved workflow has to name the ones it uses, so emit a couple
# of spare sockets rather than exactly the connected count.
AUTOGROW_SLOTS = 2


def spec_inputs(info, node_type):
    """[(name, type_string, is_widget)] in the order the node declares them.

    COMFY_AUTOGROW_V3 inputs are expanded here. /object_info declares one entry,
    `ref_images`, carrying a template and a prefix; a workflow file instead
    lists the individual sockets, `ref_images.ref_image_0` and so on. Expanding
    keeps the rest of this script honest about names that really exist.
    """
    schema = info[node_type]["input"]
    out = []
    for section in ("required", "optional"):
        for name, spec in (schema.get(section) or {}).items():
            t = spec[0]
            if t == "COMFY_AUTOGROW_V3":
                template = (spec[1] or {}).get("template", {})
                prefix = template.get("prefix", "")
                inner = ((template.get("input") or {}).get("required") or {})
                inner_type = next(iter(inner.values()))[0] if inner else "IMAGE"
                limit = min(AUTOGROW_SLOTS, template.get("max", AUTOGROW_SLOTS))
                for i in range(limit):
                    out.append((f"{name}.{prefix}{i}", inner_type, False))
            elif isinstance(t, list):        # COMBO, the options are the list
                out.append((name, "COMBO", True))
            elif t in LINK_TYPES:
                out.append((name, t, False))
            else:                            # INT / FLOAT / STRING / BOOLEAN
                out.append((name, t, True))
    return out


def build(info, nodes, links_spec, groups, notes):
    """Assemble the ComfyUI UI-format graph.

    nodes:      [{id, type, pos, size, title?, widgets?}]
    links_spec: [(src_id, src_slot, dst_id, dst_input_name)]
    """
    by_id = {n["id"]: n for n in nodes}

    # Links first, so each node knows which of its inputs carry one.
    links, incoming, outgoing = [], {}, {}
    for i, (src, slot, dst, dst_input) in enumerate(links_spec, start=1):
        t = None
        for name, typ, is_widget in spec_inputs(info, by_id[dst]["type"]):
            if name == dst_input:
                t = typ
                break
        if t is None:
            raise SystemExit(f"{by_id[dst]['type']} has no input {dst_input!r}")
        links.append([i, src, slot, dst, 0, t])
        incoming[(dst, dst_input)] = i
        outgoing.setdefault((src, slot), []).append(i)

    built = []
    for order, n in enumerate(nodes):
        node_type = n["type"]

        # Frontend-only nodes (MarkdownNote) are not in /object_info.
        if node_type == "MarkdownNote":
            built.append({
                "id": n["id"], "type": "MarkdownNote", "pos": n["pos"],
                "size": n["size"], "flags": {}, "order": order, "mode": 0,
                "inputs": [], "outputs": [],
                "title": n.get("title", "Note"),
                "properties": {}, "widgets_values": [n["text"]],
                "color": "#432", "bgcolor": "#653",
            })
            continue

        declared = spec_inputs(info, node_type)
        link_inputs = [d for d in declared if not d[2]]
        widget_inputs = [d for d in declared if d[2]]

        entries = []
        for name, typ, _ in link_inputs:
            entry = {"localized_name": name, "name": name, "type": typ,
                     "link": incoming.get((n["id"], name))}
            if "." in name:                  # autogrow slot, optional shape
                entry["label"] = name.split(".", 1)[1]
                entry["shape"] = 7
            entries.append(entry)
        values, named = [], {}
        for name, typ, _ in widget_inputs:
            entries.append({"localized_name": name, "name": name, "type": typ,
                            "widget": {"name": name}, "link": None})
            v = (n.get("widgets") or {}).get(name)
            values.append(v)
            named[name] = v

        outs = []
        spec = info[node_type]
        for i, otype in enumerate(spec.get("output") or []):
            oname = (spec.get("output_name") or [])[i] if spec.get("output_name") else otype
            outs.append({"localized_name": oname, "name": oname, "type": otype,
                         "links": outgoing.get((n["id"], i)) or None})

        node = {
            "id": n["id"], "type": node_type, "pos": n["pos"], "size": n["size"],
            "flags": {}, "order": order, "mode": 0,
            "inputs": entries, "outputs": outs,
            "properties": {"Node name for S&R": node_type},
            "widgets_values": values, "widgets_values_named": named,
        }
        if n.get("title"):
            node["title"] = n["title"]
        built.append(node)

    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": len(links),
        "nodes": built,
        "links": links,
        "groups": groups,
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def group(title, x, y, w, h, colour="#3f789e"):
    return {"id": abs(hash(title)) % 10000, "title": title,
            "bounding": [x, y, w, h], "color": colour, "font_size": 24,
            "flags": {}}


def note(nid, pos, size, title, text):
    return {"id": nid, "type": "MarkdownNote", "pos": pos, "size": size,
            "title": title, "text": text}


# ---------------------------------------------------------------------------
# 01: one control video, the smallest thing that works.
# ---------------------------------------------------------------------------
def workflow_single(info):
    N = [
        note(1, [40, 40], [560, 300], "Read me first", """## Single control video

The smallest useful H3 Fun-ControlNet graph: a **prompt**, an optional **first
frame**, and one **control video** that the generation follows frame by frame.

**Set these before running**

1. `Load Video` -> your control pass (depth, canny, pose, HED or MLSD).
   It must be the **same width, height and frame count** as the generation.
   The node raises with both numbers if they disagree.
2. `Load Image` -> a first frame, or delete it and the prompt does the work.
3. Prompt, width, height, length on the H3 node. Length snaps to H3's
   **17n + 5** grid at 24fps: 5, 22, 39, 56, 73, 90, 107 ...

**The ControlNet checkpoint must be curve-form.** The original full-width
release will not load and says so. See the README.

**strength 1.0 is the right starting point.** Higher does not improve
adherence, it saturates: washed out, subject smeared."""),
        note(2, [40, 370], [560, 250], "The schedule window", """## start_percent / end_percent

Control applies over this slice of the sampling schedule. The default 0.0 to
1.0 means it is asserted at every step, **including the last ones where texture
forms**.

If your control pass has featureless regions (a smooth ground plane, an empty
sky) the ControlNet keeps insisting those surfaces are smooth and suppresses
whatever texture the prompt asks for.

Ending around **0.6** pins the structure while the noise is still coarse and
frees the late steps to put in detail the control cannot describe."""),

        {"id": 10, "type": "UNETLoader", "pos": [660, 60], "size": [340, 82],
         "title": "H3 base model",
         "widgets": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                     "weight_dtype": "default"}},
        {"id": 11, "type": "CLIPLoader", "pos": [660, 190], "size": [340, 106],
         "title": "Qwen3-VL text encoder",
         "widgets": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                     "type": "minimax", "device": "default"}},
        {"id": 12, "type": "VAELoader", "pos": [660, 340], "size": [340, 58],
         "title": "Video VAE",
         "widgets": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        {"id": 13, "type": "VAELoader", "pos": [660, 430], "size": [340, 58],
         "title": "Audio VAE",
         "widgets": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},

        {"id": 20, "type": "H3FunControlLoader", "pos": [660, 560], "size": [340, 58],
         "title": "ControlNet (curve-form only)",
         "widgets": {"control_net_name": "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"}},
        {"id": 21, "type": "LoadVideo", "pos": [660, 660], "size": [340, 110],
         "title": "Control video: depth / canny / pose / HED / MLSD",
         "widgets": {"file": ""}},
        {"id": 22, "type": "GetVideoComponents", "pos": [660, 810], "size": [340, 78],
         "title": "Video -> IMAGE batch"},
        {"id": 23, "type": "H3FunControlApply", "pos": [1060, 560], "size": [360, 170],
         "title": "Apply H3 Fun ControlNet",
         "widgets": {"strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
        {"id": 24, "type": "MiniMaxH3SigmaShift", "pos": [1060, 770], "size": [360, 82],
         "title": "Sigma shift (video 12 / audio 3)",
         "widgets": {"shift_video": 12.0, "shift_audio": 3.0}},

        {"id": 30, "type": "LoadImage", "pos": [1060, 60], "size": [360, 314],
         "title": "First frame (optional)", "widgets": {"image": ""}},
        {"id": 31, "type": "MiniMaxH3ImageToVideo", "pos": [1470, 60], "size": [400, 300],
         "title": "Prompt + first frame -> conditioning",
         "widgets": {"prompt": "a person walking across an open field, cinematic",
                     "width": 1280, "height": 704, "length": 90}},

        {"id": 40, "type": "BasicGuider", "pos": [1470, 420], "size": [300, 60],
         "title": "Guider (no CFG, no negative)"},
        {"id": 41, "type": "KSamplerSelect", "pos": [1470, 520], "size": [300, 58],
         "widgets": {"sampler_name": "res_multistep"}},
        {"id": 42, "type": "BasicScheduler", "pos": [1470, 620], "size": [300, 106],
         "widgets": {"scheduler": "simple", "steps": 20, "denoise": 1.0}},
        {"id": 43, "type": "RandomNoise", "pos": [1470, 770], "size": [300, 82],
         "widgets": {"noise_seed": 0, "control_after_generate": "randomize"}},
        {"id": 44, "type": "SamplerCustomAdvanced", "pos": [1820, 420], "size": [320, 130],
         "title": "Sample"},

        {"id": 50, "type": "VAEDecode", "pos": [2190, 420], "size": [280, 58],
         "title": "Decode video"},
        {"id": 51, "type": "VAEDecodeAudio", "pos": [2190, 520], "size": [280, 58],
         "title": "Decode audio"},
        {"id": 52, "type": "CreateVideo", "pos": [2190, 620], "size": [280, 106],
         "widgets": {"fps": 24, "bit_depth": "auto", "color_space": 'sRGB'}},
        {"id": 53, "type": "SaveVideo", "pos": [2190, 770], "size": [280, 110],
         "widgets": {"filename_prefix": "h3_funcontrol", "format": "auto",
                     "codec": "auto"}},
    ]
    L = [
        (10, 0, 23, "model"), (20, 0, 23, "control_net"), (12, 0, 23, "vae"),
        (21, 0, 22, "video"), (22, 0, 23, "control_video"),
        (23, 0, 24, "model"),
        (11, 0, 31, "clip"), (12, 0, 31, "vae"), (30, 0, 31, "first_frame"),
        (24, 0, 40, "model"), (31, 0, 40, "conditioning"),
        (24, 0, 42, "model"),
        (43, 0, 44, "noise"), (40, 0, 44, "guider"), (41, 0, 44, "sampler"),
        (42, 0, 44, "sigmas"), (31, 1, 44, "latent_image"),
        (44, 0, 50, "samples"), (12, 0, 50, "vae"),
        (44, 0, 51, "samples"), (13, 0, 51, "vae"),
        (50, 0, 52, "images"), (51, 0, 52, "audio"),
        (52, 0, 53, "video"),
    ]
    G = [
        group("Models", 640, 0, 380, 510),
        group("CONTROL: this is the node", 640, 520, 800, 380, "#8e3f3f"),
        group("Conditioning", 1040, 0, 840, 380),
        group("Sampling", 1450, 400, 700, 470),
        group("Output", 2170, 400, 320, 500),
    ]
    return build(info, N, L, G, None)


# ---------------------------------------------------------------------------
# 02: two chained controls, a character reference, and CFG.
# ---------------------------------------------------------------------------
def workflow_dual(info):
    N = [
        note(1, [40, 40], [600, 430], "Read me first", """## Two controls, chained

Two `Apply H3 Fun ControlNet` nodes in series. The second takes the MODEL the
first returned, so each runs its own control stream and adds its own
contribution.

### Strength is a BUDGET, not two dials

**The two contributions sum.** Depth 1.0 plus pose 1.0 behaves like a single
control at 2.0 and saturates: washed out, subject smeared, detail gone.

Keep the total near **1.0**. The values here, depth 0.3 and pose 0.7, came from
testing on a figure crossing a locked-off frame.

### Why pose carries most of it there

On that shot, depth alone let the figure drift in distance from camera, growing
across the shot, while pose held it. The likely reason is token resolution:
H3's VAE downsamples 16x and then patches 2x2, so at 1280x704 the token grid is
40 by 22 per latent frame. A figure about 100px tall is roughly 3 of 22
vertical tokens. A depth ramp across three tokens says very little, where a
skeleton's joints stay unambiguous.

**This is not a law.** At a larger subject size depth may well win. Try both:
set one strength to 0 to hear only the other."""),
        note(2, [40, 500], [600, 340], "Character reference, and CFG", """## Why MiniMaxH3ReferenceToVideo

`MiniMaxH3ImageToVideo` accepts only a first frame, so a character can reach
the model only by being painted into that frame. If the frame comes from an
untextured source the character stays untextured, and no prompt wording fixes
it.

`MiniMaxH3ReferenceToVideo` takes real reference images that are present for
**every** frame. `ref_image_size` of `max` uses a 2048px short edge for the best
identity, but reference tokens ride through every sampling step, so it is
noticeably slower than `match`.

## CFG

`BasicGuider` has **no negative input at all**, so a negative prompt does
nothing until you use `CFGGuider`.

Common advice for H3 is to leave guidance at 1. With a long, specific style
brief we measured clearly better adherence at **cfg 4**. It costs about 1.9x,
since the model runs twice per step. Try both."""),

        {"id": 10, "type": "UNETLoader", "pos": [700, 60], "size": [360, 82],
         "title": "H3 ref2va base (takes reference images)",
         "widgets": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                     "weight_dtype": "default"}},
        {"id": 11, "type": "CLIPLoader", "pos": [700, 190], "size": [360, 106],
         "widgets": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                     "type": "minimax", "device": "default"}},
        {"id": 12, "type": "VAELoader", "pos": [700, 340], "size": [360, 58],
         "title": "Video VAE",
         "widgets": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        {"id": 13, "type": "VAELoader", "pos": [700, 430], "size": [360, 58],
         "title": "Audio VAE",
         "widgets": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        {"id": 20, "type": "H3FunControlLoader", "pos": [700, 560], "size": [360, 58],
         "title": "One loader feeds both towers",
         "widgets": {"control_net_name": "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"}},

        {"id": 21, "type": "LoadVideo", "pos": [700, 680], "size": [360, 110],
         "title": "Control 1: depth", "widgets": {"file": ""}},
        {"id": 22, "type": "GetVideoComponents", "pos": [700, 830], "size": [360, 78]},
        {"id": 25, "type": "LoadVideo", "pos": [700, 950], "size": [360, 110],
         "title": "Control 2: pose", "widgets": {"file": ""}},
        {"id": 26, "type": "GetVideoComponents", "pos": [700, 1100], "size": [360, 78]},

        {"id": 23, "type": "H3FunControlApply", "pos": [1120, 680], "size": [360, 170],
         "title": "Apply 1: depth, strength 0.3",
         "widgets": {"strength": 0.3, "start_percent": 0.0, "end_percent": 1.0}},
        {"id": 27, "type": "H3FunControlApply", "pos": [1120, 900], "size": [360, 170],
         "title": "Apply 2: pose, strength 0.7   (0.3 + 0.7 = 1.0)",
         "widgets": {"strength": 0.7, "start_percent": 0.0, "end_percent": 1.0}},
        {"id": 24, "type": "MiniMaxH3SigmaShift", "pos": [1120, 1110], "size": [360, 82],
         "widgets": {"shift_video": 12.0, "shift_audio": 3.0}},

        {"id": 30, "type": "LoadImage", "pos": [1120, 60], "size": [360, 314],
         "title": "Character / style reference", "widgets": {"image": ""}},
        {"id": 31, "type": "MiniMaxH3ReferenceToVideo", "pos": [1540, 60], "size": [420, 340],
         "title": "Positive: references + prompt",
         "widgets": {"prompt": "<Picture 1> is the character: use their face, hair and costume. "
                               "They walk across an open field. Cinematic, shallow depth of field.",
                     "width": 1280, "height": 704, "length": 90,
                     "ref_image_size": "match"}},
        {"id": 32, "type": "MiniMaxH3ReferenceToVideo", "pos": [1540, 440], "size": [420, 340],
         "title": "Negative: same references, negative text",
         "widgets": {"prompt": "blurry, low detail, watermark, text, deformed hands, extra limbs",
                     "width": 1280, "height": 704, "length": 90,
                     "ref_image_size": "match"}},

        {"id": 40, "type": "CFGGuider", "pos": [2020, 60], "size": [320, 120],
         "title": "CFGGuider: the negative only works here",
         "widgets": {"cfg": 4.0}},
        {"id": 41, "type": "KSamplerSelect", "pos": [2020, 230], "size": [320, 58],
         "widgets": {"sampler_name": "res_multistep"}},
        {"id": 42, "type": "BasicScheduler", "pos": [2020, 330], "size": [320, 106],
         "widgets": {"scheduler": "simple", "steps": 28, "denoise": 1.0}},
        {"id": 43, "type": "RandomNoise", "pos": [2020, 480], "size": [320, 82],
         "widgets": {"noise_seed": 0, "control_after_generate": "randomize"}},
        {"id": 44, "type": "SamplerCustomAdvanced", "pos": [2020, 620], "size": [320, 130]},

        {"id": 50, "type": "VAEDecode", "pos": [2400, 620], "size": [280, 58]},
        {"id": 51, "type": "VAEDecodeAudio", "pos": [2400, 720], "size": [280, 58]},
        {"id": 52, "type": "CreateVideo", "pos": [2400, 820], "size": [280, 106],
         "widgets": {"fps": 24, "bit_depth": "auto", "color_space": 'sRGB'}},
        {"id": 53, "type": "SaveVideo", "pos": [2400, 970], "size": [280, 110],
         "widgets": {"filename_prefix": "h3_funcontrol_dual", "format": "auto",
                     "codec": "auto"}},
    ]
    L = [
        (21, 0, 22, "video"), (25, 0, 26, "video"),
        (10, 0, 23, "model"), (20, 0, 23, "control_net"), (12, 0, 23, "vae"),
        (22, 0, 23, "control_video"),
        (23, 0, 27, "model"), (20, 0, 27, "control_net"), (12, 0, 27, "vae"),
        (26, 0, 27, "control_video"),
        (27, 0, 24, "model"),
        (11, 0, 31, "clip"), (12, 0, 31, "vae"), (13, 0, 31, "audio_vae"),
        (30, 0, 31, "ref_images.ref_image_0"),
        (11, 0, 32, "clip"), (12, 0, 32, "vae"), (13, 0, 32, "audio_vae"),
        (30, 0, 32, "ref_images.ref_image_0"),
        (24, 0, 40, "model"), (31, 0, 40, "positive"), (32, 0, 40, "negative"),
        (24, 0, 42, "model"),
        (43, 0, 44, "noise"), (40, 0, 44, "guider"), (41, 0, 44, "sampler"),
        (42, 0, 44, "sigmas"), (31, 1, 44, "latent_image"),
        (44, 0, 50, "samples"), (12, 0, 50, "vae"),
        (44, 0, 51, "samples"), (13, 0, 51, "vae"),
        (50, 0, 52, "images"), (51, 0, 52, "audio"),
        (52, 0, 53, "video"),
    ]
    G = [
        group("Models", 680, 0, 400, 640),
        group("CONTROL: chained, strengths SUM to about 1.0", 680, 650, 820, 560, "#8e3f3f"),
        group("Conditioning and references", 1100, 0, 880, 790),
        group("Sampling", 2000, 40, 360, 730),
        group("Output", 2380, 600, 320, 500),
    ]
    return build(info, N, L, G, None)


# ---------------------------------------------------------------------------
# 03: the same recipe as 02, plus optional sparse attention.
# ---------------------------------------------------------------------------
def _default_for(info, node_type, name):
    """A widget's declared default, from whichever section declares it."""
    schema = info[node_type]["input"]
    for section in ("required", "optional"):
        spec = (schema.get(section) or {}).get(name)
        if spec is None:
            continue
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if "default" in opts:
            return opts["default"]
        if isinstance(spec[0], list) and spec[0]:
            return spec[0][0]
        if "options" in opts and opts["options"]:
            return opts["options"][0]
        return ""
    raise KeyError(f"{node_type} has no input {name}")


def workflow_solattn(info):
    """02 with a SolAttnPatch inserted after the control towers.

    Kept as a separate file on purpose: Sol-Attn is a third-party node pack and
    needs Triton, so most people will not have it. Workflows 01 and 02 use
    nothing but ComfyUI core and this node.
    """
    graph = workflow_dual(info)

    patch_id = 60
    # Rewire: sigma shift currently reads from the second control apply (27).
    # Put the patch between them.
    for link in graph["links"]:
        _lid, src, _sslot, dst, _dslot, _t = link
        if src == 27 and dst == 24:
            link[1] = patch_id                      # sigma shift now reads the patch
    for n in graph["nodes"]:
        if n["id"] == 27:
            for o in n["outputs"]:
                o["links"] = [graph["last_link_id"] + 1]
        if n["id"] == 24:
            pass

    new_link = [graph["last_link_id"] + 1, 27, 0, patch_id, 0, "MODEL"]
    graph["links"].append(new_link)
    graph["last_link_id"] = new_link[0]

    node = {
        "id": patch_id, "type": "SolAttnPatch",
        "pos": [1540, 900], "size": [420, 400],
        "flags": {}, "order": 99, "mode": 0,
        "title": "Sol-Attn  (OPTIONAL, needs the SolAttn node pack + Triton)",
        "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL",
                    "link": new_link[0]}],
        "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL",
                     "links": [next(l[0] for l in graph["links"]
                                    if l[1] == patch_id and l[3] == 24)]}],
        "properties": {"Node name for S&R": "SolAttnPatch"},
        "widgets_values": [], "widgets_values_named": {},
    }
    # Fill the widgets from the live schema so nothing is missed.
    values, named = [], {}
    for name, typ, is_widget in spec_inputs(info, "SolAttnPatch"):
        if not is_widget:
            continue
        node["inputs"].append({"localized_name": name, "name": name, "type": typ,
                               "widget": {"name": name}, "link": None})
        # Anything not named in SOLATTN_WIDGETS falls back to the node's own
        # declared default, so a change in that pack does not break this.
        v = SOLATTN_WIDGETS.get(name, _default_for(info, "SolAttnPatch", name))
        values.append(v)
        named[name] = v
    node["widgets_values"] = values
    node["widgets_values_named"] = named
    graph["nodes"].append(node)
    graph["last_node_id"] = max(graph["last_node_id"], patch_id)

    graph["nodes"].append({
        "id": 61, "type": "MarkdownNote", "pos": [40, 880], "size": [600, 420],
        "flags": {}, "order": 100, "mode": 0, "inputs": [], "outputs": [],
        "title": "Sol-Attn: read before enabling",
        "properties": {}, "color": "#432", "bgcolor": "#653",
        "widgets_values": ["""## Optional speed-up

This graph is workflow 02 with one extra node. It needs
[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) and a
working Triton. If you do not have those, use workflow 02 instead: nothing else
here depends on it.

### morton MUST stay off

`morton` reorders video tokens into Z-order. This ControlNet adds its
contribution at **raster-order row offsets**, so a reordered stream puts the
control on the wrong tokens.

With `morton: true` **the control disappears completely and nothing errors**.
You get a clean-looking video that has quietly ignored its control input. See
Trap 8 in the README.

### Measured, RTX 5090

90 frames at 1280x704, 28 steps, depth 0.3 + pose 0.7, same seed throughout:

| | time | control |
|---|---|---|
| no Sol-Attn | 332s | follows the control |
| `morton: true` | 240s | **gone** |
| `morton: false` | **220s (1.51x)** | follows the control |

Your numbers will differ. Triton compiles its kernels on first use, so the
first run is a compile and not a measurement: run it twice before believing a
time.

`dense_blocks` of `0,-1` keeps the first and last transformer blocks exact.
They are the most approximation-sensitive, and block 0 is also where this
ControlNet's stream enters."""],
    })
    graph["groups"].append(group("OPTIONAL: sparse attention", 1520, 870, 460, 450, "#3f8e5f"))
    return graph


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    info = fetch(args.server)
    for name in ("H3FunControlLoader", "H3FunControlApply"):
        if name not in info:
            raise SystemExit(f"{name} is not installed on {args.server}")

    OUT.mkdir(parents=True, exist_ok=True)
    builders = [("01_single_control.json", workflow_single),
                ("02_depth_plus_pose_reference.json", workflow_dual)]
    # 03 needs the Sol-Attn pack present to read its widget list. Skip it
    # rather than emit a graph with guessed inputs.
    if "SolAttnPatch" in info:
        builders.append(("03_optional_solattn_speedup.json", workflow_solattn))
    else:
        print("  (SolAttnPatch not installed, skipping workflow 03)")
    for filename, fn in builders:
        graph = fn(info)
        (OUT / filename).write_text(json.dumps(graph, indent=2), encoding="utf-8")
        n_notes = sum(1 for n in graph["nodes"] if n["type"] == "MarkdownNote")
        print(f"  {filename}: {len(graph['nodes'])} nodes "
              f"({n_notes} notes), {len(graph['links'])} links, "
              f"{len(graph['groups'])} groups")


if __name__ == "__main__":
    main()
