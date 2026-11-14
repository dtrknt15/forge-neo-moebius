import gradio as gr
from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img
from .tools import MODEL_SPECS, get_custom_model_specs


def _get_opt(p: StableDiffusionProcessingImg2Img, key: str, default):
    if p is not None and hasattr(p, "override_settings"):
        overridden = p.override_settings.get(key, None)
        if overridden is not None:
            return overridden
    return shared.opts.data.get(key, default)


def get_moebius_upscaler(p: StableDiffusionProcessingImg2Img = None) -> str:
    return _get_opt(p, "moebius_upscaler", "ESRGAN_4x")


def get_moebius_resolution(p: StableDiffusionProcessingImg2Img = None) -> int:
    return int(_get_opt(p, "moebius_resolution", 512))


def get_moebius_max_resolution(p: StableDiffusionProcessingImg2Img = None) -> int:
    return int(_get_opt(p, "moebius_max_resolution", 1024))


def get_moebius_steps(p: StableDiffusionProcessingImg2Img = None) -> int:
    return int(_get_opt(p, "moebius_steps", 20))


def get_moebius_cfg(p: StableDiffusionProcessingImg2Img = None) -> float:
    return float(_get_opt(p, "moebius_cfg", 2.5))


def get_moebius_compensate(p: StableDiffusionProcessingImg2Img = None) -> bool:
    return bool(_get_opt(p, "moebius_compensate", False))


def get_moebius_seed(p: StableDiffusionProcessingImg2Img = None) -> int:
    mode = _get_opt(p, "moebius_seed_mode", "Random")
    if mode == "Follow img2img seed" and p is not None and getattr(p, "seed", -1) not in (-1, None):
        return int(p.seed)
    if mode == "Fixed (0)":
        return 0
    return -1


def get_moebius_unload() -> bool:
    return bool(shared.opts.data.get("moebius_unload_after_use", True))


def get_enabled_model_keys() -> list[str]:
    enabled = []
    for key, spec in MODEL_SPECS.items():
        default_enabled = key != "pretrained"
        if shared.opts.data.get(spec["option_key"], default_enabled):
            enabled.append(key)
    enabled.extend(get_custom_model_specs().keys())
    return enabled


moebius_settings = {
    "moebius_enable_pretrained": shared.OptionInfo(
        False,
        "Enable Moebius (Pretrained) in Inpaint Masked content",
    ).info("Requires UI reload / restart to update Masked content choices."),
    "moebius_enable_places2": shared.OptionInfo(
        True,
        "Enable Moebius (Places2 FT) in Inpaint Masked content",
    ).info("Requires UI reload / restart to update Masked content choices."),
    "moebius_enable_celebahq": shared.OptionInfo(
        True,
        "Enable Moebius (CelebA-HQ FT) in Inpaint Masked content",
    ).info("Requires UI reload / restart to update Masked content choices."),
    "moebius_enable_ffhq": shared.OptionInfo(
        True,
        "Enable Moebius (FFHQ FT) in Inpaint Masked content",
    ).info("Requires UI reload / restart to update Masked content choices."),
    "moebius_upscaler": shared.OptionInfo(
        "ESRGAN_4x",
        "Upscaler for Moebius masked content",
        gr.Dropdown,
        lambda: {"choices": [x.name for x in shared.sd_upscalers]},
    ),
    "moebius_resolution": shared.OptionInfo(
        512,
        "Resolution (short side) for Moebius inpainting",
        gr.Slider,
        {"minimum": 256, "maximum": 1024, "step": 64},
    ).info("512 is native resolution for Moebius."),
    "moebius_max_resolution": shared.OptionInfo(
        1024,
        "Max long-side resolution for elongated crops",
        gr.Slider,
        {"minimum": 512, "maximum": 2048, "step": 64},
    ).info("Prevents excessive VRAM usage on very wide/tall mask regions."),
    "moebius_steps": shared.OptionInfo(
        20,
        "Sampling steps for Moebius",
        gr.Slider,
        {"minimum": 1, "maximum": 50, "step": 1},
    ),
    "moebius_cfg": shared.OptionInfo(
        2.5,
        "CFG scale for Moebius",
        gr.Slider,
        {"minimum": 1.0, "maximum": 10.0, "step": 0.1},
    ),
    "moebius_compensate": shared.OptionInfo(
        False,
        "Enable Moebius boundary color/luminance compensation",
    ).info("Applies official paste_compensate blending at mask edges."),
    "moebius_seed_mode": shared.OptionInfo(
        "Random",
        "Seed mode for Moebius in Inpaint",
        gr.Radio,
        {"choices": ["Random", "Follow img2img seed", "Fixed (0)"]},
    ),
    "moebius_unload_after_use": shared.OptionInfo(
        True,
        "Unload Moebius models to CPU after inference",
    ).info("Recommended to save VRAM for Stable Diffusion / Flux."),
}

shared.options_templates.update(
    shared.options_section(("moebius_inpaint", "Moebius Inpaint"), moebius_settings)
)
