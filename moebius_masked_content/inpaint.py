import copy
from dataclasses import dataclass
from typing import Any
from PIL import Image, ImageOps
from modules import shared
from modules.images import resize_image

from .model import MoebiusInpaintModel
from .tools import (
    get_all_model_specs,
    crop,
    uncrop,
    are_images_the_same,
    apply_mask_blur,
    limit_size_with_max,
)

colorfix = None
try:
    from modules import colorfix
except ImportError:
    try:
        from srmodule import colorfix
    except ImportError:
        pass

moebius_model = MoebiusInpaintModel()


@dataclass
class CacheData:
    image: Any
    mask: Any
    invert: Any
    upscaler: Any
    padding: Any
    resolution: Any
    max_resolution: Any
    blur: Any
    model_key: Any
    steps: Any
    cfg: Any
    seed: Any
    compensate: Any
    result: Any


cached_data: CacheData | None = None


def moebius_inpaint(
    image: Image.Image,
    mask: Image.Image,
    invert: int | bool,
    upscaler: str,
    padding: int | None,
    resolution: int,
    blur: int,
    model_key: str = "places2",
    steps: int = 20,
    cfg: float = 2.5,
    seed: int = -1,
    max_resolution: int = 1024,
    compensate: bool = False,
) -> Image.Image:
    global cached_data
    all_specs = get_all_model_specs()
    if model_key not in all_specs:
        model_key = model_key.lower()
    if model_key not in all_specs:
        raise ValueError(f"Unknown Moebius model: {model_key}")

    invert_flag = bool(invert)
    use_cache = seed != -1
    if (
        use_cache
        and cached_data is not None
        and cached_data.invert == invert_flag
        and cached_data.upscaler == upscaler
        and cached_data.padding == padding
        and cached_data.resolution == resolution
        and cached_data.max_resolution == max_resolution
        and cached_data.blur == blur
        and cached_data.model_key == model_key
        and cached_data.steps == steps
        and cached_data.cfg == cfg
        and cached_data.seed == seed
        and cached_data.compensate == compensate
        and are_images_the_same(cached_data.image, image)
        and are_images_the_same(cached_data.mask, mask)
    ):
        result = copy.copy(cached_data.result)
        shared.state.assign_current_image(result)
        return result

    for_cache = CacheData(
        image.copy(),
        mask.copy(),
        invert_flag,
        upscaler,
        padding,
        resolution,
        max_resolution,
        blur,
        model_key,
        steps,
        cfg,
        seed,
        compensate,
        None,
    )

    if invert_flag:
        mask = ImageOps.invert(mask.convert("L"))
    else:
        mask = mask.convert("L")
    mask = apply_mask_blur(mask, blur)

    init_image = copy.copy(image)
    work_image = copy.copy(init_image)
    mask_not_cropped = mask
    if padding is not None:
        work_image = crop(work_image, mask_not_cropped, padding, resolution)
        mask = crop(mask, mask_not_cropped, padding, resolution)

    eff_res = min(*work_image.size, resolution)
    new_w, new_h = limit_size_with_max(work_image.size, eff_res, max_resolution)
    new_w = max(64, new_w - (new_w % 64))
    new_h = max(64, new_h - (new_h % 64))
    image_res = work_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_res = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    label = all_specs[model_key]["label"]
    shared.state.textinfo = f"{label} inpainting"
    tmp_image = moebius_model(
        image_res,
        mask_res,
        model_key=model_key,
        resolution=eff_res,
        steps=steps,
        cfg=cfg,
        seed=seed,
        compensate=compensate,
    )

    inpainted = image_res.copy()
    inpainted.paste(tmp_image.resize(image_res.size, Image.Resampling.LANCZOS), mask_res)
    shared.state.assign_current_image(inpainted)

    w, h = work_image.size
    shared.state.textinfo = f"upscaling {label} inpainted"
    inpainted = inpainted.convert("RGB")
    before_upscale = inpainted
    inpainted = resize_image(0, inpainted, w, h, upscaler)
    if colorfix:
        inpainted = colorfix.wavelet_color_fix(
            inpainted, before_upscale.resize(inpainted.size)
        ).convert("RGBA")

    result = work_image.copy()
    result.paste(inpainted, mask)
    if padding is not None:
        result = uncrop(result, init_image, mask_not_cropped, padding)

    shared.state.textinfo = ""
    if use_cache:
        for_cache.result = result.copy()
        cached_data = for_cache

    return result
