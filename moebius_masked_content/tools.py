import os
import random
import cv2
import numpy as np
from PIL import Image, ImageChops
from huggingface_hub import hf_hub_download
from modules import shared, masking


EXTENSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(EXTENSION_ROOT, "weights")
CUSTOM_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH, "custom")
VENDOR_PATH = os.path.join(EXTENSION_ROOT, "moebius_vendor")

CUSTOM_WEIGHT_EXTS = (".bin", ".pt", ".pth", ".safetensors")

MODEL_SPECS = {
    "pretrained": {
        "label": "Moebius (Pretrained)",
        "repo_id": "hustvl/Moebius",
        "filename": "pretrained/diffusion_pytorch_model.bin",
        "option_key": "moebius_enable_pretrained",
    },
    "places2": {
        "label": "Moebius (Places2)",
        "repo_id": "hustvl/Moebius",
        "filename": "ft_places2/diffusion_pytorch_model.bin",
        "option_key": "moebius_enable_places2",
    },
    "celebahq": {
        "label": "Moebius (CelebA-HQ)",
        "repo_id": "hustvl/Moebius",
        "filename": "ft_celebahq/diffusion_pytorch_model.bin",
        "option_key": "moebius_enable_celebahq",
    },
    "ffhq": {
        "label": "Moebius (FFHQ)",
        "repo_id": "hustvl/Moebius",
        "filename": "ft_ffhq/diffusion_pytorch_model.bin",
        "option_key": "moebius_enable_ffhq",
    },
}


def get_custom_model_specs() -> dict[str, dict]:
    """Scan weights/custom/ for user-supplied Moebius FT checkpoints."""
    os.makedirs(CUSTOM_WEIGHTS_PATH, exist_ok=True)
    specs: dict[str, dict] = {}
    for entry in sorted(os.listdir(CUSTOM_WEIGHTS_PATH)):
        if entry.startswith("."):
            continue
        full_path = os.path.join(CUSTOM_WEIGHTS_PATH, entry)
        if not os.path.isfile(full_path) or not entry.lower().endswith(CUSTOM_WEIGHT_EXTS):
            continue
        stem = os.path.splitext(entry)[0]
        key = f"custom:{entry}"
        specs[key] = {
            "label": f"Moebius (Custom: {stem})",
            "local_path": full_path,
            "is_custom": True,
        }
    return specs


def get_all_model_specs() -> dict[str, dict]:
    """Return built-in model specs merged with discovered custom FT models."""
    specs = dict(MODEL_SPECS)
    specs.update(get_custom_model_specs())
    return specs


def ensure_vae_downloaded() -> str:
    """Download the companion PixelHacker VAE into weights/vae if missing."""
    vae_dir = os.path.join(WEIGHTS_PATH, "vae")
    os.makedirs(vae_dir, exist_ok=True)
    for fname in ("config.json", "diffusion_pytorch_model.bin"):
        target = os.path.join(vae_dir, fname)
        if not os.path.exists(target):
            hf_hub_download(
                repo_id="hustvl/PixelHacker",
                filename=f"vae/{fname}",
                local_dir=WEIGHTS_PATH,
            )
    return vae_dir


def ensure_model_downloaded(model_key: str) -> str:
    """Return local checkpoint path, downloading built-in weights if missing."""
    all_specs = get_all_model_specs()
    if model_key not in all_specs:
        raise ValueError(f"Unknown Moebius model key: {model_key}")
    spec = all_specs[model_key]
    if spec.get("is_custom"):
        return spec["local_path"]

    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    target = os.path.join(WEIGHTS_PATH, spec["filename"])
    if not os.path.exists(target):
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=WEIGHTS_PATH,
        )
    return target


def crop(image: Image.Image, orig_mask: Image.Image, padding: int, resolution: int) -> Image.Image:
    crop_region = masking.get_crop_region(orig_mask, padding)
    crop_region = masking.expand_crop_region(crop_region, 1, 1, orig_mask.width, orig_mask.height)
    if shared.opts.data.get("integer_only_masked", False):
        x1, y1, x2, y2 = crop_region
        new_w, new_h = limit_size_by_min_dimension((x2 - x1, y2 - y1), resolution)
        crop_region = masking.fix_crop_region_integer_scale(
            crop_region, new_w, new_h, orig_mask.width, orig_mask.height
        )
    return image.crop(crop_region)


def uncrop(image: Image.Image, orig_image: Image.Image, orig_mask: Image.Image, padding: int) -> Image.Image:
    crop_region = masking.get_crop_region(orig_mask, padding)
    crop_region = masking.expand_crop_region(crop_region, 1, 1, orig_mask.width, orig_mask.height)
    x1, y1, x2, y2 = crop_region
    patch_size = (x2 - x1, y2 - y1)
    patch = image.convert("RGBA")
    if patch.size != patch_size:
        patch = patch.resize(patch_size)
    mask = orig_mask.crop(crop_region).resize(patch_size).convert("L")
    result = orig_image.convert("RGBA")
    result.paste(patch, (x1, y1), mask)
    return result


def are_images_the_same(image_one: Image.Image, image_two: Image.Image) -> bool:
    if image_one.size != image_two.size:
        return False
    diff = ImageChops.difference(image_one.convert("RGB"), image_two.convert("RGB"))
    return diff.getbbox() is None


def limit_size_by_min_dimension(size: tuple, limit: int) -> tuple[int, int]:
    w, h = size
    k = limit / min(w, h)
    return int(w * k), int(h * k)


def limit_size_with_max(size: tuple, min_limit: int, max_limit: int) -> tuple[int, int]:
    """Scale so the short side targets min_limit while clamping the long side to max_limit."""
    new_w, new_h = limit_size_by_min_dimension(size, min_limit)
    long_side = max(new_w, new_h)
    if max_limit > 0 and long_side > max_limit:
        scale = max_limit / long_side
        new_w = int(new_w * scale)
        new_h = int(new_h * scale)
    return new_w, new_h


def apply_mask_blur(image_mask: Image.Image, mask_blur: int) -> Image.Image:
    original_mode = image_mask.mode
    if mask_blur > 0:
        np_mask = np.array(image_mask).astype(np.uint8)
        kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)
        image_mask = Image.fromarray(np_mask).convert(original_mode)
    return image_mask


def generate_seed() -> int:
    return int(random.randrange(4294967294))
