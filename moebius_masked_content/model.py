import os
import sys
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from modules import devices, shared

from .tools import (
    VENDOR_PATH,
    ensure_vae_downloaded,
    ensure_model_downloaded,
    generate_seed,
)
from .options import get_moebius_unload


def _ensure_vendor_on_path():
    if VENDOR_PATH not in sys.path:
        sys.path.insert(0, VENDOR_PATH)


class MoebiusInpaintModel:
    """Lazy-loading manager for Moebius removal models and PixelHacker VAE."""

    def __init__(self):
        self.removal_model = None
        self.vae = None
        self.scheduler = None
        self.pipeline = None
        self.active_key = None
        self.dtype = torch.float16

    def _get_dtype(self) -> torch.dtype:
        if devices.device.type == "cpu" or getattr(devices, "dtype", None) == torch.float32:
            return torch.float32
        return torch.float16

    def _load(self, model_key: str):
        _ensure_vendor_on_path()
        from removal.v1_2 import build_removal_model, load_cfg, load_removal_model
        from removal.v1_2.pipeline import RemovalSDXLPipeline_BatchMode

        target_device = devices.device
        self.dtype = self._get_dtype()

        if self.vae is None:
            vae_dir = ensure_vae_downloaded()
            self.vae = AutoencoderKL.from_pretrained(vae_dir)
            self.vae.requires_grad_(False).eval()

        if self.removal_model is None:
            cfg_path = os.path.join(VENDOR_PATH, "config", "model_cfg", "moebius.yaml")
            cfg = load_cfg(cfg_path)
            self.removal_model = build_removal_model(cfg, 20)
            self.removal_model.requires_grad_(False).eval()

        if self.active_key != model_key:
            weight_path = ensure_model_downloaded(model_key)
            load_removal_model(
                self.removal_model,
                weight_path,
                device=target_device,
                dtype=self.dtype,
                strict=True,
            )
            self.active_key = model_key
        else:
            self.removal_model.to(device=target_device, dtype=self.dtype)

        self.vae.to(device=target_device, dtype=self.dtype)

        if self.scheduler is None:
            self.scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
            )

        if self.pipeline is None:
            self.pipeline = RemovalSDXLPipeline_BatchMode(
                removal_model=self.removal_model,
                vae=self.vae,
                scheduler=self.scheduler,
                device=target_device,
                dtype=self.dtype,
            )
        else:
            self.pipeline.device = target_device
            self.pipeline.dtype = self.dtype
            self.pipeline.input_ids = self.pipeline.input_ids.to(device=target_device)

    def _unload_if_needed(self):
        should_unload = (
            get_moebius_unload()
            or getattr(shared.cmd_opts, "lowvram", False)
            or getattr(shared.cmd_opts, "medvram", False)
        )
        if should_unload:
            if self.removal_model is not None:
                self.removal_model.to("cpu")
            if self.vae is not None:
                self.vae.to("cpu")
            devices.torch_gc()

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        model_key: str = "places2",
        resolution: int = 512,
        steps: int = 20,
        cfg: float = 2.5,
        seed: int = -1,
        compensate: bool = False,
    ) -> Image.Image:
        self._load(model_key)
        actual_seed = generate_seed() if seed in (-1, None) else int(seed)

        rgb_image = image.convert("RGB")
        l_mask = mask.convert("L")

        try:
            with torch.no_grad():
                results = self.pipeline(
                    [rgb_image],
                    [l_mask],
                    image_size=resolution,
                    num_steps=steps,
                    guidance_scale=cfg,
                    retry=actual_seed,
                    paste=compensate,
                    compensate=compensate,
                    mute=True,
                )
            return results[0]
        finally:
            self._unload_if_needed()
