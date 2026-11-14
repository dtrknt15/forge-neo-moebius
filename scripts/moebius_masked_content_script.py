import importlib
import gradio as gr
from PIL import Image
import modules.scripts
from modules.processing import StableDiffusionProcessingImg2Img, create_binary_mask

from moebius_masked_content import tools, options


INPAINTING_FILL_ELEMENTS = ["img2img_inpainting_fill", "replacer_inpainting_fill"]
MODEL_CHOICE_INDICES: dict[int, str] = {}


def _get_label_to_key() -> dict[str, str]:
    return {spec["label"]: key for key, spec in tools.get_all_model_specs().items()}


def get_runtime_functions(reload_modules=True):
    from moebius_masked_content import tools as t, model as m, inpaint as inp, options as opt

    if reload_modules:
        importlib.reload(t)
        importlib.reload(m)
        importlib.reload(inp)
        importlib.reload(opt)

    return (
        inp.moebius_inpaint,
        opt.get_moebius_upscaler,
        opt.get_moebius_resolution,
        opt.get_moebius_max_resolution,
        opt.get_moebius_steps,
        opt.get_moebius_cfg,
        opt.get_moebius_seed,
        opt.get_moebius_compensate,
    )


class Script(modules.scripts.Script):
    def title(self):
        return "Moebius-masked-content"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if component.elem_id == "mode_img2img":
            self.expect_selected_tab = True
        elif getattr(self, "expect_selected_tab", False) and isinstance(component, gr.Number):
            self.selected_tab = component
            self.expect_selected_tab = False

        if component.elem_id == "img2img_batch_inpaint_mask_dir":
            all_specs = tools.get_all_model_specs()
            enabled_keys = options.get_enabled_model_keys() or [
                k for k in all_specs if k != "pretrained"
            ]
            choices = [all_specs[k]["label"] for k in enabled_keys if k in all_specs]
            with gr.Row():
                self.batch_moebius = gr.Checkbox(
                    label="Use Moebius in Batch",
                    value=False,
                    elem_id="img2img_batch_moebius",
                    info="Match each image with a mask of the same filename in the mask directory.",
                )
                self.batch_moebius_model = gr.Dropdown(
                    label="Batch Moebius Model",
                    choices=choices,
                    value=choices[0],
                    elem_id="img2img_batch_moebius_model",
                )

    def ui(self, is_img2img):
        if not is_img2img:
            return None
        checkbox = getattr(self, "batch_moebius", None)
        if checkbox is None:
            checkbox = gr.Checkbox(
                label="Use Moebius in Batch",
                value=False,
                elem_id="img2img_batch_moebius",
            )
        model_dd = getattr(self, "batch_moebius_model", None)
        if model_dd is None:
            all_specs = tools.get_all_model_specs()
            enabled_keys = options.get_enabled_model_keys() or [
                k for k in all_specs if k != "pretrained"
            ]
            choices = [all_specs[k]["label"] for k in enabled_keys if k in all_specs]
            model_dd = gr.Dropdown(
                label="Batch Moebius Model",
                choices=choices,
                value=choices[0],
                elem_id="img2img_batch_moebius_model",
            )
        selected_tab = getattr(self, "selected_tab", None)
        controls = [checkbox, model_dd]
        if selected_tab is not None:
            controls.append(selected_tab)
        return controls

    def before_process(
        self,
        p: StableDiffusionProcessingImg2Img,
        batch_moebius=False,
        batch_moebius_model=None,
        selected_tab=None,
        *args,
    ):
        if not hasattr(p, "inpainting_fill"):
            return

        is_batch = selected_tab == 5 if selected_tab is not None else bool(batch_moebius)
        use_batch_moebius = is_batch and bool(batch_moebius)
        if is_batch and not use_batch_moebius:
            return

        all_specs = tools.get_all_model_specs()
        if use_batch_moebius:
            label_to_key = _get_label_to_key()
            model_key = label_to_key.get(batch_moebius_model, "places2")
            p.extra_generation_params.pop("Masked content", None)
        else:
            model_key = MODEL_CHOICE_INDICES.get(p.inpainting_fill)
            if model_key is None:
                return

        if getattr(p, "image_mask", None) is None or not p.init_images:
            return

        padding = p.inpaint_full_res_padding if p.inpaint_full_res else None
        (
            moebius_inpaint,
            get_upscaler,
            get_resolution,
            get_max_resolution,
            get_steps,
            get_cfg,
            get_seed,
            get_compensate,
        ) = get_runtime_functions(reload_modules=not use_batch_moebius)

        mask = p.image_mask
        if use_batch_moebius:
            mask = create_binary_mask(mask)
            if mask.size != p.init_images[0].size:
                mask = mask.resize(p.init_images[0].size, Image.Resampling.NEAREST)
            p.image_mask = mask

        result = moebius_inpaint(
            p.init_images[0],
            mask,
            p.inpainting_mask_invert,
            get_upscaler(p),
            padding,
            get_resolution(p),
            p.mask_blur,
            model_key=model_key,
            steps=get_steps(p),
            cfg=get_cfg(p),
            seed=get_seed(p),
            max_resolution=get_max_resolution(p),
            compensate=get_compensate(p),
        )
        p.init_images = [result] * len(p.init_images)
        p.extra_generation_params["Masked content"] = all_specs[model_key]["label"]
        p.inpainting_fill = 1  # original


def add_into_masked_content(component, **kwargs):
    elem_id = kwargs.get("elem_id", None)
    if elem_id not in INPAINTING_FILL_ELEMENTS:
        return
    all_specs = tools.get_all_model_specs()
    enabled_keys = options.get_enabled_model_keys()
    for key in enabled_keys:
        if key not in all_specs:
            continue
        label = all_specs[key]["label"]
        new_element = (label, label)
        if new_element not in component.choices:
            component.choices.append(new_element)
        idx = component.choices.index(new_element)
        if elem_id == "img2img_inpainting_fill" or idx not in MODEL_CHOICE_INDICES:
            MODEL_CHOICE_INDICES[idx] = key


modules.scripts.script_callbacks.on_after_component(add_into_masked_content)
