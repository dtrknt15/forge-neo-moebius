import gradio as gr
from PIL import Image
from modules import scripts_postprocessing
from modules.api.api import encode_pil_to_base64, decode_base64_to_image

from moebius_masked_content.inpaint import moebius_inpaint
from moebius_masked_content.options import (
    get_enabled_model_keys,
    get_moebius_upscaler,
    get_moebius_resolution,
    get_moebius_max_resolution,
    get_moebius_steps,
    get_moebius_cfg,
    get_moebius_compensate,
)
from moebius_masked_content.tools import (
    get_all_model_specs,
    limit_size_by_min_dimension,
    generate_seed,
)

if hasattr(scripts_postprocessing.ScriptPostprocessing, "process_firstpass"):
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None


def get_current_image(image):
    if image is None:
        return None
    max_resolution = 1280
    pil_img = decode_base64_to_image(image)
    pil_img = pil_img.resize(limit_size_by_min_dimension(pil_img.size, max_resolution))
    encoded = "data:image/png;base64," + encode_pil_to_base64(pil_img).decode()
    return gr.Image.update(encoded)


class ScriptPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = "Moebius Inpaint"
    order = 17550

    def ui(self):
        all_specs = get_all_model_specs()
        enabled_keys = get_enabled_model_keys() or [
            k for k in all_specs if k != "pretrained"
        ]
        model_choices = [all_specs[k]["label"] for k in enabled_keys if k in all_specs]
        default_res = get_moebius_resolution()
        default_steps = get_moebius_steps()
        default_cfg = get_moebius_cfg()
        default_compensate = get_moebius_compensate()

        with (
            InputAccordion(False, label=self.name, elem_id="moebius_extras")
            if InputAccordion
            else gr.Accordion(self.name, open=False, elem_id="moebius_extras")
            as enable
        ):
            if not InputAccordion:
                with gr.Row():
                    enable = gr.Checkbox(False, label="Enable")

            with gr.Row():
                model = gr.Radio(
                    label="Model",
                    choices=model_choices,
                    value=model_choices[0],
                )

            with gr.Row():
                create_canvas = gr.Button("Create canvas")
                mask_source = gr.CheckboxGroup(
                    ["Draw mask", "Upload mask"],
                    value=["Draw mask"],
                    label="Canvas mask source",
                )
                mask_brush_color = gr.ColorPicker(
                    "#84FF9A",
                    label="Brush color",
                    info="visual only, use when brush color is hard to see",
                )

            with gr.Row():
                input_mask = gr.Image(
                    label="Mask",
                    show_label=False,
                    elem_id="moebius_extras_mask",
                    source="upload",
                    interactive=True,
                    type="pil",
                    tool="sketch",
                    image_mode="RGBA",
                    brush_color="#84FF9A",
                )

            with gr.Row():
                blur = gr.Slider(label="Mask blur", minimum=0, maximum=128, value=2, step=1)
                padding = gr.Slider(
                    label="Padding",
                    minimum=-1,
                    maximum=512,
                    value=90,
                    step=1,
                    info="-1 for no padding",
                )
                resolution = gr.Slider(
                    label="Resolution",
                    minimum=256,
                    maximum=1024,
                    value=default_res,
                    step=64,
                )

            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=default_steps, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=default_cfg, step=0.1)
                seed = gr.Number(value=-1, label="Seed", minimum=-1, step=1)

            with gr.Row():
                invert = gr.Checkbox(label="Invert mask", value=False)
                compensate = gr.Checkbox(label="Boundary compensate", value=default_compensate)
                include_mask = gr.Checkbox(label="Include mask", value=False)

            mask_brush_color.change(
                fn=lambda color: gr.Image.update(brush_color=color),
                inputs=[mask_brush_color],
                outputs=[input_mask],
            )

            dummy_component = gr.Label(visible=False)
            create_canvas.click(
                fn=get_current_image,
                _js="getCurrentExtraSourceImg_moebius",
                inputs=[dummy_component],
                outputs=[input_mask],
                postprocess=False,
            )

        return {
            "enable": enable,
            "model": model,
            "mask_source": mask_source,
            "input_mask": input_mask,
            "blur": blur,
            "padding": padding,
            "resolution": resolution,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "invert": invert,
            "compensate": compensate,
            "include_mask": include_mask,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args["enable"]:
            return

        label_to_key = {
            spec["label"]: key for key, spec in get_all_model_specs().items()
        }
        padding = None if args["padding"] == -1 else int(args["padding"])
        model_label = args["model"]
        model_key = label_to_key.get(model_label, "places2")
        invert = bool(args["invert"])
        resolution = int(args["resolution"])
        blur = int(args["blur"])
        steps = int(args["steps"])
        cfg = float(args["cfg"])
        compensate = bool(args["compensate"])
        seed = int(args["seed"])
        if seed == -1:
            seed = generate_seed()

        mask = None
        if args["input_mask"]:
            if "Upload mask" in args["mask_source"]:
                mask = args["input_mask"]["image"].convert("L").resize(pp.image.size)
            if "Draw mask" in args["mask_source"]:
                mask = Image.new("L", pp.image.size, 0) if mask is None else mask
                draw_mask = args["input_mask"]["mask"].convert("L").resize(pp.image.size)
                mask.paste(draw_mask, draw_mask)

        if mask is None:
            return

        pp.image = moebius_inpaint(
            pp.image,
            mask,
            invert,
            get_moebius_upscaler(),
            padding,
            resolution,
            blur,
            model_key=model_key,
            steps=steps,
            cfg=cfg,
            seed=seed,
            max_resolution=get_moebius_max_resolution(),
            compensate=compensate,
        )

        pp.info[self.name] = (
            f"model='{model_label}', blur={blur}, padding={padding}, "
            f"resolution={resolution}, steps={steps}, cfg={cfg}, "
            f"compensate={compensate}, seed={seed}, invert={invert}"
        )
        if args["include_mask"]:
            pp.extra_images.append(mask)
