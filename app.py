"""
Gradio Interactive Web Application for AI Text-to-Image Generation.
Launch with:
    python app.py
"""

import os
import random
from typing import List, Tuple
from PIL import Image
import gradio as gr
from pipeline import TextToImageGenerator

# Global lazy-loaded generator instance
generator_instance = None


def get_generator():
    global generator_instance
    if generator_instance is None:
        generator_instance = TextToImageGenerator()
    return generator_instance


def generate_ui(
    prompt: str,
    negative_prompt: str,
    style_preset: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    randomize_seed: bool,
    num_images: int,
    scheduler: str,
) -> Tuple[List[Image.Image], str, int]:
    if not prompt or not prompt.strip():
        return [], "Please enter a prompt describing the image you want to create.", seed

    gen = get_generator()

    # Switch scheduler if user changed it in UI
    if scheduler != getattr(gen, "scheduler_name", None):
        gen.set_scheduler(scheduler)

    if randomize_seed or seed == -1:
        seed = random.randint(0, 2**32 - 1)

    try:
        images, paths = gen.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            style_preset=style_preset,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale),
            width=int(width),
            height=int(height),
            seed=int(seed),
            num_images=int(num_images),
            save_images=True,
            save_metadata=True,
        )

        status_msg = f"Successfully generated {len(images)} image(s) [Seed: {seed} | Steps: {steps} | CFG: {cfg_scale}]\nSaved to: {', '.join([os.path.basename(p) for p in paths])}"
        return images, status_msg, seed

    except Exception as e:
        return [], f"Error during generation: {str(e)}", seed


def build_interface():
    custom_css = """
    .main-title { text-align: center; margin-bottom: 8px; font-weight: 800; font-size: 2.2rem; }
    .subtitle { text-align: center; color: #6b7280; margin-bottom: 24px; font-size: 1.05rem; }
    .generate-btn { font-size: 1.1rem !important; padding: 12px !important; font-weight: 700 !important; }
    """

    with gr.Blocks(title="AI Text-to-Image Studio", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("<h1 class='main-title'> AI Text-to-Image Studio</h1>")
        gr.Markdown("<p class='subtitle'>Transform creative text descriptions into high-resolution visuals using Stable Diffusion</p>")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="E.g., A serene Japanese garden with cherry blossoms, wooden bridge, koi pond, cinematic lighting...",
                    lines=3,
                    max_lines=6,
                )

                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (What to exclude)",
                    value=TextToImageGenerator.DEFAULT_NEGATIVE_PROMPT,
                    lines=2,
                    max_lines=4,
                )

                style_dropdown = gr.Dropdown(
                    label="Style Preset",
                    choices=list(TextToImageGenerator.STYLE_PRESETS.keys()),
                    value="None",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        steps_slider = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=1,
                            info="Higher steps = finer details, but takes longer."
                        )
                        cfg_slider = gr.Slider(
                            label="Guidance Scale (CFG)",
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            info="How closely the generation follows the prompt."
                        )

                    with gr.Row():
                        width_slider = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                        )
                        height_slider = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                        )

                    with gr.Row():
                        num_images_slider = gr.Slider(
                            label="Batch Count (Number of Images)",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                        )
                        scheduler_dropdown = gr.Dropdown(
                            label="Sampling Algorithm",
                            choices=list(TextToImageGenerator.SCHEDULER_MAP.keys()),
                            value="dpm++_2m_karras",
                        )

                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                        )
                        randomize_seed_checkbox = gr.Checkbox(
                            label="Randomize Seed",
                            value=True,
                        )

                generate_button = gr.Button(
                    "Generate Image",
                    variant="primary",
                    elem_classes=["generate-btn"]
                )

            with gr.Column(scale=1):
                gallery_output = gr.Gallery(
                    label="Generated Output",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height=512,
                    object_fit="contain",
                )
                status_box = gr.Textbox(
                    label="Generation Status",
                    interactive=False,
                    lines=2,
                )

        # Examples
        gr.Examples(
            examples=[
                [
                    "A cute fluffy cybernetic kitten in a neon futuristic Tokyo alleyway, glowing eyes, rain reflections",
                    TextToImageGenerator.DEFAULT_NEGATIVE_PROMPT,
                    "Cyberpunk",
                    30,
                    7.5,
                    512,
                    512,
                    42,
                    False,
                    1,
                    "dpm++_2m_karras",
                ],
                [
                    "Portrait of an elderly wise wizard reading an ancient glowing spellbook in a grand library",
                    TextToImageGenerator.DEFAULT_NEGATIVE_PROMPT,
                    "Photorealistic",
                    30,
                    7.5,
                    512,
                    512,
                    1234,
                    False,
                    1,
                    "dpm++_2m_karras",
                ],
                [
                    "Floating fantasy islands with cascading waterfalls and pink cherry blossom trees under a sunset sky",
                    TextToImageGenerator.DEFAULT_NEGATIVE_PROMPT,
                    "Digital Art",
                    35,
                    8.0,
                    512,
                    512,
                    999,
                    False,
                    1,
                    "dpm++_2m_karras",
                ],
            ],
            inputs=[
                prompt_input,
                negative_prompt_input,
                style_dropdown,
                steps_slider,
                cfg_slider,
                width_slider,
                height_slider,
                seed_input,
                randomize_seed_checkbox,
                num_images_slider,
                scheduler_dropdown,
            ],
        )

        generate_button.click(
            fn=generate_ui,
            inputs=[
                prompt_input,
                negative_prompt_input,
                style_dropdown,
                steps_slider,
                cfg_slider,
                width_slider,
                height_slider,
                seed_input,
                randomize_seed_checkbox,
                num_images_slider,
                scheduler_dropdown,
            ],
            outputs=[gallery_output, status_box, seed_input],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    # Launch with share=False for local, can be configured in app.launch(share=True) for public URL
    app.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
