"""
AI Text-to-Image Generation Pipeline Module.
Provides a robust, device-agnostic, and feature-complete wrapper around Stable Diffusion.
"""

import os
import time
import json
from datetime import datetime
from typing import Optional, List, Union, Tuple
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    AutoencoderKL,
)


class TextToImageGenerator:
    """
    Production-ready wrapper for Stable Diffusion pipelines.
    Supports CUDA, MPS (Apple Silicon), and CPU with automatic precision handling,
    multiple high-performance schedulers, memory optimizations, and image saving.
    """

    SCHEDULER_MAP = {
        "dpm++_2m_karras": lambda config: DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        ),
        "dpm_multistep": lambda config: DPMSolverMultistepScheduler.from_config(config),
        "euler_a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
        "euler": lambda config: EulerDiscreteScheduler.from_config(config),
        "ddim": lambda config: DDIMScheduler.from_config(config),
    }

    STYLE_PRESETS = {
        "None": "",
        "Photorealistic": "photorealistic, 8k resolution, highly detailed, professional photography, natural lighting, sharp focus",
        "Cinematic": "cinematic still, dramatic lighting, 35mm photograph, film grain, depth of field, blockbuster movie aesthetic",
        "Anime / Manga": "anime artwork, vivid colors, Studio Ghibli style, clean linework, highly detailed manga illustration",
        "Digital Art": "digital concept art, trending on ArtStation, dynamic composition, vibrant palette, fantasy illustration",
        "3D Render": "octane 3D render, Unreal Engine 5, raytracing, volumetric lighting, photorealistic textures, 8k",
        "Oil Painting": "oil on canvas, visible brushstrokes, textured, classical fine art masterpiece, rich pigments",
        "Cyberpunk": "cyberpunk style, neon lights, futuristic cityscape, rainy reflections, high tech, atmospheric glow",
        "Vintage / Retro": "vintage 1970s photography, warm retro color grading, Polaroid aesthetics, nostalgic grain",
    }

    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, distorted, deformed, bad anatomy, bad hands, missing fingers, "
        "extra limbs, duplicate, ugly, text, watermark, signature, grainy, low resolution"
    )

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        scheduler_name: str = "dpm++_2m_karras",
        enable_attention_slicing: bool = True,
        enable_xformers: bool = False,
        cpu_offload: bool = False,
        output_dir: str = "outputs",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the Text-to-Image Pipeline.

        :param model_id: Hugging Face model repository or local path.
        :param device: "cuda", "mps", "cpu", or None for auto-detection.
        :param torch_dtype: torch.float16, torch.float32, or None for auto-detection.
        :param scheduler_name: Key in SCHEDULER_MAP (default: dpm++_2m_karras).
        :param enable_attention_slicing: Enable to reduce VRAM usage on GPUs.
        :param enable_xformers: Enable xformers memory efficient attention if installed.
        :param cpu_offload: Offload model weights to CPU when not in use (saves VRAM).
        :param output_dir: Directory where generated images are saved.
        :param hf_token: Optional Hugging Face auth token.
        """
        self.model_id = model_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Device and Dtype Resolution
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        else:
            self.torch_dtype = torch_dtype

        print(f"[Pipeline] Initializing on device='{self.device}' with dtype={self.torch_dtype}")
        print(f"[Pipeline] Loading model '{self.model_id}'...")

        # 2. Pipeline Loading
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
        }
        if hf_token:
            load_kwargs["token"] = hf_token

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            **load_kwargs
        )

        # 3. Scheduler Configuration
        self.set_scheduler(scheduler_name)

        # 4. Memory Optimizations
        if cpu_offload and self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
            print("[Pipeline] Sequential CPU offload enabled.")
        else:
            self.pipe = self.pipe.to(self.device)

        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()

        if enable_xformers and self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[Pipeline] xformers memory-efficient attention enabled.")
            except Exception as e:
                print(f"[Pipeline] xformers not available ({e}), using default attention.")

        print("[Pipeline] Model loaded successfully and ready for generation.")

    def set_scheduler(self, scheduler_name: str):
        """Switch the diffusion scheduler."""
        if scheduler_name in self.SCHEDULER_MAP:
            self.pipe.scheduler = self.SCHEDULER_MAP[scheduler_name](self.pipe.scheduler.config)
            self.scheduler_name = scheduler_name
            print(f"[Pipeline] Scheduler set to '{scheduler_name}'.")
        else:
            print(f"[Pipeline] Unknown scheduler '{scheduler_name}'. Available: {list(self.SCHEDULER_MAP.keys())}")

    def format_prompt(self, prompt: str, style_preset: Optional[str] = None) -> str:
        """Combine user prompt with style presets."""
        style_suffix = self.STYLE_PRESETS.get(style_preset, "") if style_preset else ""
        if style_suffix:
            return f"{prompt.strip()}, {style_suffix}"
        return prompt.strip()

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style_preset: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        num_images: int = 1,
        save_images: bool = True,
        save_metadata: bool = True,
    ) -> Tuple[List[Image.Image], List[str]]:
        """
        Generate images from a text prompt.

        :param prompt: Text prompt describing the desired image.
        :param negative_prompt: Concepts to avoid (defaults to clean negative prompt).
        :param style_preset: Optional artistic style preset name.
        :param num_inference_steps: Denoising steps (default: 30).
        :param guidance_scale: Classifier-Free Guidance (CFG) scale (default: 7.5).
        :param width: Image width (must be multiple of 8, default: 512).
        :param height: Image height (must be multiple of 8, default: 512).
        :param seed: Random seed for deterministic generation.
        :param num_images: Number of images to generate for this prompt.
        :param save_images: Whether to save generated images to disk.
        :param save_metadata: Whether to save generation metadata JSON alongside images.
        :return: (List of PIL Images, List of saved image filepaths).
        """
        # 1. Format prompts
        final_prompt = self.format_prompt(prompt, style_preset)
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        # 2. Generator / Seed Setup
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # For GPU, using CUDA generator; for CPU, cpu generator
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[Generate] Starting inference: {num_images} image(s), steps={num_inference_steps}, CFG={guidance_scale}, seed={seed}")
        start_time = time.time()

        # 3. Run Diffusion Pipeline
        with torch.inference_mode():
            output = self.pipe(
                prompt=[final_prompt] * num_images,
                negative_prompt=[negative_prompt] * num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        elapsed = time.time() - start_time
        images = output.images
        print(f"[Generate] Completed in {elapsed:.2f}s ({elapsed/num_images:.2f}s/image).")

        saved_paths = []
        if save_images:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for idx, img in enumerate(images):
                img_filename = f"img_{timestamp}_s{seed}_{idx}.png"
                img_path = os.path.join(self.output_dir, img_filename)
                img.save(img_path, format="PNG")
                saved_paths.append(img_path)

                if save_metadata:
                    meta_filename = f"img_{timestamp}_s{seed}_{idx}.json"
                    meta_path = os.path.join(self.output_dir, meta_filename)
                    metadata = {
                        "prompt": prompt,
                        "final_prompt": final_prompt,
                        "negative_prompt": negative_prompt,
                        "style_preset": style_preset,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "width": width,
                        "height": height,
                        "seed": seed,
                        "index": idx,
                        "scheduler": getattr(self, "scheduler_name", "unknown"),
                        "model_id": self.model_id,
                        "device": self.device,
                        "inference_time_sec": round(elapsed, 2),
                        "created_at": datetime.now().isoformat(),
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)

        return images, saved_paths
