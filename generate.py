"""
Command-Line Interface (CLI) for Text-to-Image Generation.
Example:
    python generate.py --prompt "A majestic lion sitting on a cliff during golden hour" --style "Photorealistic" --seed 42 --steps 30
"""

import argparse
import sys
import os
from pipeline import TextToImageGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="AI Text-to-Image Generation CLI")
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired image."
    )
    parser.add_argument(
        "-n", "--negative-prompt",
        type=str,
        default=None,
        help="Concepts or elements to exclude from the image."
    )
    parser.add_argument(
        "-s", "--style",
        type=str,
        default="None",
        choices=list(TextToImageGenerator.STYLE_PRESETS.keys()),
        help="Artistic style preset."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of denoising inference steps (default: 30)."
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="Classifier-Free Guidance (CFG) scale (default: 7.5)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512, must be divisible by 8)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512, must be divisible by 8)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)."
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm++_2m_karras",
        choices=list(TextToImageGenerator.SCHEDULER_MAP.keys()),
        help="Diffusion scheduler / sampler algorithm."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for saved images."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model repository ID or local path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Compute device (default: auto-detected)."
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable sequential CPU offload for low VRAM GPUs."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" AI Text-to-Image Generation ")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Style Preset: {args.style}")
    print(f"Steps: {args.steps} | CFG: {args.cfg} | Resolution: {args.width}x{args.height}")
    print("=" * 60)

    try:
        generator = TextToImageGenerator(
            model_id=args.model_id,
            device=args.device,
            scheduler_name=args.scheduler,
            cpu_offload=args.cpu_offload,
            output_dir=args.output_dir,
        )

        images, paths = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            style_preset=args.style,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            seed=args.seed,
            num_images=args.count,
            save_images=True,
            save_metadata=True,
        )

        print("\nGenerated Images:")
        for path in paths:
            print(f"  -> Saved: {os.path.abspath(path)}")
        print("\nGeneration finished successfully.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
