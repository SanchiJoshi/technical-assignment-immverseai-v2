"""CLI Inference Script for Manuscript Specific Layout Region Detection.

Usage:
    python inference.py --input ./data/test_images --output ./results
    python inference.py --input ./data/test_images/pic_1.png --output ./results --conf 0.50
"""

import argparse
import os
import sys
import glob
import time
from typing import List

# Ensure src is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import ManuscriptLayoutPipeline


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def collect_images(input_path: str) -> List[str]:
    """Finds all valid image files from a file path or directory.

    Args:
        input_path: Path to a single image file or a directory.

    Returns:
        List of absolute or relative image paths.
    """
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return [input_path]
        else:
            print(f"[!] Warning: File {input_path} is not a supported image format.")
            return []

    elif os.path.isdir(input_path):
        collected = []
        for ext in SUPPORTED_EXTENSIONS:
            collected.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            collected.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(list(set(collected)))
    else:
        print(f"[!] Error: Input path '{input_path}' does not exist.")
        return []


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="ImmverseAI Technical Assignment: Manuscript Specific Layout Region Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to an input manuscript image or a folder containing manuscript images."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Destination directory to save annotated images and JSON prediction metadata."
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.50,
        help="Confidence threshold for region classification (0.0 to 1.0)."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference compute device ('cpu' or 'cuda')."
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    print("=" * 70)
    print("  MANUSCRIPT SPECIFIC LAYOUT REGION DETECTION PIPELINE")
    print("  ImmverseAI Technical Evaluation")
    print("=" * 70)

    # 1. Collect inputs
    image_paths = collect_images(args.input)
    if not image_paths:
        print(f"[!] No valid images found at: {args.input}")
        sys.exit(1)

    print(f"[*] Found {len(image_paths)} image(s) to process.")
    print(f"[*] Output destination: {os.path.abspath(args.output)}")
    print(f"[*] Confidence threshold: {args.conf}")
    print(f"[*] Compute device: {args.device}\n")

    # 2. Initialize pipeline
    print("[*] Initializing hybrid layout detection engine...")
    try:
        pipeline = ManuscriptLayoutPipeline(
            confidence_threshold=args.conf,
            device=args.device
        )
    except Exception as e:
        print(f"[!] Error initializing pipeline: {e}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 3. Execute Batch Inference
    total_start = time.perf_counter()
    processed_count = 0
    total_regions_found = 0

    print("-" * 70)
    for idx, img_path in enumerate(image_paths, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(args.output, f"{base_name}_annotated.png")
        out_json_path = os.path.join(args.output, f"{base_name}_predictions.json")

        print(f"[{idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)} ... ", end="", flush=True)

        try:
            res = pipeline.process_image(
                image_path=img_path,
                save_annotated_path=out_img_path,
                save_json_path=out_json_path
            )

            num_reg = res["summary"]["total_regions_detected"]
            latency = res["performance"]["processing_time_ms"]
            substrate = res["image_metadata"]["substrate_type"]
            total_regions_found += num_reg
            processed_count += 1

            print(f"DONE ({latency:.1f}ms) | Substrate: {substrate} | Regions: {num_reg}")
            
            # Print class breakdown
            counts = res["summary"]["class_distribution"]
            breakdown = ", ".join([f"{k}: {v}" for k, v in counts.items() if v > 0])
            print(f"      |-- Breakdown: {breakdown}")

        except Exception as e:
            print(f"FAILED: {e}")

    total_time = (time.perf_counter() - total_start) * 1000.0
    print("-" * 70)
    print(f"[OK] Batch processing complete!")
    print(f"[*] Total images processed: {processed_count}/{len(image_paths)}")
    print(f"[*] Total layout regions localized: {total_regions_found}")
    print(f"[*] Total elapsed time: {total_time:.2f}ms (Avg {total_time/max(1, processed_count):.2f}ms/page)")
    print(f"[*] Outputs saved to: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
