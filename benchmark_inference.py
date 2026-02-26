"""
benchmark_inference.py -- Cold Start & Inference Latency Benchmark

Measures ML model performance independently from protocol overhead.
Tests model loading time, cold start inference, and warm inference latency.

This benchmark answers: "How fast is the model itself?"

Benchmarks:
  1. Model loading time (.pth weights vs TorchScript .pt)
  2. Cold start inference (first prediction after load)
  3. Warm inference latency (avg, P50, P95, P99)
  4. Memory usage (model size on disk and in RAM)
  5. Preprocessing time (image decode + transform)

Usage:
  1. Place model files in models/ (from Google Colab training)
  2. Run:  python benchmark_inference.py
  3. Options:
       python benchmark_inference.py --iterations 100
       python benchmark_inference.py --export
       python benchmark_inference.py --images-dir test_images
"""

import json
import time
import statistics
import argparse
import os
import io
import gc
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import psutil


# =============================================================================
# CONFIG
# =============================================================================

MODELS_DIR = Path(__file__).parent / "models"
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
RESULTS_FILE = "benchmark_results_inference.json"

PTH_PATH = MODELS_DIR / "food_classifier.pth"
TRACED_PATH = MODELS_DIR / "food_classifier_traced.pt"
CLASSES_PATH = MODELS_DIR / "food_classes.json"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LoadResult:
    """Results for model loading benchmark."""
    format_name: str
    load_time_ms: float = 0.0
    file_size_mb: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_used_mb: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class InferenceResult:
    """Results for inference benchmark."""
    format_name: str
    cold_start_ms: float = 0.0
    preprocess_times_ms: List[float] = field(default_factory=list)
    inference_times_ms: List[float] = field(default_factory=list)
    total_times_ms: List[float] = field(default_factory=list)

    @property
    def avg_preprocess(self) -> float:
        return statistics.mean(self.preprocess_times_ms) if self.preprocess_times_ms else 0

    @property
    def avg_inference(self) -> float:
        return statistics.mean(self.inference_times_ms) if self.inference_times_ms else 0

    @property
    def avg_total(self) -> float:
        return statistics.mean(self.total_times_ms) if self.total_times_ms else 0

    @property
    def p50_inference(self) -> float:
        if not self.inference_times_ms:
            return 0
        return statistics.median(self.inference_times_ms)

    @property
    def p95_inference(self) -> float:
        if not self.inference_times_ms:
            return 0
        s = sorted(self.inference_times_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def p99_inference(self) -> float:
        if not self.inference_times_ms:
            return 0
        s = sorted(self.inference_times_ms)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def p50_total(self) -> float:
        if not self.total_times_ms:
            return 0
        return statistics.median(self.total_times_ms)

    @property
    def p95_total(self) -> float:
        if not self.total_times_ms:
            return 0
        s = sorted(self.total_times_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def p99_total(self) -> float:
        if not self.total_times_ms:
            return 0
        s = sorted(self.total_times_ms)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def throughput(self) -> float:
        total_s = sum(self.total_times_ms) / 1000.0
        return len(self.total_times_ms) / total_s if total_s > 0 else 0


# =============================================================================
# HELPERS
# =============================================================================

def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def get_transform():
    """Image preprocessing pipeline (must match training transforms)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_class_info() -> tuple:
    """Load class names and count from food_classes.json."""
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH) as f:
            info = json.load(f)
        return info.get("classes", []), info.get("num_classes", 101)
    return [], 101


def generate_test_images(count: int = 10) -> List[bytes]:
    """Generate synthetic test images."""
    images = []
    for i in range(count):
        np.random.seed(42 + i)
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        images.append(buf.getvalue())
    return images


def load_test_images() -> List[bytes]:
    """Load real test images or generate synthetic ones."""
    images = []
    if TEST_IMAGES_DIR.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in TEST_IMAGES_DIR.glob(ext):
                images.append(path.read_bytes())

    if images:
        print(f"    Loaded {len(images)} test images from {TEST_IMAGES_DIR}/")
        return images

    print(f"    No test images found -- generating 10 synthetic 224x224 JPEGs")
    return generate_test_images(10)


# =============================================================================
# MODEL LOADING BENCHMARK
# =============================================================================

def benchmark_load_pth(num_classes: int) -> LoadResult:
    """Benchmark loading .pth weights file."""
    result = LoadResult(format_name=".pth (weights)")

    if not PTH_PATH.exists():
        result.error = f"File not found: {PTH_PATH}"
        return result

    result.file_size_mb = get_file_size_mb(PTH_PATH)

    # Force garbage collection before measuring
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    result.memory_before_mb = get_memory_mb()

    start = time.perf_counter()
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, num_classes),
        )
        model.load_state_dict(torch.load(PTH_PATH, map_location="cpu", weights_only=True))
        model.eval()
        result.load_time_ms = (time.perf_counter() - start) * 1000
        result.memory_after_mb = get_memory_mb()
        result.memory_used_mb = result.memory_after_mb - result.memory_before_mb
        result.success = True

        # Clean up
        del model
        gc.collect()
    except Exception as e:
        result.error = str(e)

    return result


def benchmark_load_traced() -> LoadResult:
    """Benchmark loading TorchScript .pt model."""
    result = LoadResult(format_name=".pt (TorchScript)")

    if not TRACED_PATH.exists():
        result.error = f"File not found: {TRACED_PATH}"
        return result

    result.file_size_mb = get_file_size_mb(TRACED_PATH)

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    result.memory_before_mb = get_memory_mb()

    start = time.perf_counter()
    try:
        model = torch.jit.load(str(TRACED_PATH), map_location="cpu")
        model.eval()
        result.load_time_ms = (time.perf_counter() - start) * 1000
        result.memory_after_mb = get_memory_mb()
        result.memory_used_mb = result.memory_after_mb - result.memory_before_mb
        result.success = True

        del model
        gc.collect()
    except Exception as e:
        result.error = str(e)

    return result


# =============================================================================
# INFERENCE BENCHMARK
# =============================================================================

def benchmark_inference(
    model,
    images: List[bytes],
    transform,
    iterations: int,
    warmup: int,
    format_name: str,
) -> InferenceResult:
    """
    Benchmark inference: cold start + warm iterations.

    The first inference (cold start) is measured separately because
    it includes JIT compilation, memory allocation, and cache filling.
    """
    result = InferenceResult(format_name=format_name)
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cpu")

    # -- Cold Start (first inference ever after loading) --
    img_bytes = images[0]
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    cold_start = time.perf_counter()
    with torch.no_grad():
        _ = model(tensor)
    result.cold_start_ms = (time.perf_counter() - cold_start) * 1000

    # -- Warmup + Measured Iterations --
    for i in range(warmup + iterations):
        img_bytes = images[i % len(images)]

        # Measure preprocessing
        pre_start = time.perf_counter()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        pre_ms = (time.perf_counter() - pre_start) * 1000

        # Measure inference only
        inf_start = time.perf_counter()
        with torch.no_grad():
            output = model(tensor)
            _ = torch.softmax(output, dim=1)
        inf_ms = (time.perf_counter() - inf_start) * 1000

        total_ms = pre_ms + inf_ms

        if i >= warmup:
            result.preprocess_times_ms.append(pre_ms)
            result.inference_times_ms.append(inf_ms)
            result.total_times_ms.append(total_ms)

    return result


# =============================================================================
# LOAD AND BENCHMARK A MODEL FORMAT
# =============================================================================

def run_format_benchmark(
    format_name: str,
    load_fn,
    images: List[bytes],
    transform,
    iterations: int,
    warmup: int,
    num_classes: int,
) -> tuple:
    """Load a model and benchmark its inference. Returns (LoadResult, InferenceResult or None)."""

    # Load benchmark
    if "traced" in format_name.lower() or "torchscript" in format_name.lower():
        load_result = load_fn()
    else:
        load_result = load_fn(num_classes)

    if not load_result.success:
        print(f"    [SKIP] {format_name}: {load_result.error}")
        return load_result, None

    # Now load model again for inference benchmark (need to keep it alive)
    gc.collect()
    if "traced" in format_name.lower() or "torchscript" in format_name.lower():
        model = torch.jit.load(str(TRACED_PATH), map_location="cpu")
    else:
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, num_classes),
        )
        model.load_state_dict(torch.load(PTH_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # Inference benchmark
    inf_result = benchmark_inference(
        model, images, transform, iterations, warmup, format_name
    )

    del model
    gc.collect()

    return load_result, inf_result


# =============================================================================
# REPORT
# =============================================================================

def print_report(
    load_results: List[LoadResult],
    inf_results: List[InferenceResult],
    images: List[bytes],
    iterations: int,
):
    """Print formatted benchmark report."""
    avg_img_kb = statistics.mean(len(img) for img in images) / 1024

    print(f"\n{'='*80}")
    print(f"  ML Inference Benchmark -- Cold Start & Latency")
    print(f"{'='*80}")
    print(f"  Device:      CPU ({torch.get_num_threads()} threads)")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)} (not used in this benchmark)")
    print(f"  Iterations:  {iterations}")
    print(f"  Test images: {len(images)} (avg {avg_img_kb:.1f} KB)")
    print(f"  PyTorch:     {torch.__version__}")

    # -- Model Loading --
    print(f"\n  {'-'*70}")
    print(f"  MODEL LOADING")
    print(f"  {'-'*70}")

    successful_loads = [r for r in load_results if r.success]
    for r in load_results:
        status = "OK" if r.success else "N/A"
        print(f"    [{status}] {r.format_name:25s}", end="")
        if r.success:
            print(f"  {r.load_time_ms:8.1f} ms  |  {r.file_size_mb:.1f} MB on disk  |  ~{r.memory_used_mb:.1f} MB in RAM")
        else:
            print(f"  {r.error}")

    if len(successful_loads) == 2:
        faster = min(successful_loads, key=lambda r: r.load_time_ms)
        slower = max(successful_loads, key=lambda r: r.load_time_ms)
        speedup = slower.load_time_ms / faster.load_time_ms
        print(f"\n    Winner: {faster.format_name} ({speedup:.1f}x faster load)")

    # -- Inference --
    valid_inf = [r for r in inf_results if r is not None]
    if not valid_inf:
        print(f"\n  No inference results available.")
        print(f"{'='*80}\n")
        return

    for r in valid_inf:
        print(f"\n  {'-'*70}")
        print(f"  INFERENCE -- {r.format_name}")
        print(f"  {'-'*70}")
        print(f"    Cold start (1st):     {r.cold_start_ms:8.2f} ms")
        print(f"    Preprocessing avg:    {r.avg_preprocess:8.2f} ms")
        print(f"    Inference avg:        {r.avg_inference:8.2f} ms")
        print(f"    Total avg (pre+inf):  {r.avg_total:8.2f} ms")
        print(f"    P50 (inference):      {r.p50_inference:8.2f} ms")
        print(f"    P95 (inference):      {r.p95_inference:8.2f} ms")
        print(f"    P99 (inference):      {r.p99_inference:8.2f} ms")
        print(f"    Throughput:           {r.throughput:8.1f} images/sec")

    # -- Comparison --
    if len(valid_inf) >= 2:
        print(f"\n  {'-'*70}")
        print(f"  COMPARISON")
        print(f"  {'-'*70}")

        hdr = "    {:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}"
        print(hdr.format("Format", "Cold(ms)", "Avg(ms)", "P50(ms)", "P95(ms)", "Img/s"))
        print(f"    {'-'*75}")

        for r in valid_inf:
            print(hdr.format(
                r.format_name,
                f"{r.cold_start_ms:.1f}",
                f"{r.avg_inference:.2f}",
                f"{r.p50_inference:.2f}",
                f"{r.p95_inference:.2f}",
                f"{r.throughput:.1f}",
            ))

        faster = min(valid_inf, key=lambda r: r.avg_inference)
        slower = max(valid_inf, key=lambda r: r.avg_inference)
        speedup = slower.avg_inference / faster.avg_inference if faster.avg_inference > 0 else 0
        print(f"\n    Winner: {faster.format_name} ({speedup:.1f}x faster inference)")

    print(f"\n{'='*80}\n")


# =============================================================================
# EXPORT
# =============================================================================

def export_json(
    load_results: List[LoadResult],
    inf_results: List[InferenceResult],
    images: List[bytes],
    iterations: int,
    filename: str = RESULTS_FILE,
):
    """Export results to JSON."""
    data = {
        "benchmark": "ML Inference -- Cold Start & Latency",
        "device": "cpu",
        "pytorch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "iterations": iterations,
        "num_test_images": len(images),
        "avg_image_size_bytes": round(statistics.mean(len(img) for img in images)),
        "loading": [],
        "inference": [],
    }

    for r in load_results:
        data["loading"].append({
            "format": r.format_name,
            "load_time_ms": round(r.load_time_ms, 2),
            "file_size_mb": round(r.file_size_mb, 2),
            "memory_used_mb": round(r.memory_used_mb, 2),
            "success": r.success,
            "error": r.error or None,
        })

    for r in inf_results:
        if r is None:
            continue
        data["inference"].append({
            "format": r.format_name,
            "cold_start_ms": round(r.cold_start_ms, 2),
            "avg_preprocess_ms": round(r.avg_preprocess, 2),
            "avg_inference_ms": round(r.avg_inference, 2),
            "avg_total_ms": round(r.avg_total, 2),
            "p50_inference_ms": round(r.p50_inference, 2),
            "p95_inference_ms": round(r.p95_inference, 2),
            "p99_inference_ms": round(r.p99_inference, 2),
            "throughput_images_sec": round(r.throughput, 2),
            "iterations": len(r.inference_times_ms),
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML Inference Benchmark -- Cold Start & Latency"
    )
    parser.add_argument("--iterations", "-n", type=int, default=100,
                        help="Inference iterations per model format (default: 100)")
    parser.add_argument("--warmup", "-w", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--export", action="store_true",
                        help="Export results to benchmark_results_inference.json")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory containing test images")
    args = parser.parse_args()

    if args.images_dir:
        global TEST_IMAGES_DIR
        TEST_IMAGES_DIR = Path(args.images_dir)

    print(f"\n{'='*60}")
    print(f"  ML Inference Benchmark -- Cold Start & Latency")
    print(f"{'='*60}")

    # Check model files
    print(f"\n  Checking model files...")
    pth_exists = PTH_PATH.exists()
    traced_exists = TRACED_PATH.exists()
    classes_exist = CLASSES_PATH.exists()

    print(f"    .pth weights:    {'FOUND' if pth_exists else 'NOT FOUND'} ({PTH_PATH})")
    print(f"    TorchScript .pt: {'FOUND' if traced_exists else 'NOT FOUND'} ({TRACED_PATH})")
    print(f"    Classes JSON:    {'FOUND' if classes_exist else 'NOT FOUND'} ({CLASSES_PATH})")

    if not pth_exists and not traced_exists:
        print(f"\n  [ERROR] No model files found in {MODELS_DIR}/")
        print(f"  Train the model on Google Colab and place files here:")
        print(f"    - food_classifier.pth")
        print(f"    - food_classifier_traced.pt")
        print(f"    - food_classes.json")
        return

    # Load class info
    class_names, num_classes = load_class_info()
    print(f"    Classes:         {num_classes}")

    # Load test images
    print(f"\n  Loading test images...")
    images = load_test_images()

    # Setup
    transform = get_transform()
    load_results = []
    inf_results = []

    # -- Benchmark .pth format --
    if pth_exists:
        print(f"\n  [1/2] Benchmarking .pth (weights) format...")
        load_r, inf_r = run_format_benchmark(
            ".pth (weights)", benchmark_load_pth,
            images, transform, args.iterations, args.warmup, num_classes,
        )
        load_results.append(load_r)
        inf_results.append(inf_r)
    else:
        print(f"\n  [1/2] Skipping .pth -- file not found")

    # -- Benchmark TorchScript format --
    if traced_exists:
        print(f"  [2/2] Benchmarking TorchScript (.pt) format...")
        load_r, inf_r = run_format_benchmark(
            "TorchScript (.pt)", benchmark_load_traced,
            images, transform, args.iterations, args.warmup, num_classes,
        )
        load_results.append(load_r)
        inf_results.append(inf_r)
    else:
        print(f"  [2/2] Skipping TorchScript -- file not found")

    # Print report
    print_report(load_results, inf_results, images, args.iterations)

    # Export
    if args.export:
        export_json(load_results, inf_results, images, args.iterations)


if __name__ == "__main__":
    main()
