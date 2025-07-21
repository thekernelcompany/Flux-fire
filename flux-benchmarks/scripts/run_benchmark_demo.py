#!/usr/bin/env python3
"""
Demo script to run a quick benchmark test for blog post data collection
This provides a minimal working example that can be completed in ~30 minutes
"""

import os
import sys
import torch
from pathlib import Path
import json
import time

def check_environment():
    """Check if environment is properly set up"""
    issues = []
    
    # Check GPU
    if not torch.cuda.is_available():
        issues.append("No CUDA GPU available")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU detected: {gpu_name} with {vram:.1f}GB VRAM")
        
        if vram < 24:
            issues.append(f"GPU has only {vram:.1f}GB VRAM. FLUX requires 24GB+")
    
    # Check HF token
    if not os.environ.get("HF_TOKEN"):
        issues.append("HF_TOKEN environment variable not set")
    else:
        print("✓ HuggingFace token found")
    
    # Check dependencies
    try:
        import pandas
        import matplotlib
        import seaborn
        print("✓ Data visualization packages installed")
    except ImportError as e:
        issues.append(f"Missing package: {e.name}")
    
    if issues:
        print("\n❌ Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✅ Environment check passed!")
    return True

def run_minimal_benchmark():
    """Run a minimal benchmark for quick results"""
    print("\n=== Running Minimal Benchmark Demo ===")
    print("This will take approximately 20-30 minutes\n")
    
    from flux_kontext_standalone import FluxService, InferenceRequest
    
    # Test configuration
    test_prompt = "A majestic mountain landscape at sunset with vibrant colors"
    test_configs = [
        {"name": "512x512", "width": 512, "height": 512, "steps": 20},
        {"name": "1024x1024", "width": 1024, "height": 1024, "steps": 20},
        {"name": "1024x1024_highsteps", "width": 1024, "height": 1024, "steps": 28},
    ]
    
    results = {
        "baseline": {},
        "optimized": {},
        "metadata": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": test_prompt
        }
    }
    
    # Test 1: Baseline (no optimizations)
    print("Testing baseline performance...")
    service_baseline = FluxService()
    
    # Override to disable optimizations
    service_baseline._optimize = lambda: None
    service_baseline._compile = lambda: None
    service_baseline._load_mega_cache = lambda: None
    service_baseline._save_mega_cache = lambda: None
    
    service_baseline.load_model()
    
    for config in test_configs:
        print(f"  - Testing {config['name']}...")
        request = InferenceRequest(
            prompt=test_prompt,
            width=config['width'],
            height=config['height'],
            steps=config['steps'],
            seed=42
        )
        
        # Warmup
        _ = service_baseline.inference(request)
        
        # Actual measurement (3 runs)
        times = []
        for i in range(3):
            result = service_baseline.inference(request)
            times.append(result['inference_time'])
            print(f"    Run {i+1}: {result['inference_time']:.2f}s")
        
        avg_time = sum(times) / len(times)
        results['baseline'][config['name']] = {
            'times': times,
            'average': avg_time,
            'config': config
        }
    
    # Clean up
    del service_baseline
    torch.cuda.empty_cache()
    
    # Test 2: Fully optimized
    print("\nTesting optimized performance...")
    service_optimized = FluxService()
    service_optimized.load_model()
    
    for config in test_configs:
        print(f"  - Testing {config['name']}...")
        request = InferenceRequest(
            prompt=test_prompt,
            width=config['width'],
            height=config['height'],
            steps=config['steps'],
            seed=42
        )
        
        # Warmup
        _ = service_optimized.inference(request)
        
        # Actual measurement (3 runs)
        times = []
        for i in range(3):
            result = service_optimized.inference(request)
            times.append(result['inference_time'])
            print(f"    Run {i+1}: {result['inference_time']:.2f}s")
        
        avg_time = sum(times) / len(times)
        results['optimized'][config['name']] = {
            'times': times,
            'average': avg_time,
            'config': config,
            'image_path': result['images'][0]['filepath']  # Save last image for quality check
        }
    
    # Calculate speedups
    print("\n=== RESULTS SUMMARY ===")
    for config_name in results['baseline'].keys():
        baseline_time = results['baseline'][config_name]['average']
        optimized_time = results['optimized'][config_name]['average']
        speedup = baseline_time / optimized_time
        
        print(f"\n{config_name}:")
        print(f"  Baseline:  {baseline_time:.2f}s")
        print(f"  Optimized: {optimized_time:.2f}s")
        print(f"  Speedup:   {speedup:.2f}x")
    
    # Save results
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/benchmark_results.json")
    
    # Generate simple visualization
    try:
        import matplotlib.pyplot as plt
        
        configs = list(results['baseline'].keys())
        baseline_times = [results['baseline'][c]['average'] for c in configs]
        optimized_times = [results['optimized'][c]['average'] for c in configs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(configs))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], baseline_times, width, label='Baseline', color='#FF6B6B')
        bars2 = ax.bar([i + width/2 for i in x], optimized_times, width, label='Optimized', color='#4ECDC4')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Inference Time (seconds)')
        ax.set_title('FLUX.1-Kontext Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=150)
        print(f"Chart saved to {output_dir}/performance_comparison.png")
    except Exception as e:
        print(f"Could not generate chart: {e}")
    
    return results

def generate_blog_metrics(results):
    """Generate key metrics for blog post"""
    # Focus on 1024x1024 standard config
    standard_config = "1024x1024"
    
    baseline_time = results['baseline'][standard_config]['average']
    optimized_time = results['optimized'][standard_config]['average']
    speedup = baseline_time / optimized_time
    
    print("\n=== BLOG POST METRICS ===")
    print(f"\n📊 Headline Numbers:")
    print(f"- Speedup achieved: {speedup:.2f}x faster")
    print(f"- Time per image: {baseline_time:.1f}s → {optimized_time:.1f}s")
    print(f"- Time saved: {baseline_time - optimized_time:.1f} seconds per image")
    
    # Calculate daily savings
    images_per_day = 10000  # Medium business scenario
    daily_time_saved = (baseline_time - optimized_time) * images_per_day / 3600
    print(f"- Daily time saved: {daily_time_saved:.1f} GPU-hours")
    
    # Cost savings (assuming $3/hour GPU)
    daily_cost_saved = daily_time_saved * 3
    monthly_cost_saved = daily_cost_saved * 30
    print(f"- Monthly cost saved: ${monthly_cost_saved:,.0f}")
    
    print("\n📈 For different workloads:")
    for scenario, daily_images in [("Hobby", 100), ("Small Biz", 1000), ("Enterprise", 100000)]:
        time_saved = (baseline_time - optimized_time) * daily_images / 3600
        cost_saved = time_saved * 3 * 30  # Monthly
        print(f"- {scenario} ({daily_images:,}/day): ${cost_saved:,.0f}/month saved")

def main():
    """Main demo execution"""
    print("FLUX.1-Kontext Benchmark Demo")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\nPlease fix environment issues before running benchmark.")
        sys.exit(1)
    
    # Run benchmark
    try:
        results = run_minimal_benchmark()
        generate_blog_metrics(results)
        
        print("\n✅ Demo complete! You now have:")
        print("- Performance comparison data")
        print("- Sample images from both versions")
        print("- Key metrics for your blog post")
        print("\nNext steps:")
        print("1. Run full benchmark suite for complete data")
        print("2. Run quality assessment on generated images")
        print("3. Generate full visualization suite")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()