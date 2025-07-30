#!/usr/bin/env python3
"""
Profiling Analysis Script for FLUX.1-Kontext
Analyzes torch profiler traces and provides specific optimization recommendations
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re

def parse_chrome_trace(trace_file: Path) -> Dict:
    """Parse Chrome trace JSON file"""
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    return trace_data

def analyze_gpu_kernels(trace_data: Dict) -> List[Tuple[str, float, float]]:
    """Analyze GPU kernel performance from trace data"""
    kernels = []
    
    for event in trace_data.get('traceEvents', []):
        if event.get('cat') == 'kernel' and 'dur' in event:
            name = event.get('name', 'unknown')
            duration_us = event.get('dur', 0)
            duration_ms = duration_us / 1000.0
            ts = event.get('ts', 0)
            
            kernels.append((name, duration_ms, ts))
    
    # Sort by duration descending
    kernels.sort(key=lambda x: x[1], reverse=True)
    return kernels

def analyze_memory_operations(trace_data: Dict) -> List[Tuple[str, float]]:
    """Analyze memory operations from trace data"""
    memory_ops = []
    
    for event in trace_data.get('traceEvents', []):
        if 'memory' in event.get('name', '').lower() and 'dur' in event:
            name = event.get('name', 'unknown')
            duration_us = event.get('dur', 0)
            duration_ms = duration_us / 1000.0
            
            memory_ops.append((name, duration_ms))
    
    memory_ops.sort(key=lambda x: x[1], reverse=True)
    return memory_ops

def analyze_attention_patterns(trace_data: Dict) -> Dict:
    """Analyze attention operation patterns"""
    attention_ops = []
    
    for event in trace_data.get('traceEvents', []):
        name = event.get('name', '').lower()
        if any(keyword in name for keyword in ['attention', 'matmul', 'bmm', 'flash']):
            if 'dur' in event:
                duration_us = event.get('dur', 0)
                duration_ms = duration_us / 1000.0
                attention_ops.append((event.get('name', 'unknown'), duration_ms))
    
    attention_ops.sort(key=lambda x: x[1], reverse=True)
    
    # Categorize attention operations
    flash_attn_time = sum(dur for name, dur in attention_ops if 'flash' in name.lower())
    standard_attn_time = sum(dur for name, dur in attention_ops if 'flash' not in name.lower())
    
    return {
        'operations': attention_ops[:10],
        'flash_attention_time': flash_attn_time,
        'standard_attention_time': standard_attn_time,
        'total_attention_time': flash_attn_time + standard_attn_time
    }

def generate_optimization_recommendations(
    kernels: List[Tuple[str, float, float]], 
    memory_ops: List[Tuple[str, float]], 
    attention_analysis: Dict,
    total_time_ms: float
) -> List[str]:
    """Generate specific optimization recommendations based on profiling data"""
    recommendations = []
    
    # Analyze top kernels
    if kernels:
        top_kernel_time = sum(k[1] for k in kernels[:5])
        kernel_percentage = (top_kernel_time / total_time_ms) * 100
        
        recommendations.append(f"Top 5 GPU kernels consume {kernel_percentage:.1f}% of total time ({top_kernel_time:.1f}ms)")
        
        # Check for specific optimization opportunities
        for name, duration, _ in kernels[:10]:
            name_lower = name.lower()
            
            if 'gemm' in name_lower or 'matmul' in name_lower:
                if duration > 50:  # >50ms
                    recommendations.append(f"Large GEMM operation detected ({duration:.1f}ms) - consider mixed precision or quantization")
            
            if 'conv' in name_lower:
                if duration > 30:
                    recommendations.append(f"Slow convolution detected ({duration:.1f}ms) - consider optimized conv implementations")
            
            if 'softmax' in name_lower:
                if duration > 20:
                    recommendations.append(f"Slow softmax detected ({duration:.1f}ms) - FlashAttention could help")
            
            if 'copy' in name_lower or 'memcpy' in name_lower:
                if duration > 10:
                    recommendations.append(f"Memory copy overhead detected ({duration:.1f}ms) - reduce data movement")
    
    # Analyze attention patterns
    if attention_analysis['total_attention_time'] > 0:
        attn_percentage = (attention_analysis['total_attention_time'] / total_time_ms) * 100
        recommendations.append(f"Attention operations consume {attn_percentage:.1f}% of total time")
        
        if attention_analysis['standard_attention_time'] > attention_analysis['flash_attention_time']:
            recommendations.append("Standard attention is slower than FlashAttention - ensure FlashAttention is properly enabled")
        
        if attention_analysis['total_attention_time'] > 300:  # >300ms
            recommendations.append("High attention overhead - consider attention slicing or lower precision")
    
    # Memory operation analysis
    if memory_ops:
        total_memory_time = sum(op[1] for op in memory_ops)
        if total_memory_time > 100:  # >100ms
            memory_percentage = (total_memory_time / total_time_ms) * 100
            recommendations.append(f"Memory operations consume {memory_percentage:.1f}% of time - optimize data layout")
    
    # General recommendations based on total time
    if total_time_ms > 1000:  # >1s
        time_reduction_needed = total_time_ms - 1000
        recommendations.append(f"Need to reduce inference time by {time_reduction_needed:.0f}ms to achieve sub-1s")
        
        # Aggressive optimization suggestions
        recommendations.extend([
            "Consider reducing model resolution to 512x512 (4x speedup)",
            "Use fewer denoising steps (8-10 instead of 14+)",
            "Enable INT8 quantization for transformer blocks",
            "Use TensorRT compilation for maximum optimization",
            "Consider model pruning or distillation",
            "Use attention slicing: pipe.enable_attention_slicing(1)",
            "Try xFormers optimized attention",
            "Profile individual transformer layers for targeted optimization"
        ])
    
    return recommendations

def analyze_profiling_directory(profiling_dir: Path) -> Dict:
    """Analyze all profiling files in a directory"""
    results = {
        'timing_analysis': None,
        'memory_analysis': None,
        'gpu_kernels': [],
        'memory_operations': [],
        'attention_analysis': {},
        'recommendations': []
    }
    
    # Read timing analysis
    timing_file = profiling_dir / "timing_analysis.txt"
    if timing_file.exists():
        with open(timing_file, 'r') as f:
            results['timing_analysis'] = f.read()
    
    # Read memory analysis
    memory_file = profiling_dir / "memory_analysis.txt"
    if memory_file.exists():
        with open(memory_file, 'r') as f:
            results['memory_analysis'] = f.read()
    
    # Analyze trace files
    trace_files = list(profiling_dir.glob("*_trace.json"))
    total_time_ms = 0
    
    for trace_file in trace_files:
        try:
            trace_data = parse_chrome_trace(trace_file)
            
            # Extract total time from trace
            if trace_data.get('traceEvents'):
                max_ts = max(event.get('ts', 0) + event.get('dur', 0) for event in trace_data['traceEvents'] if 'ts' in event)
                min_ts = min(event.get('ts', 0) for event in trace_data['traceEvents'] if 'ts' in event)
                total_time_ms = max(total_time_ms, (max_ts - min_ts) / 1000.0)
            
            kernels = analyze_gpu_kernels(trace_data)
            memory_ops = analyze_memory_operations(trace_data)
            attention_analysis = analyze_attention_patterns(trace_data)
            
            results['gpu_kernels'].extend(kernels)
            results['memory_operations'].extend(memory_ops)
            
            # Merge attention analysis
            if not results['attention_analysis']:
                results['attention_analysis'] = attention_analysis
            else:
                results['attention_analysis']['operations'].extend(attention_analysis['operations'])
                results['attention_analysis']['flash_attention_time'] += attention_analysis['flash_attention_time']
                results['attention_analysis']['standard_attention_time'] += attention_analysis['standard_attention_time']
                results['attention_analysis']['total_attention_time'] += attention_analysis['total_attention_time']
        
        except Exception as e:
            print(f"Error analyzing {trace_file}: {e}")
    
    # Sort combined results
    results['gpu_kernels'].sort(key=lambda x: x[1], reverse=True)
    results['memory_operations'].sort(key=lambda x: x[1], reverse=True)
    
    # Generate recommendations
    results['recommendations'] = generate_optimization_recommendations(
        results['gpu_kernels'],
        results['memory_operations'],
        results['attention_analysis'],
        total_time_ms
    )
    
    results['total_time_ms'] = total_time_ms
    
    return results

def print_analysis_report(results: Dict):
    """Print comprehensive analysis report"""
    print("="*80)
    print("FLUX.1-Kontext Profiling Analysis Report")
    print("="*80)
    
    if results['total_time_ms'] > 0:
        print(f"\nTotal Inference Time: {results['total_time_ms']:.1f}ms ({results['total_time_ms']/1000:.3f}s)")
        if results['total_time_ms'] > 1000:
            print(f"❌ Sub-1s target missed by: {(results['total_time_ms'] - 1000):.0f}ms")
        else:
            print(f"✅ Sub-1s target achieved!")
    
    # Top GPU kernels
    if results['gpu_kernels']:
        print(f"\n🔥 TOP GPU BOTTLENECKS:")
        print("-" * 60)
        for i, (name, duration, _) in enumerate(results['gpu_kernels'][:10], 1):
            percentage = (duration / results['total_time_ms']) * 100 if results['total_time_ms'] > 0 else 0
            print(f"{i:2d}. {name[:45]:45} | {duration:7.1f}ms | {percentage:5.1f}%")
    
    # Attention analysis
    if results['attention_analysis'].get('total_attention_time', 0) > 0:
        attn = results['attention_analysis']
        attn_percentage = (attn['total_attention_time'] / results['total_time_ms']) * 100 if results['total_time_ms'] > 0 else 0
        print(f"\n🧠 ATTENTION ANALYSIS:")
        print("-" * 40)
        print(f"Total attention time: {attn['total_attention_time']:.1f}ms ({attn_percentage:.1f}%)")
        print(f"FlashAttention time:  {attn['flash_attention_time']:.1f}ms")
        print(f"Standard attn time:   {attn['standard_attention_time']:.1f}ms")
        
        if attn['standard_attention_time'] > attn['flash_attention_time']:
            print("⚠️  Standard attention is dominating - FlashAttention not fully utilized")
    
    # Memory operations
    if results['memory_operations']:
        total_memory_time = sum(op[1] for op in results['memory_operations'][:5])
        memory_percentage = (total_memory_time / results['total_time_ms']) * 100 if results['total_time_ms'] > 0 else 0
        print(f"\n💾 MEMORY BOTTLENECKS:")
        print("-" * 50)
        print(f"Top 5 memory ops: {total_memory_time:.1f}ms ({memory_percentage:.1f}%)")
        for i, (name, duration) in enumerate(results['memory_operations'][:5], 1):
            print(f"{i}. {name[:40]:40} | {duration:7.1f}ms")
    
    # Optimization recommendations
    if results['recommendations']:
        print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 50)
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i:2d}. {rec}")
    
    # Quick wins for sub-1s
    print(f"\n⚡ QUICK WINS FOR SUB-1S INFERENCE:")
    print("-" * 40)
    print("1. Reduce resolution: 1024x1024 → 512x512 (4x speedup)")
    print("2. Fewer steps: 14+ → 8-10 steps with DPM-Solver++")
    print("3. Enable TensorRT: torch.compile(..., backend='tensorrt')")
    print("4. Use INT8 quantization for transformer")
    print("5. Enable attention slicing: pipe.enable_attention_slicing()")
    print("6. Use xFormers: pip install xformers")

def main():
    parser = argparse.ArgumentParser(description="Analyze FLUX profiling results")
    parser.add_argument("profiling_dir", type=str, help="Path to profiling results directory")
    parser.add_argument("--export", type=str, help="Export detailed report to file")
    
    args = parser.parse_args()
    
    profiling_dir = Path(args.profiling_dir)
    if not profiling_dir.exists():
        print(f"Error: Profiling directory {profiling_dir} does not exist")
        return
    
    print(f"Analyzing profiling results from: {profiling_dir}")
    results = analyze_profiling_directory(profiling_dir)
    
    print_analysis_report(results)
    
    if args.export:
        export_path = Path(args.export)
        with open(export_path, 'w') as f:
            f.write("FLUX.1-Kontext Profiling Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Inference Time: {results['total_time_ms']:.1f}ms\n\n")
            
            f.write("Top GPU Kernels:\n")
            for name, duration, _ in results['gpu_kernels'][:20]:
                percentage = (duration / results['total_time_ms']) * 100 if results['total_time_ms'] > 0 else 0
                f.write(f"{name:50} | {duration:7.1f}ms | {percentage:5.1f}%\n")
            
            f.write("\nOptimization Recommendations:\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"\nDetailed report exported to: {export_path}")

if __name__ == "__main__":
    main() 