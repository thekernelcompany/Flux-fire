#!/usr/bin/env python3
"""Extract key metrics from benchmark results for blog post"""

import json
import pandas as pd

# Load results
with open('benchmark_results/benchmark_results.json', 'r') as f:
    data = json.load(f)

# Get summary data
df = pd.DataFrame(data['summary'])

# Find baseline and optimized for 1024x1024 with 20 steps
print('=== KEY BENCHMARK METRICS ===\n')

# Helper to find matching rows
def find_matching(df, opt_level, res_w, res_h, steps):
    matches = []
    for idx, row in df.iterrows():
        if (row['optimization_level'] == opt_level and 
            str(row['steps']) == str(steps) and
            str(row['batch_size']) == '1'):
            res = row['resolution']
            if isinstance(res, list) and len(res) == 2:
                if res[0] == res_w and res[1] == res_h:
                    matches.append(row)
    return matches

# Get baseline and optimized times
baseline_matches = find_matching(df, 'baseline', 1024, 1024, 20)
optimized_matches = find_matching(df, 'full_optimization', 1024, 1024, 20)

if baseline_matches and optimized_matches:
    baseline_time = baseline_matches[0]['inference_time_mean']
    optimized_time = optimized_matches[0]['inference_time_mean']
    speedup = baseline_time / optimized_time
    
    print(f'Resolution: 1024x1024 @ 20 steps')
    print(f'Baseline time: {baseline_time:.2f}s')
    print(f'Optimized time: {optimized_time:.2f}s')
    print(f'SPEEDUP: {speedup:.2f}x')
    print(f'Time saved per image: {baseline_time - optimized_time:.2f}s')
    
    # Cost calculations
    print(f'\n=== COST IMPACT (10,000 images/day) ===')
    daily_time_saved = (baseline_time - optimized_time) * 10000 / 3600
    print(f'GPU hours saved per day: {daily_time_saved:.1f}')
    print(f'Monthly savings (@ $3/GPU-hr): ${daily_time_saved * 30 * 3:,.0f}')
    
# Check different resolutions
print('\n=== PERFORMANCE ACROSS RESOLUTIONS ===')
resolutions = [(512, 512), (768, 768), (1024, 1024), (1280, 720)]
for w, h in resolutions:
    opt_matches = find_matching(df, 'full_optimization', w, h, 20)
    if opt_matches:
        print(f'{w}x{h}: {opt_matches[0]["inference_time_mean"]:.2f}s')

# Check optimization breakdown
print('\n=== OPTIMIZATION BREAKDOWN ===')
opt_levels = ['baseline', 'compile_only', 'para_attn_only', 'standard_opts', 'full_optimization']
for opt in opt_levels:
    matches = find_matching(df, opt, 1024, 1024, 20)
    if matches:
        time = matches[0]['inference_time_mean']
        speedup_vs_base = baseline_time / time if 'baseline_time' in locals() else 1.0
        print(f'{opt:20s}: {time:.2f}s ({speedup_vs_base:.2f}x)')

# Memory usage
print('\n=== MEMORY USAGE ===')
for opt in ['baseline', 'full_optimization']:
    matches = find_matching(df, opt, 1024, 1024, 20)
    if matches:
        mem = matches[0]['memory_used_mean']
        print(f'{opt}: {mem:.1f}GB')

# Create blog post snippet
print('\n=== BLOG POST SNIPPET ===')
if 'speedup' in locals():
    print(f"""
Our optimizations achieved a **{speedup:.2f}x speedup** on the FLUX.1-Kontext model:

- Baseline: {baseline_time:.2f} seconds per image
- Optimized: {optimized_time:.2f} seconds per image
- Time saved: {baseline_time - optimized_time:.2f} seconds ({(1 - optimized_time/baseline_time)*100:.0f}% reduction)

For a medium-scale deployment generating 10,000 images daily, this translates to:
- **{daily_time_saved:.0f} GPU-hours saved per day**
- **${daily_time_saved * 30 * 3:,.0f} monthly cost savings** (at $3/GPU-hour)
""")