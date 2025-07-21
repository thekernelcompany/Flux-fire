#!/usr/bin/env python3
"""
Cost analysis specifically for NVIDIA H100 GPU
Based on actual benchmark results from H100 PCIe with 79.1GB VRAM
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Load actual benchmark results
with open('benchmark_results/benchmark_results.json', 'r') as f:
    data = json.load(f)

# Get actual times from H100 benchmark
baseline_time = 7.11  # seconds
optimized_time = 2.74  # seconds
speedup = baseline_time / optimized_time

# H100 pricing from major cloud providers (per hour)
h100_pricing = {
    'AWS p5.48xlarge (8x H100)': 98.32 / 8,  # $12.29 per GPU
    'Google Cloud a3-highgpu-8g': 25.21 / 8,  # $3.15 per GPU  
    'Azure NC40ads H100 v5': 5.50,  # Per GPU hour
    'Lambda Labs H100': 2.49,  # On-demand
    'CoreWeave H100 PCIe': 2.28,  # Similar to yours
    'RunPod H100 PCIe': 2.99,  # Community cloud
}

# Calculate costs for different daily volumes
daily_volumes = [100, 1000, 5000, 10000, 50000, 100000]

def calculate_monthly_cost(images_per_day, time_per_image, hourly_rate):
    """Calculate monthly cost for given volume and performance"""
    hours_per_day = (images_per_day * time_per_image) / 3600
    monthly_cost = hours_per_day * 30 * hourly_rate
    return monthly_cost

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Cost comparison by provider
providers = list(h100_pricing.keys())
baseline_costs = []
optimized_costs = []
savings = []

for provider, rate in h100_pricing.items():
    # For 10,000 images/day scenario
    base_cost = calculate_monthly_cost(10000, baseline_time, rate)
    opt_cost = calculate_monthly_cost(10000, optimized_time, rate)
    baseline_costs.append(base_cost)
    optimized_costs.append(opt_cost)
    savings.append(base_cost - opt_cost)

x = np.arange(len(providers))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_costs, width, label='Baseline (7.11s)', 
                 color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, optimized_costs, width, label='Optimized (2.74s)', 
                 color='#4ECDC4', alpha=0.8)

ax1.set_xlabel('H100 Provider', fontsize=12)
ax1.set_ylabel('Monthly Cost ($)', fontsize=12)
ax1.set_title('H100 Monthly Cost Comparison (10,000 images/day)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([p.split()[0] for p in providers], rotation=45, ha='right')
ax1.legend()

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Savings across different volumes (using CoreWeave pricing as reference)
reference_rate = h100_pricing['CoreWeave H100 PCIe']  # $2.28/hour - similar to your setup

for volume in daily_volumes:
    base_cost = calculate_monthly_cost(volume, baseline_time, reference_rate)
    opt_cost = calculate_monthly_cost(volume, optimized_time, reference_rate)
    monthly_savings = base_cost - opt_cost
    
    ax2.bar(str(volume), monthly_savings, color='#6C5CE7', alpha=0.8)

ax2.set_xlabel('Daily Image Volume', fontsize=12)
ax2.set_ylabel('Monthly Savings ($)', fontsize=12)
ax2.set_title(f'Monthly Savings with {speedup:.2f}x Speedup (H100 PCIe @ $2.28/hr)', 
              fontsize=14, fontweight='bold')

# Add value labels
for i, volume in enumerate(daily_volumes):
    savings = calculate_monthly_cost(volume, baseline_time, reference_rate) - \
             calculate_monthly_cost(volume, optimized_time, reference_rate)
    ax2.text(i, savings, f'${savings:,.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('h100_cost_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create detailed cost breakdown
print("=== H100 PCIe Cost Analysis ===")
print(f"\nGPU: NVIDIA H100 PCIe (79.1GB)")
print(f"Baseline: {baseline_time:.2f}s per image")
print(f"Optimized: {optimized_time:.2f}s per image")
print(f"Speedup: {speedup:.2f}x\n")

print("Monthly Cost Comparison (10,000 images/day):")
print("-" * 60)
print(f"{'Provider':<25} {'Baseline':<12} {'Optimized':<12} {'Savings':<12}")
print("-" * 60)

for provider, rate in h100_pricing.items():
    base = calculate_monthly_cost(10000, baseline_time, rate)
    opt = calculate_monthly_cost(10000, optimized_time, rate)
    save = base - opt
    print(f"{provider:<25} ${base:>10,.0f} ${opt:>10,.0f} ${save:>10,.0f}")

# ROI calculation
print("\n=== Return on Investment ===")
dev_cost = 40 * 150  # 40 hours at $150/hour
avg_savings = np.mean(savings)
print(f"Development cost: ${dev_cost:,}")
print(f"Average monthly savings: ${avg_savings:,.0f}")
print(f"Break-even time: {dev_cost/avg_savings:.1f} months")

# Create summary image with key metrics
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'NVIDIA H100 PCIe Performance & Cost Impact', 
        fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

# Key metrics boxes
metrics = [
    ('Performance', f'{speedup:.2f}x Faster', '#4ECDC4'),
    ('Time Saved', f'{baseline_time - optimized_time:.2f}s/image', '#45B7D1'),
    ('Cost Reduction', f'{(1 - optimized_time/baseline_time)*100:.0f}%', '#6C5CE7'),
    ('Monthly Savings', f'${avg_savings:,.0f}', '#00D2FF')
]

y_pos = 0.7
for metric, value, color in metrics:
    # Box
    box = plt.Rectangle((0.1 + (metrics.index((metric, value, color)) * 0.22), y_pos - 0.15), 
                       0.2, 0.15, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    
    # Text
    ax.text(0.2 + (metrics.index((metric, value, color)) * 0.22), y_pos - 0.05, 
           metric, fontsize=12, ha='center', fontweight='bold')
    ax.text(0.2 + (metrics.index((metric, value, color)) * 0.22), y_pos - 0.1, 
           value, fontsize=16, ha='center')

# Bottom text
ax.text(0.5, 0.3, 'Deployment Scenarios:', fontsize=14, fontweight='bold', 
        ha='center', transform=ax.transAxes)

scenarios = [
    f"Startup (1K/day): ${calculate_monthly_cost(1000, baseline_time - optimized_time, reference_rate):,.0f}/mo saved",
    f"Medium (10K/day): ${calculate_monthly_cost(10000, baseline_time - optimized_time, reference_rate):,.0f}/mo saved",
    f"Enterprise (100K/day): ${calculate_monthly_cost(100000, baseline_time - optimized_time, reference_rate):,.0f}/mo saved"
]

for i, scenario in enumerate(scenarios):
    ax.text(0.5, 0.2 - i*0.05, scenario, fontsize=12, ha='center', transform=ax.transAxes)

plt.savefig('h100_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ H100 cost analysis saved to:")
print(f"  - h100_cost_analysis.png")
print(f"  - h100_summary.png")