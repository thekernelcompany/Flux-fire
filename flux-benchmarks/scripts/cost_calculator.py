#!/usr/bin/env python3
"""
Cost calculation module for FLUX.1-Kontext deployment
Calculates real-world deployment costs across different cloud providers and GPU types
"""

import json
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class GPUPricing:
    """GPU pricing information per hour"""
    provider: str
    gpu_type: str
    price_per_hour: float
    vram_gb: int
    compute_capability: float

@dataclass
class DeploymentScenario:
    """Deployment scenario configuration"""
    name: str
    daily_images: int
    average_resolution: Tuple[int, int]
    average_steps: int
    peak_hour_percentage: float  # % of daily volume in peak hour

class CostCalculator:
    """Calculate deployment costs for FLUX.1-Kontext"""
    
    def __init__(self):
        # GPU pricing data (as of 2024)
        self.gpu_pricing = [
            # AWS
            GPUPricing("AWS", "p3.2xlarge (V100)", 3.06, 16, 7.0),
            GPUPricing("AWS", "p3.8xlarge (4xV100)", 12.24, 64, 7.0),
            GPUPricing("AWS", "p4d.24xlarge (8xA100)", 32.77, 320, 8.0),
            GPUPricing("AWS", "g5.xlarge (A10G)", 1.006, 24, 8.6),
            GPUPricing("AWS", "g5.12xlarge (4xA10G)", 5.672, 96, 8.6),
            
            # Google Cloud
            GPUPricing("GCP", "n1-standard-8 + V100", 2.48, 16, 7.0),
            GPUPricing("GCP", "a2-highgpu-1g (A100)", 3.67, 40, 8.0),
            GPUPricing("GCP", "n1-standard-8 + T4", 0.95, 16, 7.5),
            
            # Azure
            GPUPricing("Azure", "NC6s_v3 (V100)", 3.06, 16, 7.0),
            GPUPricing("Azure", "NC24ads_A100_v4", 3.92, 40, 8.0),
            GPUPricing("Azure", "NCasT4_v3", 0.90, 16, 7.5),
            
            # Modal (serverless)
            GPUPricing("Modal", "A100-40GB", 3.72, 40, 8.0),
            GPUPricing("Modal", "A10G", 1.10, 24, 8.6),
            
            # RunPod
            GPUPricing("RunPod", "RTX 3090", 0.44, 24, 8.6),
            GPUPricing("RunPod", "RTX 4090", 0.74, 24, 8.9),
            GPUPricing("RunPod", "A100 40GB", 1.89, 40, 8.0),
        ]
        
        # Standard deployment scenarios
        self.scenarios = [
            DeploymentScenario("Hobby", 100, (1024, 1024), 20, 0.2),
            DeploymentScenario("Small Business", 1000, (1024, 1024), 20, 0.3),
            DeploymentScenario("Medium Business", 10000, (1024, 1024), 25, 0.4),
            DeploymentScenario("Enterprise", 100000, (1024, 1024), 28, 0.5),
            DeploymentScenario("High Resolution", 5000, (1536, 1536), 35, 0.4),
        ]
    
    def calculate_generation_cost(
        self, 
        inference_time: float,
        gpu_pricing: GPUPricing,
        include_idle_time: bool = True
    ) -> float:
        """Calculate cost per image generation"""
        # Add overhead for model loading, API handling, etc.
        overhead_factor = 1.2 if include_idle_time else 1.0
        
        # Calculate cost
        time_hours = (inference_time * overhead_factor) / 3600
        cost = time_hours * gpu_pricing.price_per_hour
        
        return cost
    
    def calculate_monthly_cost(
        self,
        scenario: DeploymentScenario,
        inference_time: float,
        gpu_pricing: GPUPricing,
        utilization_rate: float = 0.7
    ) -> Dict:
        """Calculate monthly deployment costs"""
        # Daily generations
        daily_images = scenario.daily_images
        
        # Peak hour calculations
        peak_hour_images = daily_images * scenario.peak_hour_percentage
        required_gpus = max(1, int(peak_hour_images / (3600 / inference_time)))
        
        # Calculate costs
        if gpu_pricing.provider in ["Modal", "RunPod"]:
            # Pay-per-use pricing
            monthly_generation_hours = (daily_images * 30 * inference_time) / 3600
            monthly_cost = monthly_generation_hours * gpu_pricing.price_per_hour
            cost_per_image = monthly_cost / (daily_images * 30)
        else:
            # Reserved instance pricing
            monthly_hours = 24 * 30
            monthly_cost = required_gpus * monthly_hours * gpu_pricing.price_per_hour * utilization_rate
            cost_per_image = monthly_cost / (daily_images * 30)
        
        return {
            'scenario': scenario.name,
            'gpu': f"{gpu_pricing.provider} - {gpu_pricing.gpu_type}",
            'required_gpus': required_gpus,
            'monthly_cost': monthly_cost,
            'cost_per_image': cost_per_image,
            'daily_images': daily_images,
            'monthly_images': daily_images * 30
        }
    
    def compare_optimization_savings(
        self,
        baseline_time: float,
        optimized_time: float,
        scenario: DeploymentScenario
    ) -> pd.DataFrame:
        """Compare costs between baseline and optimized versions"""
        results = []
        
        for gpu in self.gpu_pricing:
            # Skip GPUs with insufficient VRAM for FLUX
            if gpu.vram_gb < 24:
                continue
            
            baseline_cost = self.calculate_monthly_cost(scenario, baseline_time, gpu)
            optimized_cost = self.calculate_monthly_cost(scenario, optimized_time, gpu)
            
            savings = baseline_cost['monthly_cost'] - optimized_cost['monthly_cost']
            savings_percentage = (savings / baseline_cost['monthly_cost']) * 100
            
            results.append({
                'provider': gpu.provider,
                'gpu_type': gpu.gpu_type,
                'baseline_monthly_cost': baseline_cost['monthly_cost'],
                'optimized_monthly_cost': optimized_cost['monthly_cost'],
                'monthly_savings': savings,
                'savings_percentage': savings_percentage,
                'baseline_cost_per_image': baseline_cost['cost_per_image'],
                'optimized_cost_per_image': optimized_cost['cost_per_image'],
                'required_gpus_baseline': baseline_cost['required_gpus'],
                'required_gpus_optimized': optimized_cost['required_gpus']
            })
        
        return pd.DataFrame(results)
    
    def calculate_roi(
        self,
        development_hours: float,
        developer_hourly_rate: float,
        monthly_savings: float
    ) -> Dict:
        """Calculate ROI for optimization development"""
        development_cost = development_hours * developer_hourly_rate
        months_to_break_even = development_cost / monthly_savings if monthly_savings > 0 else float('inf')
        
        # Calculate savings over different periods
        savings_1_year = (monthly_savings * 12) - development_cost
        savings_2_years = (monthly_savings * 24) - development_cost
        
        roi_1_year = (savings_1_year / development_cost) * 100 if development_cost > 0 else 0
        roi_2_years = (savings_2_years / development_cost) * 100 if development_cost > 0 else 0
        
        return {
            'development_cost': development_cost,
            'monthly_savings': monthly_savings,
            'months_to_break_even': months_to_break_even,
            'savings_1_year': savings_1_year,
            'savings_2_years': savings_2_years,
            'roi_1_year': roi_1_year,
            'roi_2_years': roi_2_years
        }
    
    def create_cost_visualizations(
        self,
        baseline_time: float,
        optimized_time: float,
        output_dir: str = "cost_analysis"
    ):
        """Create cost analysis visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Cost comparison across providers
        scenario = self.scenarios[2]  # Medium Business
        df_comparison = self.compare_optimization_savings(baseline_time, optimized_time, scenario)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Monthly cost comparison
        x = range(len(df_comparison))
        width = 0.35
        
        bars1 = ax1.bar([i - width/2 for i in x], df_comparison['baseline_monthly_cost'], 
                        width, label='Baseline', alpha=0.8, color='#FF6B6B')
        bars2 = ax1.bar([i + width/2 for i in x], df_comparison['optimized_monthly_cost'], 
                        width, label='Optimized', alpha=0.8, color='#4ECDC4')
        
        ax1.set_xlabel('GPU Provider', fontsize=12)
        ax1.set_ylabel('Monthly Cost ($)', fontsize=12)
        ax1.set_title(f'Monthly Cost Comparison - {scenario.name} Scenario', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{row['provider']}\n{row['gpu_type'].split('(')[0]}" 
                            for _, row in df_comparison.iterrows()], rotation=45, ha='right')
        ax1.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}', ha='center', va='bottom')
        
        # Savings percentage
        bars3 = ax2.bar(x, df_comparison['savings_percentage'], alpha=0.8, color='#6C5CE7')
        ax2.set_xlabel('GPU Provider', fontsize=12)
        ax2.set_ylabel('Cost Savings (%)', fontsize=12)
        ax2.set_title('Percentage Cost Savings with Optimization', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{row['provider']}\n{row['gpu_type'].split('(')[0]}" 
                            for _, row in df_comparison.iterrows()], rotation=45, ha='right')
        
        # Add percentage labels
        for bar, pct in zip(bars3, df_comparison['savings_percentage']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cost scaling across scenarios
        fig, ax = plt.subplots(figsize=(12, 6))
        
        selected_gpu = next(gpu for gpu in self.gpu_pricing if "A100" in gpu.gpu_type and gpu.provider == "AWS")
        
        scenario_names = []
        baseline_costs = []
        optimized_costs = []
        
        for scenario in self.scenarios:
            scenario_names.append(scenario.name)
            base = self.calculate_monthly_cost(scenario, baseline_time, selected_gpu)
            opt = self.calculate_monthly_cost(scenario, optimized_time, selected_gpu)
            baseline_costs.append(base['monthly_cost'])
            optimized_costs.append(opt['monthly_cost'])
        
        x = range(len(scenario_names))
        bars1 = ax.bar([i - width/2 for i in x], baseline_costs, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], optimized_costs, width, label='Optimized', alpha=0.8)
        
        ax.set_xlabel('Deployment Scenario', fontsize=12)
        ax.set_ylabel('Monthly Cost ($)', fontsize=12)
        ax.set_title(f'Cost Scaling Across Scenarios ({selected_gpu.gpu_type})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'cost_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROI Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Assume 40 hours of development time at $150/hour
        dev_hours = 40
        hourly_rate = 150
        
        roi_data = []
        for _, row in df_comparison.iterrows():
            roi = self.calculate_roi(dev_hours, hourly_rate, row['monthly_savings'])
            roi_data.append({
                'provider': f"{row['provider']} - {row['gpu_type'].split('(')[0]}",
                'months_to_break_even': roi['months_to_break_even']
            })
        
        roi_df = pd.DataFrame(roi_data)
        roi_df = roi_df[roi_df['months_to_break_even'] < 24]  # Filter out very long payback periods
        
        bars = ax.bar(range(len(roi_df)), roi_df['months_to_break_even'], color='#00D2FF')
        ax.set_xlabel('GPU Provider', fontsize=12)
        ax.set_ylabel('Months to Break Even', fontsize=12)
        ax.set_title('ROI Analysis: Time to Recover Development Cost', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(roi_df)))
        ax.set_xticklabels(roi_df['provider'], rotation=45, ha='right')
        ax.axhline(y=12, color='red', linestyle='--', alpha=0.5, label='1 Year')
        ax.legend()
        
        # Add value labels
        for bar, months in zip(bars, roi_df['months_to_break_even']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{months:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'roi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cost analysis visualizations saved to {output_path}")
    
    def generate_cost_report(
        self,
        baseline_time: float,
        optimized_time: float,
        output_file: str = "cost_analysis_report.md"
    ) -> str:
        """Generate a comprehensive cost analysis report"""
        speedup = baseline_time / optimized_time
        
        # Calculate savings for medium business scenario
        scenario = self.scenarios[2]
        
        # Find best value GPU options
        df_comparison = self.compare_optimization_savings(baseline_time, optimized_time, scenario)
        best_value = df_comparison.nsmallest(3, 'optimized_monthly_cost')
        
        report = f"""# FLUX.1-Kontext Cost Analysis Report

## Executive Summary

With a **{speedup:.2f}x performance improvement**, the optimized FLUX.1-Kontext delivers substantial cost savings across all deployment scenarios.

## Key Performance Metrics

- **Baseline inference time**: {baseline_time:.2f} seconds
- **Optimized inference time**: {optimized_time:.2f} seconds
- **Performance improvement**: {speedup:.2f}x faster
- **Time saved per image**: {baseline_time - optimized_time:.2f} seconds

## Cost Savings Analysis

### Medium Business Scenario (10,000 images/day)

| Provider | GPU Type | Monthly Savings | Savings % |
|----------|----------|----------------|-----------|
"""
        
        for _, row in df_comparison.nlargest(5, 'monthly_savings').iterrows():
            report += f"| {row['provider']} | {row['gpu_type'].split('(')[0]} | ${row['monthly_savings']:,.0f} | {row['savings_percentage']:.1f}% |\n"
        
        report += f"""
### Best Value Options

For optimal cost-efficiency, we recommend:

"""
        
        for _, row in best_value.iterrows():
            report += f"- **{row['provider']} - {row['gpu_type']}**: ${row['optimized_monthly_cost']:,.0f}/month (${row['optimized_cost_per_image']:.3f}/image)\n"
        
        # ROI calculation
        roi = self.calculate_roi(40, 150, df_comparison['monthly_savings'].mean())
        
        report += f"""
## Return on Investment

Assuming 40 hours of development time at $150/hour:

- **Development cost**: ${roi['development_cost']:,.0f}
- **Average monthly savings**: ${roi['monthly_savings']:,.0f}
- **Break-even time**: {roi['months_to_break_even']:.1f} months
- **1-year net savings**: ${roi['savings_1_year']:,.0f}
- **2-year net savings**: ${roi['savings_2_years']:,.0f}

## Deployment Recommendations

1. **For startups**: Use serverless providers (Modal, RunPod) for pay-per-use pricing
2. **For scale**: Deploy on AWS/GCP with reserved instances for best rates
3. **For cost optimization**: Consider RunPod RTX 4090 instances for excellent price/performance

## Conclusion

The optimization investment pays for itself in under {roi['months_to_break_even']:.0f} months for most deployment scenarios, with substantial long-term savings.
"""
        
        return report

def main():
    # Example usage
    calc = CostCalculator()
    
    # Typical performance numbers
    baseline_time = 6.75  # seconds
    optimized_time = 2.56  # seconds
    
    # Generate cost analysis
    calc.create_cost_visualizations(baseline_time, optimized_time)
    
    # Generate report
    report = calc.generate_cost_report(baseline_time, optimized_time)
    
    with open('cost_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("Cost analysis complete!")
    print(f"Speedup: {baseline_time/optimized_time:.2f}x")
    print(f"Report saved to cost_analysis_report.md")

if __name__ == "__main__":
    main()