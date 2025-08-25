#!/usr/bin/env python3
"""
Compare results between different runs of multi-modality analysis
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Load the data
old_results = pd.read_csv('processed_data_multi_modality_combined/overall_summary_statistics.csv')
new_results = pd.read_csv('processed_data_20250815_multi/overall_summary_statistics.csv')

print("=" * 80)
print("Multi-Modality Results Comparison")
print("Old data: processed_data_multi_modality_combined (2025-08-14)")
print("New data: processed_data_20250815_multi (2025-08-15)")
print("=" * 80)

# Merge results
comparison = pd.merge(
    old_results[['Method', 'Mean_AUROC', 'Std_AUROC', 'Count']],
    new_results[['Method', 'Mean_AUROC', 'Std_AUROC', 'Count']],
    on='Method',
    suffixes=('_old', '_new'),
    how='outer'
)

# Calculate differences
comparison['AUROC_change'] = comparison['Mean_AUROC_new'] - comparison['Mean_AUROC_old']
comparison['Std_change'] = comparison['Std_AUROC_new'] - comparison['Std_AUROC_old']
comparison['Count_change'] = comparison['Count_new'] - comparison['Count_old']

# Sort by new AUROC
comparison = comparison.sort_values('Mean_AUROC_new', ascending=False)

print("\n📊 Performance Comparison (sorted by new AUROC):")
print("-" * 80)
print(f"{'Method':<12} {'Old AUROC':<12} {'New AUROC':<12} {'Change':<12} {'Status'}")
print("-" * 80)

for _, row in comparison.iterrows():
    method = row['Method']
    old_auroc = row['Mean_AUROC_old'] if pd.notna(row['Mean_AUROC_old']) else 0
    new_auroc = row['Mean_AUROC_new'] if pd.notna(row['Mean_AUROC_new']) else 0
    change = row['AUROC_change'] if pd.notna(row['AUROC_change']) else 0
    
    # Determine status
    if pd.isna(row['Mean_AUROC_old']):
        status = "🆕 NEW"
    elif pd.isna(row['Mean_AUROC_new']):
        status = "❌ REMOVED"
    elif abs(change) < 0.001:
        status = "⏸️  UNCHANGED"
    elif change > 0:
        status = f"✅ IMPROVED (+{change:.4f})"
    else:
        status = f"⚠️  DECREASED ({change:.4f})"
    
    print(f"{method:<12} {old_auroc:<12.4f} {new_auroc:<12.4f} {change:+12.4f} {status}")

# Summary statistics
print("\n📈 Summary Statistics:")
print("-" * 80)

# Top performers
print("\n🏆 Top 5 Methods (New Results):")
top5 = new_results.nlargest(5, 'Mean_AUROC')[['Method', 'Mean_AUROC', 'Std_AUROC']]
for idx, row in top5.iterrows():
    print(f"  {idx+1}. {row['Method']:<10} - AUROC: {row['Mean_AUROC']:.4f} (±{row['Std_AUROC']:.4f})")

# Methods with significant changes
significant_changes = comparison[comparison['AUROC_change'].abs() > 0.01].sort_values('AUROC_change', ascending=False)
if not significant_changes.empty:
    print("\n🔄 Significant Changes (>0.01 AUROC):")
    for _, row in significant_changes.iterrows():
        print(f"  {row['Method']:<12}: {row['AUROC_change']:+.4f}")

# Check individual benchmark improvements
print("\n" + "=" * 80)
print("Detailed Benchmark Analysis")
print("=" * 80)

# Load comprehensive tables for more detail
old_comprehensive = pd.read_csv('processed_data_multi_modality_combined/comprehensive_metrics_table.csv')
new_comprehensive = pd.read_csv('processed_data_20250815_multi/comprehensive_metrics_table.csv')

# Group by benchmark and find best methods
benchmarks = new_comprehensive['Benchmark'].unique()
print("\n🔍 Best Method per Benchmark (New Results):")
print("-" * 80)
for benchmark in sorted(benchmarks):
    bench_data = new_comprehensive[new_comprehensive['Benchmark'] == benchmark]
    if not bench_data.empty:
        best = bench_data.nlargest(1, 'Test_AUROC').iloc[0]
        print(f"{benchmark:<25}: {best['Method']:<12} (AUROC: {best['Test_AUROC']:.4f})")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)