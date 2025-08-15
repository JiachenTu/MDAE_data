#!/usr/bin/env python3
"""
Extract Raw Ablation Data to Structured CSV Files
Extracts detailed ablation data from JSON and saves as CSV for analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ablation_data(file_path: str = "ablation_analysis/raw_data/ablations_only.json"):
    """Load ablation data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_noise_corruption_data(data):
    """Extract noise corruption ablation data"""
    logger.info("Extracting noise corruption data...")
    rows = []
    
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'noise_corruption':
                summary = run.get('summary', {})
                
                row = {
                    # Identifiers
                    'benchmark': benchmark,
                    'modality': modality,
                    'run_id': run['id'],
                    'run_name': run['name'],
                    'created_at': run['created_at'],
                    'state': run['state'],
                    
                    # Ablation parameters
                    'noise_type': ablation['noise_type'],
                    'masking_ratio': ablation['masking_ratio'],
                    
                    # Test metrics
                    'test_auroc': summary.get('Test/AUROC') or summary.get('Test_AUROC'),
                    'test_ap': summary.get('Test/AP') or summary.get('Test_AP'),
                    'test_f1': summary.get('Test/F1') or summary.get('Test_F1'),
                    'test_balanced_acc': summary.get('Test/Balanced_Accuracy') or summary.get('Test_Balanced_Accuracy'),
                    'test_loss': summary.get('Test/loss'),
                    
                    # Validation metrics
                    'val_auroc': summary.get('Val/AUROC') or summary.get('Val_AUROC'),
                    'val_ap': summary.get('Val/AP') or summary.get('Val_AP'),
                    'val_f1': summary.get('Val/F1') or summary.get('Val_F1'),
                    'val_balanced_acc': summary.get('Val/Balanced_Accuracy') or summary.get('Val_Balanced_Accuracy'),
                    'val_loss': summary.get('Val/loss'),
                    
                    # Training metrics
                    'train_auroc': summary.get('Train/AUROC') or summary.get('Train_AUROC'),
                    'train_ap': summary.get('Train/AP') or summary.get('Train_AP'),
                    'train_loss': summary.get('Train/loss'),
                    
                    # Additional info
                    'notes': run.get('notes', ''),
                    'tags': ','.join(run.get('tags', []))
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} noise corruption runs")
    return df

def extract_masking_ratio_data(data):
    """Extract masking ratio ablation data"""
    logger.info("Extracting masking ratio data...")
    rows = []
    
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'masking_ratio':
                summary = run.get('summary', {})
                
                row = {
                    # Identifiers
                    'benchmark': benchmark,
                    'modality': modality,
                    'run_id': run['id'],
                    'run_name': run['name'],
                    'created_at': run['created_at'],
                    'state': run['state'],
                    
                    # Ablation parameters
                    'masking_ratio': ablation['masking_ratio'],
                    'noise_type': ablation.get('noise_type', 'Flow'),  # Default to Flow
                    
                    # Test metrics
                    'test_auroc': summary.get('Test/AUROC') or summary.get('Test_AUROC'),
                    'test_ap': summary.get('Test/AP') or summary.get('Test_AP'),
                    'test_f1': summary.get('Test/F1') or summary.get('Test_F1'),
                    'test_balanced_acc': summary.get('Test/Balanced_Accuracy') or summary.get('Test_Balanced_Accuracy'),
                    'test_loss': summary.get('Test/loss'),
                    
                    # Validation metrics
                    'val_auroc': summary.get('Val/AUROC') or summary.get('Val_AUROC'),
                    'val_ap': summary.get('Val/AP') or summary.get('Val_AP'),
                    'val_f1': summary.get('Val/F1') or summary.get('Val_F1'),
                    
                    # Training metrics
                    'train_auroc': summary.get('Train/AUROC') or summary.get('Train_AUROC'),
                    'train_loss': summary.get('Train/loss'),
                    
                    # Additional info
                    'notes': run.get('notes', '')
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} masking ratio runs")
    return df

def extract_masking_type_data(data):
    """Extract masking type ablation data"""
    logger.info("Extracting masking type data...")
    rows = []
    
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'masking_type':
                summary = run.get('summary', {})
                
                row = {
                    # Identifiers
                    'benchmark': benchmark,
                    'modality': modality,
                    'run_id': run['id'],
                    'run_name': run['name'],
                    'created_at': run['created_at'],
                    'state': run['state'],
                    
                    # Ablation parameters
                    'masking_type': ablation.get('masking_type', 'Unknown'),
                    'masking_ratio': ablation.get('masking_ratio', 90),  # Default 90%
                    
                    # Test metrics
                    'test_auroc': summary.get('Test/AUROC') or summary.get('Test_AUROC'),
                    'test_ap': summary.get('Test/AP') or summary.get('Test_AP'),
                    'test_f1': summary.get('Test/F1') or summary.get('Test_F1'),
                    'test_balanced_acc': summary.get('Test/Balanced_Accuracy') or summary.get('Test_Balanced_Accuracy'),
                    'test_loss': summary.get('Test/loss'),
                    
                    # Validation metrics
                    'val_auroc': summary.get('Val/AUROC') or summary.get('Val_AUROC'),
                    'val_ap': summary.get('Val/AP') or summary.get('Val_AP'),
                    
                    # Training metrics
                    'train_auroc': summary.get('Train/AUROC') or summary.get('Train_AUROC'),
                    'train_loss': summary.get('Train/loss'),
                    
                    # Additional info
                    'notes': run.get('notes', '')
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} masking type runs")
    return df

def extract_flowmdae_data(data):
    """Extract FlowMDAE ablation data"""
    logger.info("Extracting FlowMDAE data...")
    rows = []
    
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'flowmdae':
                summary = run.get('summary', {})
                
                row = {
                    # Identifiers
                    'benchmark': benchmark,
                    'modality': modality,
                    'run_id': run['id'],
                    'run_name': run['name'],
                    'created_at': run['created_at'],
                    'state': run['state'],
                    
                    # Ablation parameters
                    'masking_ratio': ablation['masking_ratio'],
                    'noise_level': ablation['noise_level'],
                    'param_combo': f"M{ablation['masking_ratio']}_N{ablation['noise_level']}",
                    
                    # Test metrics
                    'test_auroc': summary.get('Test/AUROC') or summary.get('Test_AUROC'),
                    'test_ap': summary.get('Test/AP') or summary.get('Test_AP'),
                    'test_f1': summary.get('Test/F1') or summary.get('Test_F1'),
                    'test_balanced_acc': summary.get('Test/Balanced_Accuracy') or summary.get('Test_Balanced_Accuracy'),
                    'test_loss': summary.get('Test/loss'),
                    
                    # Validation metrics
                    'val_auroc': summary.get('Val/AUROC') or summary.get('Val_AUROC'),
                    'val_ap': summary.get('Val/AP') or summary.get('Val_AP'),
                    'val_f1': summary.get('Val/F1') or summary.get('Val_F1'),
                    
                    # Training metrics
                    'train_auroc': summary.get('Train/AUROC') or summary.get('Train_AUROC'),
                    'train_loss': summary.get('Train/loss'),
                    
                    # Additional info
                    'notes': run.get('notes', '')
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} FlowMDAE runs")
    return df

def extract_other_ablations_data(data):
    """Extract other/unclassified ablation data for investigation"""
    logger.info("Extracting other ablation data...")
    rows = []
    
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'other':
                summary = run.get('summary', {})
                
                row = {
                    # Identifiers
                    'benchmark': benchmark,
                    'modality': modality,
                    'run_id': run['id'],
                    'run_name': run['name'],
                    'created_at': run['created_at'],
                    'state': run['state'],
                    
                    # Test metrics
                    'test_auroc': summary.get('Test/AUROC') or summary.get('Test_AUROC'),
                    'test_ap': summary.get('Test/AP') or summary.get('Test_AP'),
                    'test_f1': summary.get('Test/F1') or summary.get('Test_F1'),
                    
                    # Raw notes for investigation
                    'notes': run.get('notes', ''),
                    'raw_note': ablation.get('raw_note', '')
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} other ablation runs")
    return df

def generate_summary_statistics(dfs):
    """Generate summary statistics for all ablation types"""
    logger.info("Generating summary statistics...")
    
    stats = []
    for name, df in dfs.items():
        if not df.empty and 'test_auroc' in df.columns:
            valid_auroc = df['test_auroc'].dropna()
            if len(valid_auroc) > 0:
                stats.append({
                    'ablation_type': name,
                    'total_runs': len(df),
                    'valid_runs': len(valid_auroc),
                    'mean_auroc': valid_auroc.mean(),
                    'std_auroc': valid_auroc.std(),
                    'min_auroc': valid_auroc.min(),
                    'max_auroc': valid_auroc.max(),
                    'median_auroc': valid_auroc.median(),
                    'q25_auroc': valid_auroc.quantile(0.25),
                    'q75_auroc': valid_auroc.quantile(0.75)
                })
    
    return pd.DataFrame(stats)

def main():
    """Main extraction pipeline"""
    logger.info("Starting raw data extraction...")
    
    # Create output directory
    output_dir = Path("ablation_analysis/raw_data_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_ablation_data()
    logger.info(f"Loaded {len(data)} projects with ablation data")
    
    # Extract data for each ablation type
    noise_df = extract_noise_corruption_data(data)
    masking_ratio_df = extract_masking_ratio_data(data)
    masking_type_df = extract_masking_type_data(data)
    flowmdae_df = extract_flowmdae_data(data)
    other_df = extract_other_ablations_data(data)
    
    # Save raw data to CSV
    noise_df.to_csv(output_dir / "noise_corruption_raw.csv", index=False)
    masking_ratio_df.to_csv(output_dir / "masking_ratio_raw.csv", index=False)
    masking_type_df.to_csv(output_dir / "masking_type_raw.csv", index=False)
    flowmdae_df.to_csv(output_dir / "flowmdae_raw.csv", index=False)
    other_df.to_csv(output_dir / "other_ablations_raw.csv", index=False)
    
    # Generate and save summary statistics
    dfs = {
        'noise_corruption': noise_df,
        'masking_ratio': masking_ratio_df,
        'masking_type': masking_type_df,
        'flowmdae': flowmdae_df,
        'other': other_df
    }
    
    summary_stats = generate_summary_statistics(dfs)
    summary_stats.to_csv(output_dir / "summary_statistics.csv", index=False)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RAW DATA EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nFiles created:")
    logger.info(f"  - noise_corruption_raw.csv: {len(noise_df)} rows")
    logger.info(f"  - masking_ratio_raw.csv: {len(masking_ratio_df)} rows")
    logger.info(f"  - masking_type_raw.csv: {len(masking_type_df)} rows")
    logger.info(f"  - flowmdae_raw.csv: {len(flowmdae_df)} rows")
    logger.info(f"  - other_ablations_raw.csv: {len(other_df)} rows")
    logger.info(f"  - summary_statistics.csv: {len(summary_stats)} ablation types")
    
    # Display summary statistics
    logger.info("\nSummary Statistics:")
    print(summary_stats.to_string())
    
    # Additional analysis for noise corruption
    if not noise_df.empty:
        logger.info("\nNoise Corruption Breakdown:")
        noise_summary = noise_df.groupby('noise_type')['test_auroc'].agg(['count', 'mean', 'std'])
        print(noise_summary)
    
    # Additional analysis for FlowMDAE
    if not flowmdae_df.empty:
        logger.info("\nFlowMDAE Best Configurations:")
        best_configs = flowmdae_df.nlargest(5, 'test_auroc')[['param_combo', 'test_auroc', 'benchmark', 'modality']]
        print(best_configs)
    
    return summary_stats

if __name__ == "__main__":
    stats = main()