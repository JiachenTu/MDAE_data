#!/usr/bin/env python3
"""
Comprehensive Ablation Study Extraction Script
Extracts all runs from 61 single-modality projects with complete Notes field
"""

import wandb
import json
import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WandB entity - Updated based on actual project location
ENTITY = "t-jiachentu"

# Define all 61 single-modality projects
SINGLE_MODALITY_PROJECTS = {
    # BraTS18 (4 modalities)
    "brats18_lgg_vs_hgg": ["flair", "t1", "t1ce", "t2"],
    
    # BraTS23 (3 tasks × 4 modalities = 12 projects)
    "brats23_gli_vs_men": ["flair", "t1", "t1ce", "t2"],
    "brats23_gli_vs_met": ["flair", "t1", "t1ce", "t2"],
    "brats23_men_vs_met": ["flair", "t1", "t1ce", "t2"],
    
    # RSNA-MICCAI (4 modalities)
    "rsna_miccai_brain_tumor_mgmt_methylation": ["flair", "t1", "t1ce", "t2"],
    
    # TCGA-GBM (2 tasks with different modality counts)
    "tcga_gbm_dss_1year": ["flair", "t1", "t1ce", "t1gd", "t2"],
    "tcga_gbm_pfi_1year": ["t1", "t2"],
    
    # UCSF-PDGM (6 modalities)
    "ucsf_pdgm_idh_classification": ["asl", "flair", "swi", "t1", "t1c", "t2"],
    
    # UPenn-GBM (7 tasks × 4 modalities = 28 projects)
    "upenn_gbm_age_group": ["flair", "t1", "t1ce", "t2"],
    "upenn_gbm_gender": ["flair", "t1", "t1ce", "t2"],
    "upenn_gbm_gtr_status": ["flair", "t1", "t1ce", "t2"],
    "upenn_gbm_idh1_status": ["flair", "t1", "t1ce", "t2"],
    "upenn_gbm_survival_18month": ["flair", "t1", "t1ce", "t2"],
    "upenn_gbm_survival_1year": ["flair", "t1", "t1ce", "t2"],
    "upenn_gbm_survival_2year": ["flair", "t1", "t1ce", "t2"],
}

def generate_project_names() -> List[str]:
    """Generate all 61 single-modality project names"""
    projects = []
    for benchmark, modalities in SINGLE_MODALITY_PROJECTS.items():
        for modality in modalities:
            project_name = f"july_stratified_{benchmark}_single_{modality}"
            projects.append(project_name)
    return projects

def parse_ablation_parameters(notes: str) -> Optional[Dict]:
    """
    Parse ablation parameters from the Notes field
    Returns parsed ablation info or None if not an ablation run
    """
    if not notes:
        return None
    
    ablation_info = {}
    
    # 1. Noise Corruption Type
    noise_pattern = r"MDAE ablation Noise Corruption (VE|VP|Flow) M(\d+)"
    noise_match = re.search(noise_pattern, notes)
    if noise_match:
        ablation_info = {
            "ablation_type": "noise_corruption",
            "noise_type": noise_match.group(1),
            "masking_ratio": int(noise_match.group(2)),
            "raw_note": notes
        }
        return ablation_info
    
    # 2. Masking Type
    if "Masking Type Ablation:" in notes:
        masking_type = None
        if "Random patch" in notes:
            masking_type = "Random"
        elif "Block" in notes or "blocky" in notes:
            masking_type = "Block"
        elif "Tube" in notes:
            masking_type = "Tube"
        
        ablation_info = {
            "ablation_type": "masking_type",
            "masking_type": masking_type,
            "masking_ratio": 90,  # Fixed at 90%
            "raw_note": notes
        }
        return ablation_info
    
    # 3. Masking Ratio
    ratio_pattern = r"Masking Ratio Ablation: (Flow|VE|VP) (\d+)%"
    ratio_match = re.search(ratio_pattern, notes)
    if ratio_match:
        ablation_info = {
            "ablation_type": "masking_ratio",
            "noise_type": ratio_match.group(1),
            "masking_ratio": int(ratio_match.group(2)),
            "raw_note": notes
        }
        return ablation_info
    
    # 4. FlowMDAE Ablation
    flowmdae_pattern = r"FlowMDAE Ablation.*M(\d+)_N(\d+)"
    flowmdae_match = re.search(flowmdae_pattern, notes)
    if flowmdae_match:
        ablation_info = {
            "ablation_type": "flowmdae",
            "masking_ratio": int(flowmdae_match.group(1)),
            "noise_level": int(flowmdae_match.group(2)),
            "raw_note": notes
        }
        return ablation_info
    
    # Check if it might be an ablation but with different format
    ablation_keywords = ["ablation", "Ablation", "ABLATION", "masking", "Masking", "noise", "Noise"]
    if any(keyword in notes for keyword in ablation_keywords):
        ablation_info = {
            "ablation_type": "other",
            "raw_note": notes
        }
        return ablation_info
    
    return None

def extract_project_data(project_name: str, max_retries: int = 3) -> Dict:
    """
    Extract all runs from a single WandB project with retry logic
    """
    for attempt in range(max_retries):
        try:
            api = wandb.Api()
            runs = api.runs(f"{ENTITY}/{project_name}")
            
            # Parse benchmark and modality from project name
            parts = project_name.split("_")
            modality = parts[-1]
            benchmark = "_".join(parts[2:-2])  # Remove july_stratified_ prefix and _single_modality suffix
            
            project_data = {
                "project": project_name,
                "benchmark": benchmark,
                "modality": modality,
                "extraction_time": datetime.now().isoformat(),
                "total_runs": len(runs),
                "runs": [],
                "ablation_summary": {
                    "total_ablations": 0,
                    "noise_corruption": 0,
                    "masking_type": 0,
                    "masking_ratio": 0,
                    "flowmdae": 0,
                    "other": 0
                }
            }
            
            for run in runs:
                # Extract comprehensive run data
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "notes": run.notes or "",  # CRITICAL: Extract Notes field
                    "tags": run.tags,
                    "config": dict(run.config) if run.config else {},
                    "summary": dict(run.summary) if run.summary else {},
                    "metrics": {}
                }
                
                # Extract key metrics
                metrics_to_extract = [
                    "Test_AUROC", "Test_AP", "Test_F1", "Test_Balanced_Accuracy",
                    "Val_AUROC", "Val_AP", "Val_F1", "Val_Balanced_Accuracy",
                    "metric_Test/AUROC", "metric_Test/AP", "metric_Test/F1",
                    "metric_Val/AUROC", "metric_Val/AP", "metric_Val/F1"
                ]
                
                for metric in metrics_to_extract:
                    if metric in run.summary:
                        run_data["metrics"][metric] = run.summary[metric]
                
                # Parse ablation parameters from Notes
                ablation_params = parse_ablation_parameters(run.notes)
                run_data["ablation_parsed"] = ablation_params
                
                # Update ablation summary
                if ablation_params:
                    project_data["ablation_summary"]["total_ablations"] += 1
                    ablation_type = ablation_params.get("ablation_type", "other")
                    if ablation_type in project_data["ablation_summary"]:
                        project_data["ablation_summary"][ablation_type] += 1
                
                project_data["runs"].append(run_data)
            
            logger.info(f"✓ Extracted {project_name}: {len(runs)} runs, {project_data['ablation_summary']['total_ablations']} ablations")
            return project_data
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {project_name}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"✗ Failed to extract {project_name} after {max_retries} attempts")
                return {
                    "project": project_name,
                    "error": str(e),
                    "extraction_time": datetime.now().isoformat()
                }

def extract_all_ablations_parallel(max_workers: int = 10):
    """
    Extract data from all 61 projects in parallel
    """
    projects = generate_project_names()
    logger.info(f"Starting extraction of {len(projects)} single-modality projects")
    
    # Create output directory
    output_dir = "ablation_analysis/raw_data/extracted_with_notes"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track overall statistics
    overall_stats = {
        "total_projects": len(projects),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "total_runs": 0,
        "total_ablations": 0,
        "ablation_types": {
            "noise_corruption": 0,
            "masking_type": 0,
            "masking_ratio": 0,
            "flowmdae": 0,
            "other": 0
        },
        "extraction_start": datetime.now().isoformat()
    }
    
    # Extract data in parallel
    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_project = {
            executor.submit(extract_project_data, project): project 
            for project in projects
        }
        
        for future in as_completed(future_to_project):
            project = future_to_project[future]
            try:
                data = future.result()
                
                if "error" not in data:
                    overall_stats["successful_extractions"] += 1
                    overall_stats["total_runs"] += data["total_runs"]
                    overall_stats["total_ablations"] += data["ablation_summary"]["total_ablations"]
                    
                    # Update ablation type counts
                    for ablation_type, count in data["ablation_summary"].items():
                        if ablation_type != "total_ablations" and ablation_type in overall_stats["ablation_types"]:
                            overall_stats["ablation_types"][ablation_type] += count
                    
                    # Save individual project data
                    project_file = os.path.join(output_dir, f"{project}.json")
                    with open(project_file, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    overall_stats["failed_extractions"] += 1
                
                all_data.append(data)
                
            except Exception as e:
                logger.error(f"Failed to process {project}: {str(e)}")
                overall_stats["failed_extractions"] += 1
    
    overall_stats["extraction_end"] = datetime.now().isoformat()
    
    # Save combined data
    combined_file = os.path.join(output_dir, "../all_ablations_combined.json")
    with open(combined_file, 'w') as f:
        json.dump({
            "statistics": overall_stats,
            "projects": all_data
        }, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"✓ Successful extractions: {overall_stats['successful_extractions']}/{overall_stats['total_projects']}")
    logger.info(f"✗ Failed extractions: {overall_stats['failed_extractions']}")
    logger.info(f"Total runs extracted: {overall_stats['total_runs']}")
    logger.info(f"Total ablations found: {overall_stats['total_ablations']}")
    logger.info("\nAblation breakdown:")
    for ablation_type, count in overall_stats['ablation_types'].items():
        logger.info(f"  - {ablation_type}: {count}")
    
    # Save summary statistics
    stats_file = os.path.join(output_dir, "../extraction_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2, default=str)
    
    return overall_stats

def main():
    """Main execution function"""
    logger.info("Starting comprehensive ablation extraction")
    logger.info(f"Target: {len(generate_project_names())} single-modality projects")
    logger.info("Focus: Complete Notes field extraction for ablation studies")
    
    # Run extraction
    stats = extract_all_ablations_parallel(max_workers=10)
    
    # Create ablations-only filtered file
    logger.info("\nCreating ablations-only filtered dataset...")
    
    ablations_only = []
    output_dir = "ablation_analysis/raw_data/extracted_with_notes"
    
    for project_file in os.listdir(output_dir):
        if project_file.endswith('.json') and project_file != 'all_ablations_combined.json':
            with open(os.path.join(output_dir, project_file), 'r') as f:
                project_data = json.load(f)
                
                # Filter for ablation runs only
                ablation_runs = [
                    run for run in project_data.get("runs", [])
                    if run.get("ablation_parsed") is not None
                ]
                
                if ablation_runs:
                    ablations_only.append({
                        "project": project_data["project"],
                        "benchmark": project_data["benchmark"],
                        "modality": project_data["modality"],
                        "ablation_runs": ablation_runs,
                        "ablation_summary": project_data.get("ablation_summary", {})
                    })
    
    # Save ablations-only dataset
    ablations_file = "ablation_analysis/raw_data/ablations_only.json"
    os.makedirs(os.path.dirname(ablations_file), exist_ok=True)
    with open(ablations_file, 'w') as f:
        json.dump(ablations_only, f, indent=2, default=str)
    
    logger.info(f"✓ Ablations-only dataset saved to {ablations_file}")
    logger.info(f"  Contains {len(ablations_only)} projects with ablation studies")
    
    logger.info("\n✅ Extraction pipeline complete!")
    logger.info(f"Data saved in: ablation_analysis/raw_data/")

if __name__ == "__main__":
    main()