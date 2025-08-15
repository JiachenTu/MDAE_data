# Ablation Study Extraction Plan

## Date: August 14, 2025

## Objective
Extract and analyze comprehensive ablation study data from 61 single-modality WandB projects, focusing on three key ablation types: Noise Corruption, Masking Type, and Masking Ratio.

## Phase 1: Data Extraction Infrastructure

### 1.1 Create Enhanced Extraction Script
**File**: `extract_ablations_comprehensive.py`

**Key Features**:
- Parallel extraction using ThreadPoolExecutor
- Full Notes field extraction (critical)
- Robust error handling and retry logic
- Progress tracking with detailed logging
- Save raw and parsed data separately

**Data Structure**:
```python
{
    "project": "july_stratified_brats18_lgg_vs_hgg_single_flair",
    "benchmark": "brats18_lgg_vs_hgg",
    "modality": "flair",
    "runs": [
        {
            "id": "run_id",
            "name": "run_name",
            "notes": "full_notes_string",  # CRITICAL FIELD
            "state": "finished",
            "created_at": "timestamp",
            "config": {...},
            "summary": {
                "Test_AUROC": 0.85,
                "Test_AP": 0.82,
                "Val_AUROC": 0.83,
                # All available metrics
            },
            "ablation_parsed": {  # Parsed from Notes
                "type": "noise_corruption|masking_type|masking_ratio",
                "noise_type": "VE|VP|Flow",
                "masking_type": "Random|Block|Tube",
                "masking_ratio": 75-95,
                "other_params": {}
            }
        }
    ]
}
```

### 1.2 Projects to Extract
**Total**: 61 single-modality projects with pattern `july_stratified_*_single_*`

**Benchmarks** (15 total):
1. BraTS18: 4 projects (flair, t1, t1ce, t2)
2. BraTS23 (3 tasks × 4 modalities = 12 projects)
3. RSNA-MICCAI: 4 projects
4. TCGA-GBM (2 tasks): 7 projects
5. UCSF-PDGM: 6 projects
6. UPenn-GBM (7 tasks × 4 modalities = 28 projects)

## Phase 2: Ablation Parsing Logic

### 2.1 Noise Corruption Type Parser
```python
def parse_noise_corruption(notes):
    """Extract noise corruption ablation parameters"""
    pattern = r"MDAE ablation Noise Corruption (VE|VP|Flow) M(\d+)"
    match = re.search(pattern, notes)
    if match:
        return {
            "type": "noise_corruption",
            "noise_type": match.group(1),
            "masking_ratio": int(match.group(2))
        }
```

### 2.2 Masking Type Parser
```python
def parse_masking_type(notes):
    """Extract masking type ablation parameters"""
    if "Masking Type Ablation:" in notes:
        masking_type = None
        if "Random patch" in notes:
            masking_type = "Random"
        elif "Block" in notes or "blocky" in notes:
            masking_type = "Block"
        elif "Tube" in notes:
            masking_type = "Tube"
        
        return {
            "type": "masking_type",
            "masking_type": masking_type,
            "masking_ratio": 90  # Fixed at 90%
        }
```

### 2.3 Masking Ratio Parser
```python
def parse_masking_ratio(notes):
    """Extract masking ratio ablation parameters"""
    pattern = r"Masking Ratio Ablation: (Flow|VE|VP) (\d+)%"
    match = re.search(pattern, notes)
    if match:
        return {
            "type": "masking_ratio",
            "noise_type": match.group(1),
            "masking_ratio": int(match.group(2))
        }
```

### 2.4 FlowMDAE Parser
```python
def parse_flowmdae(notes):
    """Extract FlowMDAE ablation parameters"""
    pattern = r"FlowMDAE Ablation.*M(\d+)_N(\d+)"
    match = re.search(pattern, notes)
    if match:
        return {
            "type": "flowmdae",
            "masking_ratio": int(match.group(1)),
            "noise_level": int(match.group(2))
        }
```

## Phase 3: Analysis Pipeline

### 3.1 Create Analysis Matrices
**File**: `analyze_ablations.py`

**Analysis Types**:

1. **Noise Corruption Analysis**
   - Matrix: Benchmarks × Noise Types (VE, VP, Flow)
   - Fixed masking at 90%
   - Metrics: AUROC, AP differences

2. **Masking Type Analysis**
   - Matrix: Benchmarks × Masking Types (Random, Block, Tube)
   - Fixed masking at 90%
   - Performance comparison

3. **Masking Ratio Analysis**
   - Matrix: Benchmarks × Ratios (75%, 80%, 85%, 90%, 95%)
   - Fixed noise type (Flow)
   - Trend analysis

4. **FlowMDAE Grid Analysis**
   - 2D Grid: Masking × Noise levels
   - Optimal parameter identification

### 3.2 Statistical Analysis
```python
def compute_ablation_statistics(df):
    """Compute statistics for each ablation type"""
    return {
        "mean_performance": df.groupby('ablation_param')['Test_AUROC'].mean(),
        "std_performance": df.groupby('ablation_param')['Test_AUROC'].std(),
        "best_config": df.loc[df['Test_AUROC'].idxmax()],
        "significance_tests": perform_statistical_tests(df)
    }
```

## Phase 4: Visualization Generation

### 4.1 Ablation Performance Charts
- Bar charts comparing noise types
- Line plots for masking ratio trends
- Heatmaps for FlowMDAE parameter grids
- Box plots showing variance across modalities

### 4.2 Summary Tables
- LaTeX tables for paper inclusion
- CSV exports for further analysis
- Markdown summaries for documentation

## Phase 5: Output Structure

```
ablation_analysis/
├── raw_data/
│   ├── extracted_with_notes/       # Complete extractions
│   └── parsed_ablations/           # Parsed ablation data
├── analysis/
│   ├── noise_corruption/
│   │   ├── performance_matrix.csv
│   │   ├── statistical_tests.csv
│   │   └── visualizations/
│   ├── masking_type/
│   │   ├── performance_matrix.csv
│   │   └── visualizations/
│   ├── masking_ratio/
│   │   ├── trend_analysis.csv
│   │   └── visualizations/
│   └── flowmdae/
│       ├── parameter_grid.csv
│       └── optimal_configs.csv
├── reports/
│   ├── ABLATION_RESULTS_SUMMARY.md
│   ├── latex_tables/
│   └── key_findings.txt
└── scripts/
    ├── extract_ablations_comprehensive.py
    ├── analyze_ablations.py
    └── generate_visualizations.py
```

## Phase 6: Implementation Timeline

### Step 1: Extract Complete Data (Priority 1)
- Create extraction script with full Notes field support
- Extract all 61 projects in parallel
- Validate Notes field presence
- Save raw JSON data

### Step 2: Parse Ablation Parameters (Priority 2)
- Apply regex patterns to Notes field
- Categorize runs by ablation type
- Handle edge cases and variations
- Create structured ablation database

### Step 3: Analyze Results (Priority 3)
- Generate performance matrices
- Compute statistical significance
- Identify optimal parameters
- Compare across benchmarks/modalities

### Step 4: Create Visualizations (Priority 4)
- Generate publication-ready figures
- Create comprehensive tables
- Document key findings

## Critical Success Factors

1. **Notes Field Integrity**: Ensure all runs have Notes field extracted
2. **Pattern Matching**: Handle variations in Notes formatting
3. **Complete Coverage**: Extract from all 61 projects
4. **Parallel Processing**: Use threading for efficiency
5. **Error Recovery**: Implement retry logic for API failures

## Expected Outcomes

1. **Complete Ablation Database**: ~500+ ablation runs across all benchmarks
2. **Optimal Parameters**: Best noise type, masking type, and ratios per benchmark
3. **Statistical Significance**: Which ablations matter most
4. **Publication Assets**: Tables and figures for paper

## Quality Checks

1. Verify Notes field is non-empty for ablation runs
2. Cross-reference with discovered counts (165 in BraTS18)
3. Validate parsed parameters against known patterns
4. Check for missing benchmarks or modalities
5. Ensure metrics are complete and valid

## Next Immediate Action
Begin with Step 1: Create and run the comprehensive extraction script focusing on complete Notes field extraction from all 61 single-modality projects.