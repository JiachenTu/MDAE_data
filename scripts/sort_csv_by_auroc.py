#!/usr/bin/env python3
"""
Sort CSV files by Test_AUROC in descending order (best performance first).
"""

import pandas as pd
from pathlib import Path


def sort_csv_by_auroc(csv_path: Path) -> bool:
    """
    Sort a CSV file by Test_AUROC in descending order.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        if 'Test_AUROC' not in df.columns:
            print(f"Warning: Test_AUROC column not found in {csv_path}")
            return False
        
        # Sort by Test_AUROC in descending order (best performance first)
        # NaN values will naturally appear at the end when sorting descending
        df_sorted = df.sort_values('Test_AUROC', ascending=False)
        
        # Save the sorted CSV
        df_sorted.to_csv(csv_path, index=False)
        
        # Count non-null AUROC values for summary
        valid_auroc = df_sorted['Test_AUROC'].notna().sum()
        total_rows = len(df_sorted)
        
        print(f"Sorted {csv_path.parent.name}: {total_rows} rows, {valid_auroc} with valid AUROC")
        return True
        
    except Exception as e:
        print(f"Error sorting {csv_path}: {e}")
        return False


def main():
    """Sort all CSV files by Test_AUROC."""
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "raw_data" / "20250811"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Sorting CSV files by Test_AUROC (descending order)...")
    print("=" * 60)
    
    successful = 0
    total = 0
    
    for benchmark_dir in data_dir.iterdir():
        if benchmark_dir.is_dir():
            csv_path = benchmark_dir / "runs_summary.csv"
            if csv_path.exists():
                total += 1
                if sort_csv_by_auroc(csv_path):
                    successful += 1
                else:
                    print(f"Failed to sort: {benchmark_dir.name}")
    
    print("\n" + "=" * 60)
    print(f"Sorting complete: {successful}/{total} CSV files sorted successfully")


if __name__ == "__main__":
    main()