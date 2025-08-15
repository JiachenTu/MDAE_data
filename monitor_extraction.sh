#!/bin/bash
# Monitor extraction progress

while true; do
    clear
    echo "=================================="
    echo "WandB Data Extraction Progress"
    echo "=================================="
    echo ""
    
    # Count completed projects
    COMPLETED=$(ls -1 /home/jtu9/Documents/MDAE/MDAE_data/raw_data_full/20250814/ 2>/dev/null | wc -l)
    TOTAL=306
    PERCENTAGE=$((COMPLETED * 100 / TOTAL))
    
    echo "Projects completed: $COMPLETED / $TOTAL ($PERCENTAGE%)"
    echo ""
    
    # Show recent projects
    echo "Recently extracted projects:"
    ls -t /home/jtu9/Documents/MDAE/MDAE_data/raw_data_full/20250814/ 2>/dev/null | head -5
    echo ""
    
    # Check if still running
    if ps aux | grep -q "[p]ython scripts/extract_all_wandb_data.py"; then
        echo "Status: RUNNING ✓"
    else
        echo "Status: COMPLETED or STOPPED"
        break
    fi
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 10
done

echo ""
echo "Extraction monitoring completed."
echo "Check /home/jtu9/Documents/MDAE/MDAE_data/raw_data_full/20250814/ for results"