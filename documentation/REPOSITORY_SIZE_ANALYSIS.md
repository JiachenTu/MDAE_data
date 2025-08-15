# Repository Size Analysis and Git Limits

## Current Repository Status (August 14, 2025)

### Overall Size
- **Working Directory**: 221 MB
- **Git Directory (.git)**: 128 MB
- **Total Repository Size**: ~349 MB
- **Total Files**: 1,029 data files (JSON, CSV, PNG)

### Size Breakdown by Directory
```
raw_data/                   19 MB  (Raw WandB extraction data)
processed_data/             44 MB  (Initial processing results)
processed_data_combined/    14 MB  (Single-modality comprehensive)
processed_data_key_benchmarks/  11 MB  (Key benchmarks for paper)
processed_data_multi_modality_combined/  6.7 MB  (Multi-modality comprehensive)
```

### Git Objects
- **Loose objects**: 1,478 objects, 100.66 MB
- **Packed objects**: 574 objects in 3 packs, 25.18 MB
- **Largest files**: JSON data files (~2.2 MB max)

## Git Repository Size Limits

### GitHub Limits
1. **Repository Size**
   - **Recommended**: < 5 GB
   - **Hard limit**: 100 GB (but performance degrades well before)
   - **Your repo**: 349 MB (7% of recommended limit) ✅

2. **File Size**
   - **Warning at**: 50 MB per file
   - **Hard limit**: 100 MB per file (blocks push)
   - **Your largest file**: 2.2 MB ✅

3. **Push Size**
   - **Single push limit**: 2 GB
   - **Your recent push**: ~66 MB ✅

### GitHub Large File Storage (LFS)
For files > 100 MB, GitHub requires Git LFS:
- **Free tier**: 1 GB storage, 1 GB/month bandwidth
- **Not needed for your repo** (all files < 3 MB)

### Performance Considerations
1. **Clone Speed**
   - Current size (349 MB): Fast clone (~1-2 minutes on average connection)
   - At 1 GB: Moderate (~5-10 minutes)
   - At 5 GB: Slow (15-30 minutes)

2. **Operation Speed**
   - Git operations slow down with:
     - Many files (you have ~1,000 - still OK)
     - Large binary files (your PNGs are small)
     - Deep history (can be managed with shallow clones)

## Recommendations

### Current Status: ✅ HEALTHY
Your repository is well within all limits and best practices.

### Space Optimization (if needed in future)
1. **Remove old processed data versions**
   ```bash
   # Clean up old processed_data folders if not needed
   rm -rf processed_data/old_versions
   ```

2. **Compress large JSON files**
   ```bash
   # Compress JSON files > 1MB
   gzip raw_data/*/*.json
   ```

3. **Use .gitignore for temporary files**
   ```bash
   # Add to .gitignore
   *.tmp
   *.cache
   __pycache__/
   ```

4. **Clean Git history (if it grows)**
   ```bash
   # Remove deleted files from history
   git gc --aggressive --prune=now
   ```

5. **Archive old raw data**
   - Move old raw_data folders to cloud storage
   - Keep only latest extractions in Git

### Monitoring Commands
```bash
# Check repository size
du -sh .

# Check Git objects
git count-objects -vH

# Find large files
find . -type f -size +10M -exec ls -lh {} \;

# Check branch size before push
git diff --stat origin/main

# Clean up unnecessary files
git gc
git prune
```

## Growth Projections

At current growth rate:
- **Per analysis run**: ~20-30 MB added
- **Monthly growth** (10 analyses): ~250 MB
- **Time to 5 GB limit**: ~18 months
- **Time to 100 MB file limit**: Unlikely (largest files are 2.2 MB)

## Conclusion

Your repository is in excellent health:
- ✅ Only 7% of GitHub's recommended size
- ✅ No files near size limits
- ✅ Good structure with organized directories
- ✅ Reasonable number of files
- ✅ Fast clone and operation times

No immediate action needed. Consider archiving old raw_data folders after 6-12 months to keep repository lean.