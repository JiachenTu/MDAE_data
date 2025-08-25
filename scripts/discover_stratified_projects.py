#!/usr/bin/env python3
"""
Discover stratified projects (without july prefix)
"""

import wandb

# Initialize API
api = wandb.Api()
entity = "t-jiachentu"

print("Searching for stratified projects...")
print("="*60)

# Get all projects
projects = api.projects(entity)

# Look for stratified projects
stratified_projects = []
for project in projects:
    if 'stratified' in project.name and 'july' not in project.name:
        stratified_projects.append(project.name)
        print(f"Found: {project.name}")

print("\n" + "="*60)
print(f"Total stratified (non-july) projects found: {len(stratified_projects)}")

# Filter for UCSF and BraTS18
print("\nUCSF and BraTS18 projects:")
for proj in stratified_projects:
    if 'ucsf' in proj.lower() or 'brats18' in proj.lower():
        print(f"  - {proj}")