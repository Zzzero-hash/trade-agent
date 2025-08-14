#!/bin/bash

# Git History Cleanup Script
# This script helps remove large files from Git history to reduce repository size

echo "⚠️  WARNING: This script will rewrite Git history!"
echo "   Make sure you have backups and coordinate with your team."
echo "   Only run this on repositories where you can force-push safely."
echo ""

read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "🧹 Starting Git history cleanup..."

# Method 1: Using git filter-repo (recommended)
if command -v git-filter-repo &> /dev/null; then
    echo "Using git-filter-repo for cleanup..."
    
    # Remove large files from history
    git filter-repo --path models/rl/best_model.zip --invert-paths
    git filter-repo --path models/rl/sac.zip --invert-paths 
    git filter-repo --path models/rl/ppo_final.zip --invert-paths
    git filter-repo --path attribution_dashboard.html --invert-paths
    git filter-repo --path locust-report.html --invert-paths
    git filter-repo --path pattern_comparison.png --invert-paths
    
    # Remove test files with large policy states
    git filter-repo --path-glob 'file:/tmp/pytest-*' --invert-paths
    
else
    echo "git-filter-repo not found. Install it with:"
    echo "pip install git-filter-repo"
    echo ""
    echo "Or use BFG Repo-Cleaner as an alternative:"
    echo "1. Download bfg.jar from https://rtyley.github.io/bfg-repo-cleaner/"
    echo "2. Run: java -jar bfg.jar --strip-blobs-bigger-than 50M"
    echo "3. Run: git reflog expire --expire=now --all"
    echo "4. Run: git gc --prune=now --aggressive"
    exit 1
fi

# Clean up refs and force garbage collection
echo "🗑️  Cleaning up Git references..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Show new repository size
echo "📊 Repository size after cleanup:"
du -sh .git/

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "⚠️  IMPORTANT: You need to force-push to update the remote repository:"
echo "   git push --force-with-lease origin --all"
echo ""
echo "   Make sure to coordinate with your team as this will rewrite history."
