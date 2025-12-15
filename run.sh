#!/bin/bash

################################################################################
# Complete Validation Pipeline for F1 Strategy Optimizer
# 
# This script runs the entire statistical validation workflow:
# 1. Batch testing on multiple circuits
# 2. Statistical analysis
# 3. Plot generation
#
# Usage:
#   chmod +x run_complete_validation.sh
#   ./run_complete_validation.sh
################################################################################

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  F1 STRATEGY OPTIMIZER - COMPLETE VALIDATION PIPELINE            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
YEAR=2024
RUNS=5
MAX_CIRCUITS=""  # Set to empty for all circuits
PYTHON=python3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# STEP 1: BATCH TESTING
################################################################################

echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo "${BLUE}STEP 1: BATCH TESTING${NC}"
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration:"
echo "  Year:           $YEAR"
echo "  Runs per circuit: $RUNS"
echo "  Max circuits:   ${MAX_CIRCUITS:-all}"
echo ""
echo "${YELLOW}⏳ Starting batch test... This may take 30-60 minutes${NC}"
echo ""

# Build command
CMD="$PYTHON batch_test.py --year $YEAR --runs $RUNS"
if [ ! -z "$MAX_CIRCUITS" ]; then
    CMD="$CMD --max $MAX_CIRCUITS"
fi

# Run batch test
if $CMD; then
    echo ""
    echo "${GREEN}✓ Batch testing completed successfully${NC}"
else
    echo ""
    echo "${RED}✗ Batch testing failed. Check errors above.${NC}"
    exit 1
fi

################################################################################
# STEP 2: FIND LATEST RESULTS FILE
################################################################################

echo ""
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo "${BLUE}STEP 2: LOCATING RESULTS${NC}"
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Find the most recent results file
RESULTS_FILE=$(ls -t results/batch_results_${YEAR}_*.json 2>/dev/null | grep -v partial | head -1)

if [ -z "$RESULTS_FILE" ]; then
    echo "${RED}✗ No results file found in results/ directory${NC}"
    exit 1
fi

echo "${GREEN}✓ Found results file: $RESULTS_FILE${NC}"

# Check if CSV was also created
CSV_FILE=$(ls -t results/batch_results_${YEAR}_*.csv 2>/dev/null | grep -v partial | head -1)
if [ ! -z "$CSV_FILE" ]; then
    echo "${GREEN}✓ Found CSV file: $CSV_FILE${NC}"
fi

################################################################################
# STEP 3: STATISTICAL ANALYSIS
################################################################################

echo ""
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo "${BLUE}STEP 3: STATISTICAL ANALYSIS${NC}"
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "${YELLOW}⏳ Running statistical tests...${NC}"
echo ""

if $PYTHON statistical_analysis.py "$RESULTS_FILE"; then
    echo ""
    echo "${GREEN}✓ Statistical analysis completed${NC}"
    echo "${GREEN}  Report saved to: statistical_report.txt${NC}"
else
    echo ""
    echo "${RED}✗ Statistical analysis failed${NC}"
    exit 1
fi

################################################################################
# STEP 4: GENERATE PLOTS
################################################################################

echo ""
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo "${BLUE}STEP 4: GENERATING PLOTS${NC}"
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "${YELLOW}⏳ Creating publication-ready visualizations...${NC}"
echo ""

if $PYTHON generate_thesis_plots.py "$RESULTS_FILE"; then
    echo ""
    echo "${GREEN}✓ All plots generated successfully${NC}"
    echo "${GREEN}  Saved to: plots/ directory${NC}"
else
    echo ""
    echo "${RED}✗ Plot generation failed${NC}"
    exit 1
fi

################################################################################
# STEP 5: SUMMARY
################################################################################

echo ""
echo "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo "${GREEN}║  VALIDATION COMPLETE - ALL STEPS SUCCESSFUL                      ║${NC}"
echo "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "📁 Generated Files:"
echo "   ├── results/"
echo "   │   ├── $RESULTS_FILE"
if [ ! -z "$CSV_FILE" ]; then
    echo "   │   └── $CSV_FILE"
fi
echo "   ├── statistical_report.txt"
echo "   └── plots/"
echo "       ├── fig1_comparison_bar.png"
echo "       ├── fig2_improvement_distribution.png"
echo "       ├── fig3_scatter_correlation.png"
echo "       ├── fig4_ranking_improvement.png"
echo "       ├── fig5_results_table.png"
echo "       └── results_table.tex"
echo ""

# Extract key statistics from report (if available)
if [ -f "statistical_report.txt" ]; then
    echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo "${BLUE}KEY FINDINGS${NC}"
    echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Try to extract mean improvement
    MEAN_IMP=$(grep -A 1 "Mean:" statistical_report.txt | head -1 | awk '{print $2}')
    if [ ! -z "$MEAN_IMP" ]; then
        echo "  Mean Improvement: ${GREEN}$MEAN_IMP${NC}"
    fi
    
    # Try to extract p-value
    PVAL=$(grep "p-value:" statistical_report.txt | head -1 | awk '{print $2}')
    if [ ! -z "$PVAL" ]; then
        echo "  p-value:          ${GREEN}$PVAL${NC}"
    fi
    
    # Try to extract Cohen's d
    COHENS=$(grep "Cohen's d:" statistical_report.txt | head -1 | awk '{print $3}')
    if [ ! -z "$COHENS" ]; then
        echo "  Cohen's d:        ${GREEN}$COHENS${NC}"
    fi
    
    echo ""
    echo "📄 For full details, see: statistical_report.txt"
fi

echo ""
echo "${GREEN}🎓 Your thesis data is ready!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review statistical_report.txt"
echo "  2. Check plots/ directory for figures"
echo "  3. Use results_table.tex in your LaTeX thesis"
echo "  4. Write your Results and Discussion sections"
echo ""
echo "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"