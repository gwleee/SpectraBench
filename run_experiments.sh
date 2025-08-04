#!/bin/bash

# SpectraBench: Experiment Execution Script
# Repository: https://github.com/gwleee/SpectraBench
# 
# This script executes SpectraBench experiments with minimal user interaction.
# The system uses pre-optimized configurations from Phase 1 and Phase 2 research.

set -e  # Exit on any error

# Color definitions for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BOLD}$1${NC}"
}

# Configuration
QUICK_MODE=false
COMPARISON_MODE=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --comparison)
                COMPARISON_MODE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --quick        Quick test mode (limit=2, faster execution)"
    echo "  --comparison   Run both baseline and optimized modes for comparison"
    echo "  --help         Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                    # Standard test run (baseline mode)"
    echo "  $0 --quick            # Quick validation (limit=2)"
    echo "  $0 --comparison       # Full comparison (baseline vs optimized)"
    echo ""
    echo "Executes SpectraBench experiments using pre-optimized configurations."
}

# Pre-flight checks
preflight_checks() {
    log_header "Performing pre-flight checks..."
    
    # Check if we're in the right directory
    if [[ ! -f "entrance.py" ]] && [[ ! -f "code/experiments/entrance.py" ]]; then
        log_error "SpectraBench entrance.py not found. Please run from SpectraBench directory."
        exit 1
    fi
    
    # Check if virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warning "Virtual environment not detected. Attempting to activate..."
        if [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
            log_success "Virtual environment activated"
        else
            log_error "Virtual environment not found. Please run install.sh first."
            exit 1
        fi
    fi
    
    # Check GPU availability for comparison mode
    if [[ "$COMPARISON_MODE" == true ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            log_warning "NVIDIA GPU not detected. Comparison mode may take significantly longer."
            read -p "Continue with CPU-only mode? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Execution cancelled by user."
                exit 0
            fi
        fi
    fi
    
    log_success "Pre-flight checks completed"
}

# Create input files for automated execution
create_input_files() {
    local mode="$1"
    local full_run="$2"
    local experiment_mode="$3"
    
    local input_file="auto_input_${mode}_${TIMESTAMP}.txt"
    
    {
        echo "all"          # Select all models
        echo "all"          # Select all tasks
        echo "$full_run"    # Full run (y/n)
        echo "$experiment_mode"  # Experiment mode (1/2/3)
        if [[ "$experiment_mode" == "3" ]]; then
            echo "y"    # Confirm comparison mode
        fi
    } > "$input_file"
    
    echo "$input_file"
}

# Run test experiment
run_test_experiment() {
    log_header "Running Test Experiment (Quick Validation)"
    
    local full_run="n"
    if [[ "$QUICK_MODE" == false ]]; then
        full_run="y"
    fi
    
    local input_file=$(create_input_files "test" "$full_run" "1")
    
    log_info "Running SpectraBench in baseline mode..."
    log_info "Input configuration: all models, all tasks, full_run=$full_run, baseline mode"
    
    if [[ -f "entrance.py" ]]; then
        python entrance.py < "$input_file" 2>&1 | tee "test_execution_${TIMESTAMP}.log"
    else
        python code/experiments/entrance.py < "$input_file" 2>&1 | tee "test_execution_${TIMESTAMP}.log"
    fi
    
    # Cleanup input file
    rm -f "$input_file"
    
    if [[ $? -eq 0 ]]; then
        log_success "Test experiment completed successfully"
        return 0
    else
        log_error "Test experiment failed"
        return 1
    fi
}

# Run comparison experiment
run_comparison_experiment() {
    log_header "Running Comparison Experiment (Baseline vs Optimized)"
    
    local full_run="y"
    if [[ "$QUICK_MODE" == true ]]; then
        full_run="n"
    fi
    
    local input_file=$(create_input_files "comparison" "$full_run" "3")
    
    log_info "Running SpectraBench comparison mode..."
    log_info "This will execute both baseline and optimized modes sequentially"
    log_info "Input configuration: all models, all tasks, full_run=$full_run, comparison mode"
    
    local estimated_time="4-8 hours"
    if [[ "$QUICK_MODE" == true ]]; then
        estimated_time="30-60 minutes"
    fi
    log_info "Estimated completion time: $estimated_time"
    
    if [[ -f "entrance.py" ]]; then
        python entrance.py < "$input_file" 2>&1 | tee "comparison_execution_${TIMESTAMP}.log"
    else
        python code/experiments/entrance.py < "$input_file" 2>&1 | tee "comparison_execution_${TIMESTAMP}.log"
    fi
    
    # Cleanup input file
    rm -f "$input_file"
    
    if [[ $? -eq 0 ]]; then
        log_success "Comparison experiment completed successfully"
        return 0
    else
        log_error "Comparison experiment failed"
        return 1
    fi
}

# Generate analysis report
generate_analysis() {
    log_header "Generating Analysis Report"
    
    # Find the latest experiment directory
    LATEST_EXP_DIR=$(find experiments_results -name "exp_*" -type d | sort | tail -1 2>/dev/null)
    
    if [[ -z "$LATEST_EXP_DIR" ]]; then
        log_warning "No experiment results found in experiments_results/"
        return
    fi
    
    log_info "Latest experiment: $LATEST_EXP_DIR"
    
    # Try to run comparison analyzer if available
    if [[ -f "code/experiments/comparison_analyzer.py" ]]; then
        log_info "Running comparative analysis..."
        python code/experiments/comparison_analyzer.py 2>&1 | tee "analysis_${TIMESTAMP}.log" || true
    fi
    
    # Create simple summary
    local summary_file="experiment_summary_${TIMESTAMP}.txt"
    {
        echo "SpectraBench Experiment Summary"
        echo "==============================="
        echo "Generated: $(date)"
        echo "Experiment Directory: $LATEST_EXP_DIR"
        echo ""
        
        if [[ -d "$LATEST_EXP_DIR/model_results" ]]; then
            echo "Model Results:"
            find "$LATEST_EXP_DIR/model_results" -name "*.json" -type f | while read -r file; do
                model_name=$(basename "$(dirname "$file")")
                echo "  - $model_name"
            done
        fi
        
        echo ""
        echo "Performance Database:"
        if [[ -f "data/performanceDB/performance_history.db" ]]; then
            echo "  - Location: data/performanceDB/performance_history.db"
            if command -v sqlite3 &> /dev/null; then
                total_records=$(sqlite3 data/performanceDB/performance_history.db "SELECT COUNT(*) FROM performance_records;" 2>/dev/null || echo "N/A")
                echo "  - Total records: $total_records"
            fi
        else
            echo "  - No performance database found"
        fi
        
        echo ""
        echo "Log Files:"
        echo "  - Execution log: *_execution_${TIMESTAMP}.log"
        echo "  - Analysis log: analysis_${TIMESTAMP}.log"
        echo "  - Debug log: debug.log"
        
    } > "$summary_file"
    
    log_success "Analysis completed. Summary saved to: $summary_file"
}

# Display results summary
display_results_summary() {
    log_header "=== EXPERIMENT RESULTS SUMMARY ==="
    echo ""
    echo "Execution completed: $(date)"
    echo "Mode: $(if [[ "$COMPARISON_MODE" == true ]]; then echo "Comparison"; else echo "Test"; fi)"
    echo "Quick mode: $(if [[ "$QUICK_MODE" == true ]]; then echo "Yes"; else echo "No"; fi)"
    echo ""
    
    # Find results
    LATEST_EXP_DIR=$(find experiments_results -name "exp_*" -type d | sort | tail -1 2>/dev/null)
    
    echo "RESULTS LOCATION:"
    echo "=================="
    if [[ -n "$LATEST_EXP_DIR" ]]; then
        echo "- Main Results: $LATEST_EXP_DIR"
        echo "- Model Results: $LATEST_EXP_DIR/model_results/"
        if [[ -d "$LATEST_EXP_DIR/config" ]]; then
            echo "- Configuration: $LATEST_EXP_DIR/config/"
        fi
    else
        echo "- Results: Check experiments_results/ directory"
    fi
    
    if [[ -f "data/performanceDB/performance_history.db" ]]; then
        echo "- Performance Database: data/performanceDB/performance_history.db"
    fi
    
    echo "- Legacy Results: results/ (individual model directories)"
    echo ""
    
    echo "LOG FILES:"
    echo "=========="
    echo "- Execution logs: *_execution_${TIMESTAMP}.log"
    echo "- Analysis logs: analysis_${TIMESTAMP}.log"
    echo "- Debug log: debug.log"
    echo ""
    
    echo "NEXT STEPS:"
    echo "==========="
    echo "1. Review the execution logs for detailed results"
    echo "2. Check model results in the experiment directory"
    echo "3. Query the performance database:"
    echo "   sqlite3 data/performanceDB/performance_history.db"
    echo ""
    
    if [[ "$COMPARISON_MODE" == true ]]; then
        echo "COMPARISON ANALYSIS:"
        echo "==================="
        echo "Both baseline and optimized modes have been executed."
        echo "Use the performance database to compare scheduling efficiency."
        echo ""
    fi
    
    log_success "SpectraBench experiments completed successfully!"
}

# Main execution function
main() {
    log_header "SpectraBench Experiment Execution"
    echo "Repository: https://github.com/gwleee/SpectraBench"
    echo "Intelligent LLM Benchmark Scheduler with Adaptive Resource Management"
    echo ""
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Show execution plan
    if [[ "$COMPARISON_MODE" == true ]]; then
        log_info "Mode: Comparison (Baseline + Optimized)"
        if [[ "$QUICK_MODE" == true ]]; then
            log_info "Quick mode: Yes (limit=2, estimated 30-60 minutes)"
        else
            log_info "Full mode: Yes (full dataset, estimated 4-8 hours)"
        fi
    else
        log_info "Mode: Test (Baseline only)"
        if [[ "$QUICK_MODE" == true ]]; then
            log_info "Quick mode: Yes (limit=2, estimated 15-30 minutes)"
        else
            log_info "Standard mode: Yes (full dataset, estimated 2-4 hours)"
        fi
    fi
    
    echo ""
    read -p "Continue with experiment execution? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Experiment execution cancelled by user."
        exit 0
    fi
    
    # Execute experiment pipeline
    preflight_checks
    
    if [[ "$COMPARISON_MODE" == true ]]; then
        if run_comparison_experiment; then
            log_success "Comparison experiment completed successfully"
        else
            log_error "Comparison experiment failed"
            exit 1
        fi
    else
        if run_test_experiment; then
            log_success "Test experiment completed successfully"
        else
            log_error "Test experiment failed"
            exit 1
        fi
    fi
    
    # Generate analysis and summary
    generate_analysis
    display_results_summary
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi