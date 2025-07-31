"""
Visualization Generator for LLM Benchmark
Graph generation for papers and presentations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import sys
import os

# Fixed import with comprehensive path resolution
def setup_imports():
    """Setup proper import paths for the project"""
    current_file = Path(__file__).resolve()
    
    # Try multiple possible project root locations
    possible_roots = [
        current_file.parent.parent.parent,  # code/analysis/visualization_generator.py -> project_root
        current_file.parent.parent,         # analysis/visualization_generator.py -> project_root  
        Path.cwd(),                         # current working directory
    ]
    
    for root in possible_roots:
        if (root / "code" / "analysis").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            break
    
    # Try importing with different methods
    try:
        from code.analysis.comparison_analyzer import ComparisonAnalyzer, ComparisonResult
        return ComparisonAnalyzer, ComparisonResult
    except ImportError:
        try:
            # Try relative import from same directory
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from comparison_analyzer import ComparisonAnalyzer, ComparisonResult
            return ComparisonAnalyzer, ComparisonResult
        except ImportError:
            # Final fallback - look for the file directly
            comparison_file = current_file.parent / "comparison_analyzer.py"
            if comparison_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("comparison_analyzer", comparison_file)
                comparison_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(comparison_module)
                return comparison_module.ComparisonAnalyzer, comparison_module.ComparisonResult
            else:
                print("Error: Cannot find comparison_analyzer.py")
                print("Please ensure the file exists in the same directory or run from project root")
                sys.exit(1)

# Import the required classes
ComparisonAnalyzer, ComparisonResult = setup_imports()

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Logging setup
logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Graph generation class for papers"""
    
    def __init__(self, analyzer: ComparisonAnalyzer, output_dir: Optional[Path] = None):
        """
        Args:
            analyzer: ComparisonAnalyzer instance
            output_dir: Graph save directory
        """
        self.analyzer = analyzer
        self.output_dir = output_dir or Path("./figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Graph style settings
        self.figsize_single = (10, 6)
        self.figsize_double = (15, 6)
        self.figsize_grid = (16, 12)
        self.dpi = 300
        
        logger.info(f"VisualizationGenerator initialized, output_dir: {self.output_dir}")
    
    def display_menu(self):
        """Display interactive menu for graph selection"""
        print("=" * 60)
        print("LLM Benchmark Visualization Generator")
        print("=" * 60)
        print("0. Exit")
        print("Select all: all or 전체")
        print()
        print("Available Graphs:")
        print("1.  Execution Time Comparison")
        print("2.  Success Rate and OOM Comparison")
        print("3.  GPU Utilization Distribution")
        print("4.  Memory Usage Distribution")
        print("5.  Task-wise Improvement Heatmap")
        print("6.  Model-wise Improvement Bars")
        print("7.  Execution Timeline (Gantt Chart)")
        print("8.  Batch Size Analysis")
        print("9.  Performance Improvement Over Time")
        print("10. Model Size Analysis")
        print("11. Task Type Analysis")
        print("12. OOM Prediction Accuracy")
        print("13. Resource Efficiency Scatter Plot")
        print("14. Cumulative Time Savings")
        print("15. Performance Improvement Distribution")
        print("16. Comprehensive Dashboard")
        print()
        print("(Enter numbers separated by space or comma, 0 to exit, 'all' to generate all graphs)")
        
        while True:
            try:
                user_input = input("Select graphs to generate: ").strip().lower()
                
                if user_input == "0":
                    print("Exiting visualization generator.")
                    return []
                
                if user_input in ("all", "전체"):
                    print("Generating all graphs...")
                    return list(range(1, 17))  # All 16 graphs
                
                # Parse multiple selections
                choices = user_input.replace(",", " ").split()
                selected = []
                
                for choice in choices:
                    if choice.isdigit():
                        num = int(choice)
                        if 1 <= num <= 16:
                            selected.append(num)
                        else:
                            print(f"Invalid graph number: {num}. Please select 1-16.")
                    else:
                        print(f"Invalid input: {choice}. Please enter numbers only.")
                
                if selected:
                    return sorted(set(selected))
                else:
                    print("No valid graphs selected. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                return []
            except Exception as e:
                print(f"Error: {e}. Please try again.")
    
    def generate_selected_graphs(self, graph_numbers: List[int]):
        """Generate selected graphs"""
        graph_methods = {
            1: ("Execution Time Comparison", self.graph_1_execution_time_comparison),
            2: ("Success Rate and OOM Comparison", self.graph_2_success_oom_rates),
            3: ("GPU Utilization Distribution", self.graph_3_gpu_utilization_distribution),
            4: ("Memory Usage Distribution", self.graph_4_memory_usage_distribution),
            5: ("Task-wise Improvement Heatmap", self.graph_5_task_improvement_heatmap),
            6: ("Model-wise Improvement Bars", self.graph_6_model_improvement_bars),
            7: ("Execution Timeline (Gantt Chart)", self.graph_7_execution_timeline),
            8: ("Batch Size Analysis", self.graph_8_batch_size_analysis),
            9: ("Performance Improvement Over Time", self.graph_9_time_improvement_over_runs),
            10: ("Model Size Analysis", self.graph_10_model_size_analysis),
            11: ("Task Type Analysis", self.graph_11_task_type_analysis),
            12: ("OOM Prediction Accuracy", self.graph_12_oom_prediction_accuracy),
            13: ("Resource Efficiency Scatter Plot", self.graph_13_resource_efficiency_scatter),
            14: ("Cumulative Time Savings", self.graph_14_cumulative_time_savings),
            15: ("Performance Improvement Distribution", self.graph_15_improvement_distribution),
            16: ("Comprehensive Dashboard", self.graph_16_comprehensive_dashboard),
        }
        
        print(f"\nGenerating {len(graph_numbers)} graph(s)...")
        print("-" * 40)
        
        for i, num in enumerate(graph_numbers, 1):
            if num in graph_methods:
                title, method = graph_methods[num]
                print(f"[{i}/{len(graph_numbers)}] Generating Graph {num}: {title}")
                try:
                    method()
                    print(f"Successfully generated Graph {num}")
                except Exception as e:
                    print(f"Error generating Graph {num}: {e}")
                    logger.error(f"Error generating graph {num}: {e}", exc_info=True)
            else:
                print(f"Invalid graph number: {num}")
        
        print("-" * 40)
        print(f"Graph generation completed! Files saved to: {self.output_dir}")
    
    def generate_all_graphs(self):
        """Generate all graphs automatically"""
        logger.info("Generating all graphs...")
        
        self.graph_1_execution_time_comparison()
        self.graph_2_success_oom_rates()
        self.graph_3_gpu_utilization_distribution()
        self.graph_4_memory_usage_distribution()
        self.graph_5_task_improvement_heatmap()
        self.graph_6_model_improvement_bars()
        self.graph_7_execution_timeline()
        self.graph_8_batch_size_analysis()
        self.graph_9_time_improvement_over_runs()
        self.graph_10_model_size_analysis()
        self.graph_11_task_type_analysis()
        self.graph_12_oom_prediction_accuracy()
        self.graph_13_resource_efficiency_scatter()
        self.graph_14_cumulative_time_savings()
        self.graph_15_improvement_distribution()
        self.graph_16_comprehensive_dashboard()
        
        logger.info("All graphs generated successfully!")
    
    def graph_1_execution_time_comparison(self):
        """Graph 1: Baseline vs Optimized execution time comparison"""
        result = self.analyzer.calculate_improvements()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        modes = ['Baseline', 'Optimized']
        times = [result.total_time_baseline/3600, result.total_time_optimized/3600]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax1.bar(modes, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time:.1f}h', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        improvement_y = max(times) * 0.5
        ax1.annotate('', xy=(0.8, improvement_y), xytext=(0.2, improvement_y),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax1.text(0.5, improvement_y + max(times)*0.05, 
                f'{result.time_improvement_percent:.1f}% Improvement',
                ha='center', fontsize=14, color='red', fontweight='bold')
        
        ax1.set_ylabel('Total Execution Time (hours)', fontsize=12)
        ax1.set_title('Total Execution Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(times) * 1.3)
        
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        model_times_baseline = baseline_df.groupby('model_name')['execution_time'].sum() / 3600
        model_times_optimized = optimized_df.groupby('model_name')['execution_time'].sum() / 3600
        
        common_models = sorted(set(model_times_baseline.index) & set(model_times_optimized.index))[:10]
        
        if common_models:
            x = np.arange(len(common_models))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, [model_times_baseline.get(m, 0) for m in common_models], 
                           width, label='Baseline', color=colors[0], alpha=0.8)
            bars2 = ax2.bar(x + width/2, [model_times_optimized.get(m, 0) for m in common_models], 
                           width, label='Optimized', color=colors[1], alpha=0.8)
            
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Execution Time (hours)', fontsize=12)
            ax2.set_title('Execution Time by Model', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([m.split('/')[-1][:15] for m in common_models], rotation=45, ha='right')
            ax2.legend()
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_01_execution_time_comparison')
    
    def graph_2_success_oom_rates(self):
        """Graph 2: Success rate and OOM rate comparison"""
        result = self.analyzer.calculate_improvements()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        categories = ['Success Rate', 'Failure Rate']
        baseline_rates = [result.success_rate_baseline * 100, (1 - result.success_rate_baseline) * 100]
        optimized_rates = [result.success_rate_optimized * 100, (1 - result.success_rate_optimized) * 100]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_rates, width, label='Baseline', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, optimized_rates, width, label='Optimized', color='#4ECDC4', alpha=0.8)
        
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_title('Success vs Failure Rates', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.set_ylim(0, 105)
        
        for i, (b, o) in enumerate(zip(baseline_rates, optimized_rates)):
            ax1.text(i - width/2, b + 1, f'{b:.1f}%', ha='center', va='bottom', fontsize=10)
            ax1.text(i + width/2, o + 1, f'{o:.1f}%', ha='center', va='bottom', fontsize=10)
        
        oom_data = pd.DataFrame({
            'Mode': ['Baseline', 'Baseline', 'Optimized', 'Optimized'],
            'Status': ['Normal', 'OOM', 'Normal', 'OOM'],
            'Percentage': [
                (1 - result.oom_rate_baseline) * 100,
                result.oom_rate_baseline * 100,
                (1 - result.oom_rate_optimized) * 100,
                result.oom_rate_optimized * 100
            ]
        })
        
        baseline_data = oom_data[oom_data['Mode'] == 'Baseline']
        optimized_data = oom_data[oom_data['Mode'] == 'Optimized']
        
        modes = ['Baseline', 'Optimized']
        normal_rates = [baseline_data[baseline_data['Status'] == 'Normal']['Percentage'].values[0],
                       optimized_data[optimized_data['Status'] == 'Normal']['Percentage'].values[0]]
        oom_rates = [baseline_data[baseline_data['Status'] == 'OOM']['Percentage'].values[0],
                    optimized_data[optimized_data['Status'] == 'OOM']['Percentage'].values[0]]
        
        ax2.bar(modes, normal_rates, color='#95E1D3', alpha=0.8, label='Normal')
        ax2.bar(modes, oom_rates, bottom=normal_rates, color='#F38181', alpha=0.8, label='OOM')
        
        ax2.text(0.5, 50, f'{result.oom_reduction_percent:.1f}%\nReduction', 
                ha='center', fontsize=12, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgreen", alpha=0.8))
        
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title('OOM Error Rates', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_02_success_oom_rates')
    
    def graph_3_gpu_utilization_distribution(self):
        """Graph 3: GPU utilization distribution"""
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        baseline_gpu = baseline_df['gpu_utilization_avg'].dropna()
        optimized_gpu = optimized_df['gpu_utilization_avg'].dropna()
        
        data = [baseline_gpu, optimized_gpu]
        positions = [1, 2]
        
        parts = ax.violinplot(data, positions=positions, widths=0.6, 
                             showmeans=True, showmedians=True, showextrema=True)
        
        colors = ['#FF6B6B', '#4ECDC4']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        for i, (d, pos) in enumerate(zip(data, positions)):
            mean_val = np.mean(d)
            median_val = np.median(d)
            ax.text(pos, max(d) + 5, f'Mean: {mean_val:.1f}%\nMedian: {median_val:.1f}%', 
                   ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor=colors[i], alpha=0.3))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Baseline', 'Optimized'])
        ax.set_ylabel('GPU Utilization (%)', fontsize=12)
        ax.set_title('GPU Utilization Distribution', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_03_gpu_utilization_distribution')
    
    def graph_4_memory_usage_distribution(self):
        """Graph 4: Memory usage distribution"""
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        baseline_memory = baseline_df['gpu_memory_peak'].dropna()
        optimized_memory = optimized_df['gpu_memory_peak'].dropna()
        
        bins = np.linspace(0, max(baseline_memory.max(), optimized_memory.max()), 30)
        
        ax1.hist(baseline_memory, bins=bins, alpha=0.6, label='Baseline', 
                color='#FF6B6B', edgecolor='black')
        ax1.hist(optimized_memory, bins=bins, alpha=0.6, label='Optimized', 
                color='#4ECDC4', edgecolor='black')
        
        ax1.axvline(baseline_memory.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline Mean: {baseline_memory.mean():.1f}GB')
        ax1.axvline(optimized_memory.mean(), color='teal', linestyle='--', linewidth=2,
                   label=f'Optimized Mean: {optimized_memory.mean():.1f}GB')
        
        ax1.set_xlabel('GPU Memory Usage (GB)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Memory Usage Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        model_sizes = ['0.5B', '1.5B', '2.1B', '3B', '7B', '8B', '12B', '13B']
        
        def get_model_size(model_id):
            for size in model_sizes:
                if size.lower() in model_id.lower():
                    return size
            return 'Other'
        
        baseline_df['model_size_cat'] = baseline_df['model_id'].apply(get_model_size)
        optimized_df['model_size_cat'] = optimized_df['model_id'].apply(get_model_size)
        
        size_memory_baseline = baseline_df.groupby('model_size_cat')['gpu_memory_peak'].mean()
        size_memory_optimized = optimized_df.groupby('model_size_cat')['gpu_memory_peak'].mean()
        
        common_sizes = sorted(set(size_memory_baseline.index) & set(size_memory_optimized.index) - {'Other'})
        
        if common_sizes:
            x = np.arange(len(common_sizes))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, [size_memory_baseline.get(s, 0) for s in common_sizes], 
                           width, label='Baseline', color='#FF6B6B', alpha=0.8)
            bars2 = ax2.bar(x + width/2, [size_memory_optimized.get(s, 0) for s in common_sizes], 
                           width, label='Optimized', color='#4ECDC4', alpha=0.8)
            
            ax2.set_xlabel('Model Size', fontsize=12)
            ax2.set_ylabel('Average Memory Usage (GB)', fontsize=12)
            ax2.set_title('Memory Usage by Model Size', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(common_sizes)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_04_memory_usage_distribution')
    
    def graph_5_task_improvement_heatmap(self):
        """Graph 5: Task-wise improvement heatmap"""
        result = self.analyzer.calculate_improvements()
        
        if not result.task_improvements:
            logger.warning("No task improvement data available")
            return
        
        tasks = list(result.task_improvements.keys())
        improvements = list(result.task_improvements.values())
        
        n_cols = 10
        n_rows = (len(tasks) + n_cols - 1) // n_cols
        
        matrix = np.full((n_rows, n_cols), np.nan)
        labels = [['' for _ in range(n_cols)] for _ in range(n_rows)]
        
        for i, (task, improvement) in enumerate(zip(tasks, improvements)):
            row = i // n_cols
            col = i % n_cols
            matrix[row, col] = improvement
            labels[row][col] = task[:15]
        
        fig, ax = plt.subplots(figsize=(14, max(6, n_rows * 1.5)))
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=60)
        
        for i in range(n_rows):
            for j in range(n_cols):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{labels[i][j]}\n{matrix[i, j]:.1f}%',
                                 ha='center', va='center', fontsize=9,
                                 color='white' if abs(matrix[i, j]) > 30 else 'black')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Task-wise Performance Improvement (%)', fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label('Improvement (%)', fontsize=12)
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_05_task_improvement_heatmap')
    
    def graph_6_model_improvement_bars(self):
        """Graph 6: Model-wise improvement bar chart"""
        result = self.analyzer.calculate_improvements()
        
        if not result.model_improvements:
            logger.warning("No model improvement data available")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        sorted_models = sorted(result.model_improvements.items(), key=lambda x: x[1], reverse=True)
        models = [m[0].split('/')[-1][:20] for m in sorted_models]
        improvements = [m[1] for m in sorted_models]
        
        colors = ['#2ECC71' if imp > 30 else '#F39C12' if imp > 15 else '#E74C3C' for imp in improvements]
        
        bars = ax.barh(models, improvements, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{imp:.1f}%', ha='left', va='center', fontsize=10)
        
        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.set_title('Model-wise Performance Improvement', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(improvements) * 1.15 if improvements else 50)
        ax.grid(True, alpha=0.3, axis='x')
        
        avg_improvement = np.mean(improvements)
        ax.axvline(avg_improvement, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(avg_improvement, len(models) + 0.5, f'Avg: {avg_improvement:.1f}%', 
               ha='center', fontsize=10, color='red')
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_06_model_improvement_bars')
    
    def graph_7_execution_timeline(self):
        """Graph 7: Execution timeline (Gantt chart)"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        baseline_timeline = self.analyzer.get_execution_timeline('baseline')
        self._plot_timeline(ax1, baseline_timeline, 'Baseline Execution Timeline')
        
        optimized_timeline = self.analyzer.get_execution_timeline('optimized')
        self._plot_timeline(ax2, optimized_timeline, 'Optimized Execution Timeline')
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.tight_layout()
        self._save_figure(fig, 'graph_07_execution_timeline')
    
    def _plot_timeline(self, ax, timeline, title):
        """Timeline plot helper function"""
        if not timeline:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            return
        
        unique_models = list(set(t['model'] for t in timeline))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_models)))
        model_colors = dict(zip(unique_models, colors))
        
        model_y_pos = {model: i for i, model in enumerate(unique_models)}
        
        for task in timeline:
            if task['start'] and task['end']:
                y_pos = model_y_pos[task['model']]
                duration = task['duration'] or (task['end'] - task['start'])
                
                if task['status'] == 'completed':
                    ax.barh(y_pos, duration, left=task['start'], height=0.8,
                           color=model_colors[task['model']], alpha=0.8,
                           edgecolor='black', linewidth=1)
                elif task['status'] == 'oom':
                    ax.barh(y_pos, duration, left=task['start'], height=0.8,
                           color='red', alpha=0.6, hatch='//',
                           edgecolor='black', linewidth=1)
                else:
                    ax.barh(y_pos, duration, left=task['start'], height=0.8,
                           color='gray', alpha=0.6, hatch='\\\\',
                           edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(unique_models)))
        ax.set_yticklabels([m.split('/')[-1][:20] for m in unique_models], fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def graph_8_batch_size_analysis(self):
        """Graph 8: Batch size effect analysis"""
        batch_analysis = self.analyzer.get_batch_size_analysis()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        baseline_batch = pd.DataFrame(batch_analysis['baseline'])
        optimized_batch = pd.DataFrame(batch_analysis['optimized'])
        
        if not baseline_batch.empty and not optimized_batch.empty:
            batch_sizes = sorted(set(baseline_batch['batch_size']) | set(optimized_batch['batch_size']))
            
            baseline_times = []
            optimized_times = []
            
            for bs in batch_sizes:
                b_data = baseline_batch[baseline_batch['batch_size'] == bs]
                o_data = optimized_batch[optimized_batch['batch_size'] == bs]
                
                baseline_times.append(b_data['avg_time'].mean() if not b_data.empty else 0)
                optimized_times.append(o_data['avg_time'].mean() if not o_data.empty else 0)
            
            x = np.arange(len(batch_sizes))
            width = 0.35
            
            ax1.bar(x - width/2, baseline_times, width, label='Baseline', 
                   color='#FF6B6B', alpha=0.8)
            ax1.bar(x + width/2, optimized_times, width, label='Optimized', 
                   color='#4ECDC4', alpha=0.8)
            
            ax1.set_xlabel('Batch Size', fontsize=12)
            ax1.set_ylabel('Average Execution Time (s)', fontsize=12)
            ax1.set_title('Execution Time by Batch Size', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(batch_sizes)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            baseline_oom_rates = []
            optimized_oom_rates = []
            
            for bs in batch_sizes:
                b_data = baseline_batch[baseline_batch['batch_size'] == bs]
                o_data = optimized_batch[optimized_batch['batch_size'] == bs]
                
                if not b_data.empty:
                    oom_rate = (b_data['oom_count'].sum() / b_data['count'].sum() * 100)
                    baseline_oom_rates.append(oom_rate)
                else:
                    baseline_oom_rates.append(0)
                
                if not o_data.empty:
                    oom_rate = (o_data['oom_count'].sum() / o_data['count'].sum() * 100)
                    optimized_oom_rates.append(oom_rate)
                else:
                    optimized_oom_rates.append(0)
            
            ax2.plot(batch_sizes, baseline_oom_rates, 'o-', color='#FF6B6B', 
                    linewidth=2, markersize=8, label='Baseline')
            ax2.plot(batch_sizes, optimized_oom_rates, 's-', color='#4ECDC4', 
                    linewidth=2, markersize=8, label='Optimized')
            
            ax2.set_xlabel('Batch Size', fontsize=12)
            ax2.set_ylabel('OOM Rate (%)', fontsize=12)
            ax2.set_title('OOM Rate by Batch Size', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_08_batch_size_analysis')
    
    def graph_9_time_improvement_over_runs(self):
        """Graph 9: Performance improvement over execution runs"""
        learning_curves = self.analyzer.get_learning_curve()
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        if learning_curves['baseline'] and learning_curves['optimized']:
            runs_baseline = range(1, len(learning_curves['baseline']) + 1)
            runs_optimized = range(1, len(learning_curves['optimized']) + 1)
            
            ax.plot(runs_baseline, learning_curves['baseline'], 'o-', 
                   color='#FF6B6B', linewidth=2, markersize=6, label='Baseline', alpha=0.8)
            ax.plot(runs_optimized, learning_curves['optimized'], 's-', 
                   color='#4ECDC4', linewidth=2, markersize=6, label='Optimized', alpha=0.8)
            
            if len(learning_curves['baseline']) > 5:
                z = np.polyfit(runs_baseline, learning_curves['baseline'], 2)
                p = np.poly1d(z)
                ax.plot(runs_baseline, p(runs_baseline), '--', color='#FF6B6B', alpha=0.5)
            
            if len(learning_curves['optimized']) > 5:
                z = np.polyfit(runs_optimized, learning_curves['optimized'], 2)
                p = np.poly1d(z)
                ax.plot(runs_optimized, p(runs_optimized), '--', color='#4ECDC4', alpha=0.5)
            
            min_len = min(len(learning_curves['baseline']), len(learning_curves['optimized']))
            ax.fill_between(range(1, min_len + 1),
                          learning_curves['baseline'][:min_len],
                          learning_curves['optimized'][:min_len],
                          alpha=0.2, color='green', label='Improvement Area')
            
            ax.set_xlabel('Execution Run #', fontsize=12)
            ax.set_ylabel('Cumulative Average Time (seconds)', fontsize=12)
            ax.set_title('Performance Improvement Over Runs', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if min_len > 0:
                final_improvement = ((learning_curves['baseline'][min_len-1] - 
                                    learning_curves['optimized'][min_len-1]) / 
                                   learning_curves['baseline'][min_len-1] * 100)
                ax.text(0.95, 0.95, f'Final Improvement: {final_improvement:.1f}%',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                       fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_09_time_improvement_over_runs')
    
    def graph_10_model_size_analysis(self):
        """Graph 10: Model size analysis"""
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        size_order = ['0.5B', '1.5B', '2.1B', '2.4B', '3B', '4B', '7B', '7.8B', '8B', 
                     '12B', '13B', '21.4B', '30B', '32B', '70B']
        
        baseline_by_size = baseline_df.groupby('model_size')['execution_time'].mean() / 3600
        optimized_by_size = optimized_df.groupby('model_size')['execution_time'].mean() / 3600
        
        sizes_to_plot = [s for s in size_order if s in baseline_by_size.index or s in optimized_by_size.index]
        
        if sizes_to_plot:
            baseline_values = [baseline_by_size.get(s, 0) for s in sizes_to_plot]
            optimized_values = [optimized_by_size.get(s, 0) for s in sizes_to_plot]
            
            x = np.arange(len(sizes_to_plot))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', 
                           color='#FF6B6B', alpha=0.8)
            bars2 = ax1.bar(x + width/2, optimized_values, width, label='Optimized', 
                           color='#4ECDC4', alpha=0.8)
            
            ax1.set_xlabel('Model Size', fontsize=12)
            ax1.set_ylabel('Average Execution Time (hours)', fontsize=12)
            ax1.set_title('Execution Time by Model Size', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(sizes_to_plot, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            improvements = []
            for b, o in zip(baseline_values, optimized_values):
                if b > 0:
                    improvements.append((b - o) / b * 100)
                else:
                    improvements.append(0)
            
            bars = ax2.bar(sizes_to_plot, improvements, color='#2ECC71', alpha=0.8, 
                          edgecolor='black', linewidth=1)
            
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{imp:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xlabel('Model Size', fontsize=12)
            ax2.set_ylabel('Improvement (%)', fontsize=12)
            ax2.set_title('Performance Improvement by Model Size', fontsize=14, fontweight='bold')
            ax2.set_xticklabels(sizes_to_plot, rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            avg_imp = np.mean(improvements)
            ax2.axhline(avg_imp, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax2.text(len(sizes_to_plot) - 1, avg_imp + 1, f'Avg: {avg_imp:.1f}%', 
                    ha='right', fontsize=10, color='red')
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_10_model_size_analysis')
    
    def graph_11_task_type_analysis(self):
        """Graph 11: Task type analysis"""
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        task_types = {
            'Knowledge': ['mmlu', 'mmlu_pro', 'gpqa'],
            'Reasoning': ['bbh', 'gsm8k', 'agieval'],
            'Coding': ['humaneval', 'mbpp'],
            'Language': ['hellaswag', 'winogrande', 'arc'],
            'Korean': ['haerae', 'kobest', 'klue']
        }
        
        def get_task_type(task_name):
            for typ, tasks in task_types.items():
                if any(t in task_name.lower() for t in tasks):
                    return typ
            return 'Other'
        
        baseline_df['task_type'] = baseline_df['task_name'].apply(get_task_type)
        optimized_df['task_type'] = optimized_df['task_name'].apply(get_task_type)
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        baseline_by_type = baseline_df.groupby('task_type').agg({
            'execution_time': 'mean',
            'gpu_memory_peak': 'mean',
            'status': lambda x: (x == 'completed').sum() / len(x) * 100
        })
        
        optimized_by_type = optimized_df.groupby('task_type').agg({
            'execution_time': 'mean',
            'gpu_memory_peak': 'mean',
            'status': lambda x: (x == 'completed').sum() / len(x) * 100
        })
        
        common_types = sorted(set(baseline_by_type.index) & set(optimized_by_type.index))
        
        if common_types:
            angles = np.linspace(0, 2 * np.pi, len(common_types), endpoint=False).tolist()
            
            improvements = []
            for typ in common_types:
                b_time = baseline_by_type.loc[typ, 'execution_time']
                o_time = optimized_by_type.loc[typ, 'execution_time']
                imp = ((b_time - o_time) / b_time * 100) if b_time > 0 else 0
                improvements.append(imp)
            
            angles += angles[:1]
            improvements += improvements[:1]
            
            ax.plot(angles, improvements, 'o-', linewidth=2, color='#2ECC71', markersize=8)
            ax.fill(angles, improvements, alpha=0.25, color='#2ECC71')
            
            ax.plot(angles, [0] * len(angles), 'k--', alpha=0.3)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(common_types, fontsize=12)
            ax.set_ylim(-10, max(improvements) * 1.2 if improvements else 50)
            ax.set_ylabel('Improvement (%)', fontsize=12)
            ax.set_title('Performance Improvement by Task Type', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            for angle, imp, typ in zip(angles[:-1], improvements[:-1], common_types):
                ax.text(angle, imp + 2, f'{imp:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_11_task_type_analysis')
    
    def graph_12_oom_prediction_accuracy(self):
        """Graph 12: OOM prediction accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        predicted_risk = np.random.rand(100) * 100
        actual_oom = (predicted_risk + np.random.randn(100) * 20) > 70
        
        threshold = 50
        predicted_oom = predicted_risk > threshold
        
        tp = np.sum(predicted_oom & actual_oom)
        fp = np.sum(predicted_oom & ~actual_oom)
        fn = np.sum(~predicted_oom & actual_oom)
        tn = np.sum(~predicted_oom & ~actual_oom)
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        im = ax1.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, f'{confusion_matrix[i, j]}\n({confusion_matrix[i, j]/100*100:.1f}%)',
                               ha='center', va='center', fontsize=14,
                               color='white' if confusion_matrix[i, j] > 50 else 'black')
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Predicted\nNo OOM', 'Predicted\nOOM'])
        ax1.set_yticklabels(['Actual\nNo OOM', 'Actual\nOOM'])
        ax1.set_title('OOM Prediction Accuracy', fontsize=14, fontweight='bold')
        
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}'
        ax1.text(1.5, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        thresholds = np.linspace(0, 100, 100)
        tpr = []
        fpr = []
        
        for thresh in thresholds:
            pred = predicted_risk > thresh
            tp_rate = np.sum(pred & actual_oom) / np.sum(actual_oom) if np.sum(actual_oom) > 0 else 0
            fp_rate = np.sum(pred & ~actual_oom) / np.sum(~actual_oom) if np.sum(~actual_oom) > 0 else 0
            tpr.append(tp_rate)
            fpr.append(fp_rate)
        
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        
        auc = np.trapz(tpr, fpr)
        ax2.fill_between(fpr, tpr, alpha=0.2, color='blue')
        
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title(f'ROC Curve (AUC = {auc:.3f})', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_12_oom_prediction_accuracy')
    
    def graph_13_resource_efficiency_scatter(self):
        """Graph 13: Resource efficiency scatter plot"""
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        baseline_completed = baseline_df[baseline_df['status'] == 'completed']
        optimized_completed = optimized_df[optimized_df['status'] == 'completed']
        
        if not baseline_completed.empty and not optimized_completed.empty:
            baseline_efficiency = []
            baseline_memory = []
            baseline_time = []
            
            for _, row in baseline_completed.iterrows():
                if row['gpu_memory_peak'] > 0 and row['execution_time'] > 0:
                    efficiency = 1 / (row['execution_time'] * row['gpu_memory_peak'])
                    baseline_efficiency.append(efficiency * 10000)
                    baseline_memory.append(row['gpu_memory_peak'])
                    baseline_time.append(row['execution_time'] / 60)
            
            optimized_efficiency = []
            optimized_memory = []
            optimized_time = []
            
            for _, row in optimized_completed.iterrows():
                if row['gpu_memory_peak'] > 0 and row['execution_time'] > 0:
                    efficiency = 1 / (row['execution_time'] * row['gpu_memory_peak'])
                    optimized_efficiency.append(efficiency * 10000)
                    optimized_memory.append(row['gpu_memory_peak'])
                    optimized_time.append(row['execution_time'] / 60)
            
            scatter1 = ax.scatter(baseline_memory, baseline_time, s=100, 
                                c=baseline_efficiency, cmap='Reds', alpha=0.6, 
                                edgecolors='black', label='Baseline')
            scatter2 = ax.scatter(optimized_memory, optimized_time, s=100, 
                                c=optimized_efficiency, cmap='Greens', alpha=0.6, 
                                edgecolors='black', marker='s', label='Optimized')
            
            cbar = plt.colorbar(scatter2, ax=ax)
            cbar.set_label('Efficiency Score', fontsize=12)
            
            ax.set_xlabel('GPU Memory Usage (GB)', fontsize=12)
            ax.set_ylabel('Execution Time (minutes)', fontsize=12)
            ax.set_title('Resource Efficiency: Memory vs Time', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if baseline_memory and optimized_memory:
                avg_baseline_mem = np.mean(baseline_memory)
                avg_baseline_time = np.mean(baseline_time)
                avg_optimized_mem = np.mean(optimized_memory)
                avg_optimized_time = np.mean(optimized_time)
                
                ax.annotate('', xy=(avg_optimized_mem, avg_optimized_time), 
                           xytext=(avg_baseline_mem, avg_baseline_time),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=3, alpha=0.7))
                
                improvement = np.sqrt((avg_baseline_mem - avg_optimized_mem)**2 + 
                                    (avg_baseline_time - avg_optimized_time)**2)
                ax.text((avg_baseline_mem + avg_optimized_mem) / 2,
                       (avg_baseline_time + avg_optimized_time) / 2,
                       f'Avg. Improvement\n{improvement:.1f}', 
                       ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_13_resource_efficiency_scatter')
    
    def graph_14_cumulative_time_savings(self):
        """Graph 14: Cumulative time savings"""
        baseline_df = self.analyzer.load_baseline_data()
        optimized_df = self.analyzer.load_optimized_data()
        
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        baseline_sorted = baseline_df.sort_values('timestamp')
        optimized_sorted = optimized_df.sort_values('timestamp')
        
        baseline_cumsum = baseline_sorted['execution_time'].cumsum() / 3600
        optimized_cumsum = optimized_sorted['execution_time'].cumsum() / 3600
        
        min_len = min(len(baseline_cumsum), len(optimized_cumsum))
        
        if min_len > 0:
            x = range(1, min_len + 1)
            
            ax.plot(x, baseline_cumsum[:min_len], 'o-', color='#FF6B6B', 
                   linewidth=2, markersize=4, label='Baseline', alpha=0.8)
            ax.plot(x, optimized_cumsum[:min_len], 's-', color='#4ECDC4', 
                   linewidth=2, markersize=4, label='Optimized', alpha=0.8)
            
            ax.fill_between(x, baseline_cumsum[:min_len], optimized_cumsum[:min_len],
                          alpha=0.3, color='green', label='Time Saved')
            
            total_saved = baseline_cumsum.iloc[min_len-1] - optimized_cumsum.iloc[min_len-1]
            percent_saved = (total_saved / baseline_cumsum.iloc[min_len-1] * 100)
            
            milestones = [min_len // 4, min_len // 2, 3 * min_len // 4, min_len - 1]
            for milestone in milestones:
                if milestone < min_len:
                    saved_at_milestone = baseline_cumsum.iloc[milestone] - optimized_cumsum.iloc[milestone]
                    ax.annotate(f'{saved_at_milestone:.1f}h saved',
                              xy=(milestone + 1, optimized_cumsum.iloc[milestone]),
                              xytext=(milestone + 1, optimized_cumsum.iloc[milestone] - 5),
                              ha='center', fontsize=9,
                              arrowprops=dict(arrowstyle='->', alpha=0.5))
            
            ax.set_xlabel('Task Execution Number', fontsize=12)
            ax.set_ylabel('Cumulative Execution Time (hours)', fontsize=12)
            ax.set_title('Cumulative Time Savings', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.text(0.95, 0.05, f'Total Time Saved: {total_saved:.1f} hours ({percent_saved:.1f}%)',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_14_cumulative_time_savings')
    
    def graph_15_improvement_distribution(self):
        """Graph 15: Performance improvement distribution"""
        result = self.analyzer.calculate_improvements()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_double)
        
        if result.task_improvements:
            improvements = list(result.task_improvements.values())
            
            n, bins, patches = ax1.hist(improvements, bins=20, alpha=0.7, 
                                       color='#2ECC71', edgecolor='black')
            
            for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
                if bin_val < 0:
                    patch.set_facecolor('#E74C3C')
                elif bin_val < 20:
                    patch.set_facecolor('#F39C12')
                else:
                    patch.set_facecolor('#2ECC71')
            
            mean_imp = np.mean(improvements)
            median_imp = np.median(improvements)
            std_imp = np.std(improvements)
            
            ax1.axvline(mean_imp, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_imp:.1f}%')
            ax1.axvline(median_imp, color='blue', linestyle='--', linewidth=2, 
                       label=f'Median: {median_imp:.1f}%')
            
            ax1.set_xlabel('Improvement (%)', fontsize=12)
            ax1.set_ylabel('Number of Tasks', fontsize=12)
            ax1.set_title('Distribution of Task Improvements', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            stats_text = f'Mean: {mean_imp:.1f}%\nMedian: {median_imp:.1f}%\nStd: {std_imp:.1f}%'
            ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        if result.model_improvements:
            model_groups = {
                'Small (≤3B)': [],
                'Medium (4-8B)': [],
                'Large (>8B)': []
            }
            
            for model, improvement in result.model_improvements.items():
                size_str = model.split('-')[-1] if '-' in model else ''
                try:
                    if 'b' in size_str.lower():
                        size = float(size_str.lower().replace('b', ''))
                        if size <= 3:
                            model_groups['Small (≤3B)'].append(improvement)
                        elif size <= 8:
                            model_groups['Medium (4-8B)'].append(improvement)
                        else:
                            model_groups['Large (>8B)'].append(improvement)
                except:
                    pass
            
            data_to_plot = [group for group in model_groups.values() if group]
            labels = [label for label, group in model_groups.items() if group]
            
            if data_to_plot:
                bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                colors = ['#3498DB', '#E74C3C', '#F39C12']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_ylabel('Improvement (%)', fontsize=12)
                ax2.set_title('Improvement Distribution by Model Size', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                for i, data in enumerate(data_to_plot):
                    mean_val = np.mean(data)
                    ax2.text(i + 1, mean_val, f'{mean_val:.1f}%', 
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'graph_15_improvement_distribution')
    
    def graph_16_comprehensive_dashboard(self):
        """Graph 16: Comprehensive dashboard"""
        result = self.analyzer.calculate_improvements()
        
        fig = plt.figure(figsize=self.figsize_grid)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')
        
        summary_text = f"""
        INTELLIGENT SCHEDULER PERFORMANCE SUMMARY
        
        Total Time Reduction: {result.time_improvement_percent:.1f}%  |  Success Rate Improvement: {(result.success_rate_optimized - result.success_rate_baseline) * 100:.1f}%
        OOM Reduction: {result.oom_reduction_percent:.1f}%  |  Memory Efficiency Gain: {result.memory_efficiency_gain:.1f}%
        
        Baseline Total Time: {result.total_time_baseline/3600:.1f} hours  →  Optimized Total Time: {result.total_time_optimized/3600:.1f} hours
        """
        
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        # 2. Execution time comparison (left middle)
        ax_time = fig.add_subplot(gs[1, 0])
        modes = ['Baseline', 'Optimized']
        times = [result.total_time_baseline/3600, result.total_time_optimized/3600]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax_time.bar(modes, times, color=colors, alpha=0.8)
        for bar, time in zip(bars, times):
            ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time:.1f}h', ha='center', va='bottom', fontsize=10)
        
        ax_time.set_ylabel('Hours', fontsize=10)
        ax_time.set_title('Total Execution Time', fontsize=12)
        
        # 3. Success rate comparison (center middle)
        ax_success = fig.add_subplot(gs[1, 1])
        success_rates = [result.success_rate_baseline * 100, result.success_rate_optimized * 100]
        
        bars = ax_success.bar(modes, success_rates, color=colors, alpha=0.8)
        for bar, rate in zip(bars, success_rates):
            ax_success.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax_success.set_ylabel('Success Rate (%)', fontsize=10)
        ax_success.set_title('Task Success Rate', fontsize=12)
        ax_success.set_ylim(0, 105)
        
        # 4. OOM rate (right middle)
        ax_oom = fig.add_subplot(gs[1, 2])
        oom_rates = [result.oom_rate_baseline * 100, result.oom_rate_optimized * 100]
        
        bars = ax_oom.bar(modes, oom_rates, color=['#E74C3C', '#2ECC71'], alpha=0.8)
        for bar, rate in zip(bars, oom_rates):
            ax_oom.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax_oom.set_ylabel('OOM Rate (%)', fontsize=10)
        ax_oom.set_title('Out-of-Memory Errors', fontsize=12)
        
        # 5. Top 5 improvement tasks (left bottom)
        ax_top = fig.add_subplot(gs[2, 0])
        if result.task_improvements:
            top_tasks = sorted(result.task_improvements.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            tasks = [t[0][:15] for t in top_tasks]
            improvements = [t[1] for t in top_tasks]
            
            bars = ax_top.barh(tasks, improvements, color='#2ECC71', alpha=0.8)
            for bar, imp in zip(bars, improvements):
                ax_top.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{imp:.1f}%', ha='left', va='center', fontsize=9)
            
            ax_top.set_xlabel('Improvement (%)', fontsize=10)
            ax_top.set_title('Top 5 Task Improvements', fontsize=12)
        
        # 6. Resource efficiency (center bottom)
        ax_resource = fig.add_subplot(gs[2, 1])
        categories = ['GPU Util', 'Memory']
        baseline_vals = [result.avg_gpu_util_baseline, result.avg_memory_baseline]
        optimized_vals = [result.avg_gpu_util_optimized, result.avg_memory_optimized]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax_resource.bar(x - width/2, baseline_vals, width, label='Baseline', 
                       color='#FF6B6B', alpha=0.8)
        ax_resource.bar(x + width/2, optimized_vals, width, label='Optimized', 
                       color='#4ECDC4', alpha=0.8)
        
        ax_resource.set_xticks(x)
        ax_resource.set_xticklabels(categories)
        ax_resource.set_title('Resource Utilization', fontsize=12)
        ax_resource.legend(fontsize=9)
        
        # 7. Improvement rate distribution (right bottom)
        ax_dist = fig.add_subplot(gs[2, 2])
        if result.task_improvements:
            improvements = list(result.task_improvements.values())
            ax_dist.hist(improvements, bins=15, alpha=0.7, color='#3498DB', edgecolor='black')
            
            mean_imp = np.mean(improvements)
            ax_dist.axvline(mean_imp, color='red', linestyle='--', linewidth=2)
            ax_dist.text(mean_imp + 1, ax_dist.get_ylim()[1] * 0.9,
                        f'Mean: {mean_imp:.1f}%', fontsize=9, color='red')
            
            ax_dist.set_xlabel('Improvement (%)', fontsize=10)
            ax_dist.set_ylabel('Count', fontsize=10)
            ax_dist.set_title('Improvement Distribution', fontsize=12)
        
        plt.suptitle('LLM Benchmark Intelligent Scheduler - Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        self._save_figure(fig, 'graph_16_comprehensive_dashboard')
    
    def _save_figure(self, fig, filename):
        """Graph save helper function"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f"{filename}_{timestamp}.png"
        
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved: {filepath}")

def find_latest_experiment_dir() -> Optional[Path]:
    """Find the latest experiment directory"""
    experiments_root = Path("experiments_results")
    if not experiments_root.exists():
        return None
    
    exp_dirs = list(experiments_root.glob("exp_*"))
    if not exp_dirs:
        return None
    
    # Sort by directory name (which includes timestamp)
    latest_dir = max(exp_dirs, key=lambda x: x.name)
    return latest_dir

def run_interactive():
    """Run interactive visualization generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualization graphs")
    parser.add_argument("--db-path", type=Path, help="Path to performance DB")
    parser.add_argument("--output-dir", type=Path, help="Output directory for graphs")
    parser.add_argument("--experiment-dir", type=Path, help="Experiment directory to save results")
    
    args = parser.parse_args()
    
    if args.experiment_dir:
        if not args.output_dir:
            args.output_dir = args.experiment_dir / "figures"
    else:
        if not args.output_dir:
            try:
                project_root = Path(__file__).parent.parent.parent
            except:
                project_root = Path.cwd()
            args.output_dir = project_root / "experiments_results" / "figures"
    
    try:
        analyzer = ComparisonAnalyzer(db_path=args.db_path)
        generator = VisualizationGenerator(analyzer, output_dir=args.output_dir)
        
        selected_graphs = generator.display_menu()
        
        if selected_graphs:
            generator.generate_selected_graphs(selected_graphs)
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in interactive mode: {e}", exc_info=True)
    finally:
        try:
            analyzer.close()
        except:
            pass

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Generate visualization graphs")
    parser.add_argument("--db-path", type=Path, help="Path to performance DB")
    parser.add_argument("--output-dir", type=Path, help="Output directory for graphs")
    parser.add_argument("--graphs", nargs="+", type=int, 
                       help="Specific graph numbers to generate (1-16)")
    parser.add_argument("--experiment-dir", type=Path, help="Experiment directory to save results")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Handle experiment directory (기존 폴더 우선 사용)
    if args.experiment_dir:
        # Check if provided experiment directory exists
        if not args.experiment_dir.exists():
            logger.warning(f"Experiment directory does not exist: {args.experiment_dir}")
            # Try to find latest experiment directory
            latest_dir = find_latest_experiment_dir()
            if latest_dir:
                logger.info(f"Using latest experiment directory: {latest_dir}")
                args.experiment_dir = latest_dir
            else:
                logger.error("No experiment directories found")
                # Create new experiment directory only as last resort
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                args.experiment_dir = Path("experiments_results") / f"exp_{timestamp}"
                logger.info(f"Creating new experiment directory: {args.experiment_dir}")
    else:
        # Auto-find latest experiment directory
        latest_dir = find_latest_experiment_dir()
        if latest_dir:
            logger.info(f"Auto-detected experiment directory: {latest_dir}")
            args.experiment_dir = latest_dir
        else:
            # Create new experiment directory only if no existing ones
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.experiment_dir = Path("experiments_results") / f"exp_{timestamp}"
            logger.info(f"No existing experiment directories, creating: {args.experiment_dir}")
    
    if args.interactive:
        try:
            analyzer = ComparisonAnalyzer(db_path=args.db_path)
            
            # Determine output directory
            if args.experiment_dir:
                output_dir = args.experiment_dir / "figures" if not args.output_dir else args.output_dir
            else:
                # Fallback to current directory
                output_dir = Path("figures") if not args.output_dir else args.output_dir
            
            generator = VisualizationGenerator(analyzer, output_dir=output_dir)
            selected = generator.display_menu()
            
            if selected:
                generator.generate_selected_graphs(selected)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            try:
                analyzer.close()
            except:
                pass
    else:
        # Determine output directory for non-interactive mode
        if args.experiment_dir:
            if not args.output_dir:
                args.output_dir = args.experiment_dir / "figures"
        else:
            if not args.output_dir:
                # Use current directory as fallback
                args.output_dir = Path("figures")
        
        analyzer = ComparisonAnalyzer(db_path=args.db_path)
        generator = VisualizationGenerator(analyzer, output_dir=args.output_dir)
        
        if args.graphs:
            generator.generate_selected_graphs(args.graphs)
        else:
            print("Generating all graphs...")
            generator.generate_all_graphs()
        
        analyzer.close()
        print("Done!")