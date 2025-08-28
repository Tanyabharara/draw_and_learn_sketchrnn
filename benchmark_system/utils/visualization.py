"""
Visualization and Reporting System
Generate comprehensive charts, graphs, and reports from benchmark results
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.benchmark_engine import BenchmarkSummary, ModelBenchmarkResult
from core.metrics import ComprehensiveMetrics

# Configure logging
logger = logging.getLogger(__name__)

class BenchmarkVisualizer:
    """
    Comprehensive visualization system for benchmark results
    """
    
    def __init__(self, output_dir: Union[str, Path] = "benchmark_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for plotting libraries
        self.matplotlib_available = self._check_matplotlib()
        self.seaborn_available = self._check_seaborn()
        
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            logger.warning("matplotlib not available - visual plots will be skipped")
            return False
    
    def _check_seaborn(self) -> bool:
        """Check if seaborn is available"""
        try:
            import seaborn as sns
            return True
        except ImportError:
            logger.info("seaborn not available - using basic matplotlib styling")
            return False
    
    def generate_comprehensive_report(self, 
                                    summary: BenchmarkSummary,
                                    detailed_metrics: Optional[Dict[str, ComprehensiveMetrics]] = None) -> Path:
        """
        Generate a comprehensive benchmark report with visualizations
        
        Args:
            summary: Benchmark summary with results
            detailed_metrics: Optional detailed metrics for advanced analysis
            
        Returns:
            Path to generated report directory
        """
        logger.info("Generating comprehensive benchmark report")
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"benchmark_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate different types of visualizations
        try:
            if self.matplotlib_available:
                self._generate_performance_charts(summary, report_dir)
                self._generate_accuracy_comparison(summary, report_dir)
                self._generate_timing_analysis(summary, report_dir)
                
                if detailed_metrics:
                    self._generate_detailed_analysis_charts(detailed_metrics, report_dir)
            
            # Generate text reports (always available)
            self._generate_text_report(summary, report_dir, detailed_metrics)
            self._generate_json_report(summary, report_dir, detailed_metrics)
            self._generate_csv_summary(summary, report_dir)
            
            logger.info(f"Report generated successfully in: {report_dir}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
        
        return report_dir
    
    def _generate_performance_charts(self, summary: BenchmarkSummary, output_dir: Path):
        """Generate performance comparison charts"""
        try:
            import matplotlib.pyplot as plt
            if self.seaborn_available:
                import seaborn as sns
                sns.set_style("whitegrid")
            
            successful_results = [r for r in summary.results if r.error is None and r.metrics is not None]
            
            if len(successful_results) < 2:
                logger.warning("Need at least 2 successful results for performance comparison")
                return
            
            # Extract data
            model_names = [r.model_name for r in successful_results]
            accuracies = [r.metrics.accuracy for r in successful_results]
            inference_times = [r.metrics.inference_time_mean for r in successful_results]
            throughputs = [r.metrics.throughput for r in successful_results]
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # Accuracy comparison
            axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Inference time comparison
            axes[0, 1].bar(model_names, inference_times, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Inference Time Comparison')
            axes[0, 1].set_ylabel('Mean Inference Time (s)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Throughput comparison
            axes[1, 0].bar(model_names, throughputs, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Throughput Comparison')
            axes[1, 0].set_ylabel('Samples per Second')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Accuracy vs Speed scatter
            axes[1, 1].scatter(inference_times, accuracies, c='purple', alpha=0.7, s=100)
            for i, name in enumerate(model_names):
                axes[1, 1].annotate(name, (inference_times[i], accuracies[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_title('Accuracy vs Speed Trade-off')
            axes[1, 1].set_xlabel('Mean Inference Time (s)')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance charts generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate performance charts: {str(e)}")
    
    def _generate_accuracy_comparison(self, summary: BenchmarkSummary, output_dir: Path):
        """Generate detailed accuracy analysis"""
        try:
            import matplotlib.pyplot as plt
            
            successful_results = [r for r in summary.results if r.error is None and r.metrics is not None]
            
            if not successful_results:
                return
            
            # Create accuracy breakdown chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Model Accuracy Analysis', fontsize=16, fontweight='bold')
            
            model_names = [r.model_name for r in successful_results]
            accuracies = [r.metrics.accuracy for r in successful_results]
            precisions = [r.metrics.precision for r in successful_results]
            recalls = [r.metrics.recall for r in successful_results]
            f1_scores = [r.metrics.f1_score for r in successful_results]
            
            # Grouped bar chart for all metrics
            x = np.arange(len(model_names))
            width = 0.2
            
            ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
            ax1.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
            ax1.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
            ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            ax1.set_title('Classification Metrics Comparison')
            ax1.set_ylabel('Score')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            # Model ranking
            ranking_data = list(zip(model_names, accuracies))
            ranking_data.sort(key=lambda x: x[1], reverse=True)
            
            ranked_names = [item[0] for item in ranking_data]
            ranked_scores = [item[1] for item in ranking_data]
            
            colors = plt.cm.RdYlGn([score for score in ranked_scores])
            bars = ax2.barh(ranked_names, ranked_scores, color=colors, alpha=0.8)
            ax2.set_title('Model Accuracy Ranking')
            ax2.set_xlabel('Accuracy')
            ax2.grid(True, alpha=0.3)
            
            # Add accuracy values on bars
            for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
                ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'accuracy_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Accuracy analysis charts generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate accuracy charts: {str(e)}")
    
    def _generate_timing_analysis(self, summary: BenchmarkSummary, output_dir: Path):
        """Generate timing and performance analysis"""
        try:
            import matplotlib.pyplot as plt
            
            successful_results = [r for r in summary.results if r.error is None and r.metrics is not None]
            
            if not successful_results:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Timing and Performance Analysis', fontsize=16, fontweight='bold')
            
            model_names = [r.model_name for r in successful_results]
            inference_means = [r.metrics.inference_time_mean for r in successful_results]
            inference_stds = [r.metrics.inference_time_std for r in successful_results]
            throughputs = [r.metrics.throughput for r in successful_results]
            execution_times = [r.execution_time for r in successful_results]
            
            # Inference time with error bars
            axes[0, 0].bar(model_names, inference_means, yerr=inference_stds, 
                          capsize=5, alpha=0.7, color='lightblue')
            axes[0, 0].set_title('Inference Time (Mean ± Std)')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Throughput comparison
            axes[0, 1].bar(model_names, throughputs, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Model Throughput')
            axes[0, 1].set_ylabel('Samples per Second')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Total execution time
            axes[1, 0].bar(model_names, execution_times, alpha=0.7, color='orange')
            axes[1, 0].set_title('Total Execution Time')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Performance efficiency (accuracy/time)
            efficiency = [acc/time if time > 0 else 0 for acc, time in 
                         zip([r.metrics.accuracy for r in successful_results], execution_times)]
            axes[1, 1].bar(model_names, efficiency, alpha=0.7, color='purple')
            axes[1, 1].set_title('Performance Efficiency (Accuracy/Time)')
            axes[1, 1].set_ylabel('Efficiency Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'timing_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Timing analysis charts generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate timing charts: {str(e)}")
    
    def _generate_detailed_analysis_charts(self, 
                                         detailed_metrics: Dict[str, ComprehensiveMetrics], 
                                         output_dir: Path):
        """Generate charts from detailed metrics analysis"""
        try:
            import matplotlib.pyplot as plt
            
            if not detailed_metrics:
                return
            
            # Confidence analysis
            self._plot_confidence_analysis(detailed_metrics, output_dir)
            
            # Category performance analysis
            self._plot_category_analysis(detailed_metrics, output_dir)
            
            logger.info("Detailed analysis charts generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate detailed analysis charts: {str(e)}")
    
    def _plot_confidence_analysis(self, detailed_metrics: Dict[str, ComprehensiveMetrics], output_dir: Path):
        """Plot confidence and calibration analysis"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Model Confidence Analysis', fontsize=16, fontweight='bold')
            
            model_names = list(detailed_metrics.keys())
            confidence_means = [m.statistical_metrics.confidence_mean for m in detailed_metrics.values()]
            calibration_errors = [m.statistical_metrics.calibration_error for m in detailed_metrics.values()]
            reliability_scores = [m.statistical_metrics.reliability_score for m in detailed_metrics.values()]
            
            # Confidence levels
            axes[0].bar(model_names, confidence_means, alpha=0.7, color='skyblue')
            axes[0].set_title('Average Confidence Scores')
            axes[0].set_ylabel('Confidence')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Calibration error
            axes[1].bar(model_names, calibration_errors, alpha=0.7, color='lightcoral')
            axes[1].set_title('Calibration Error (Lower is Better)')
            axes[1].set_ylabel('Expected Calibration Error')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            # Reliability scores
            axes[2].bar(model_names, reliability_scores, alpha=0.7, color='lightgreen')
            axes[2].set_title('Reliability Scores (Higher is Better)')
            axes[2].set_ylabel('Reliability Score')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot confidence analysis: {str(e)}")
    
    def _plot_category_analysis(self, detailed_metrics: Dict[str, ComprehensiveMetrics], output_dir: Path):
        """Plot per-category performance analysis"""
        try:
            import matplotlib.pyplot as plt
            
            # Collect category data from all models
            all_categories = set()
            for metrics in detailed_metrics.values():
                for cat_analysis in metrics.category_analysis:
                    all_categories.add(cat_analysis.category_name)
            
            if not all_categories:
                logger.info("No category analysis data available")
                return
            
            all_categories = sorted(list(all_categories))
            model_names = list(detailed_metrics.keys())
            
            # Create category performance matrix
            category_accuracies = {}
            for model_name, metrics in detailed_metrics.items():
                category_accuracies[model_name] = {}
                for cat_analysis in metrics.category_analysis:
                    category_accuracies[model_name][cat_analysis.category_name] = cat_analysis.accuracy
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data matrix
            data_matrix = []
            for model_name in model_names:
                row = []
                for category in all_categories:
                    accuracy = category_accuracies[model_name].get(category, 0.0)
                    row.append(accuracy)
                data_matrix.append(row)
            
            data_matrix = np.array(data_matrix)
            
            # Create heatmap
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(all_categories)))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.set_yticklabels(model_names)
            
            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(all_categories)):
                    text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title('Per-Category Model Performance Heatmap')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Accuracy', rotation=270, labelpad=15)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'category_performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot category analysis: {str(e)}")
    
    def _generate_text_report(self, 
                            summary: BenchmarkSummary, 
                            output_dir: Path,
                            detailed_metrics: Optional[Dict[str, ComprehensiveMetrics]] = None):
        """Generate comprehensive text report"""
        try:
            report_path = output_dir / 'benchmark_report.txt'
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("COMPREHENSIVE BENCHMARK REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Models Evaluated: {summary.total_models}\n")
                f.write(f"Successful Evaluations: {summary.successful_models}\n")
                f.write(f"Failed Evaluations: {summary.failed_models}\n")
                f.write(f"Total Execution Time: {summary.total_execution_time:.2f} seconds\n")
                f.write(f"Average Time per Model: {summary.total_execution_time/summary.total_models:.2f} seconds\n")
                f.write(f"Benchmark Date: {summary.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Hardware Information
                f.write("HARDWARE CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                hw_info = summary.hardware_info
                f.write(f"Environment: {hw_info.get('environment', 'Unknown')}\n")
                primary_device = hw_info.get('primary_device', {})
                f.write(f"Primary Device: {primary_device.get('name', 'Unknown')}\n")
                f.write(f"Device Type: {primary_device.get('type', 'Unknown')}\n")
                f.write(f"Memory: {primary_device.get('memory_gb', 'Unknown')} GB\n\n")
                
                # Model Results
                f.write("INDIVIDUAL MODEL RESULTS\n")
                f.write("-" * 40 + "\n")
                
                successful_results = [r for r in summary.results if r.error is None]
                failed_results = [r for r in summary.results if r.error is not None]
                
                # Sort by accuracy
                successful_results.sort(key=lambda x: x.metrics.accuracy if x.metrics else 0, reverse=True)
                
                for i, result in enumerate(successful_results, 1):
                    f.write(f"\n{i}. Model: {result.model_name}\n")
                    f.write(f"   ID: {result.model_id}\n")
                    if result.metrics:
                        f.write(f"   Accuracy: {result.metrics.accuracy:.4f}\n")
                        f.write(f"   Precision: {result.metrics.precision:.4f}\n")
                        f.write(f"   Recall: {result.metrics.recall:.4f}\n")
                        f.write(f"   F1-Score: {result.metrics.f1_score:.4f}\n")
                        f.write(f"   Inference Time: {result.metrics.inference_time_mean:.4f} ± {result.metrics.inference_time_std:.4f} s\n")
                        f.write(f"   Throughput: {result.metrics.throughput:.2f} samples/sec\n")
                    f.write(f"   Total Execution Time: {result.execution_time:.2f} s\n")
                    if result.memory_peak_mb:
                        f.write(f"   Peak Memory Usage: {result.memory_peak_mb:.1f} MB\n")
                
                # Failed models
                if failed_results:
                    f.write("\nFAILED MODELS\n")
                    f.write("-" * 20 + "\n")
                    for result in failed_results:
                        f.write(f"Model: {result.model_name}\n")
                        f.write(f"Error: {result.error}\n\n")
                
                # Performance Rankings
                f.write("\nPERFORMANCE RANKINGS\n")
                f.write("-" * 40 + "\n")
                
                if successful_results:
                    f.write("Top 3 by Accuracy:\n")
                    for i, result in enumerate(successful_results[:3], 1):
                        f.write(f"  {i}. {result.model_name}: {result.metrics.accuracy:.4f}\n")
                    
                    f.write("\nTop 3 by Speed:\n")
                    speed_sorted = sorted(successful_results, key=lambda x: x.metrics.inference_time_mean)
                    for i, result in enumerate(speed_sorted[:3], 1):
                        f.write(f"  {i}. {result.model_name}: {result.metrics.inference_time_mean:.4f} s\n")
                    
                    f.write("\nTop 3 by Throughput:\n")
                    throughput_sorted = sorted(successful_results, key=lambda x: x.metrics.throughput, reverse=True)
                    for i, result in enumerate(throughput_sorted[:3], 1):
                        f.write(f"  {i}. {result.model_name}: {result.metrics.throughput:.2f} samples/sec\n")
                
                # Detailed metrics section
                if detailed_metrics:
                    f.write("\nDETAILED ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    
                    for model_name, metrics in detailed_metrics.items():
                        f.write(f"\nModel: {model_name}\n")
                        f.write(f"Confidence Analysis:\n")
                        f.write(f"  Mean Confidence: {metrics.statistical_metrics.confidence_mean:.4f}\n")
                        f.write(f"  Confidence Std: {metrics.statistical_metrics.confidence_std:.4f}\n")
                        f.write(f"  Calibration Error: {metrics.statistical_metrics.calibration_error:.4f}\n")
                        f.write(f"  Reliability Score: {metrics.statistical_metrics.reliability_score:.4f}\n")
                        
                        if metrics.recommendations:
                            f.write(f"Recommendations:\n")
                            for rec in metrics.recommendations:
                                f.write(f"  • {rec}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Text report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate text report: {str(e)}")
    
    def _generate_json_report(self, 
                            summary: BenchmarkSummary, 
                            output_dir: Path,
                            detailed_metrics: Optional[Dict[str, ComprehensiveMetrics]] = None):
        """Generate JSON report for programmatic access"""
        try:
            report_data = summary.to_dict()
            
            if detailed_metrics:
                report_data['detailed_metrics'] = {
                    model_name: metrics.to_dict() 
                    for model_name, metrics in detailed_metrics.items()
                }
            
            report_path = output_dir / 'benchmark_results.json'
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"JSON report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {str(e)}")
    
    def _generate_csv_summary(self, summary: BenchmarkSummary, output_dir: Path):
        """Generate CSV summary for spreadsheet analysis"""
        try:
            import csv
            
            successful_results = [r for r in summary.results if r.error is None and r.metrics is not None]
            
            csv_path = output_dir / 'benchmark_summary.csv'
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = [
                    'model_name', 'model_id', 'accuracy', 'precision', 'recall', 'f1_score',
                    'inference_time_mean', 'inference_time_std', 'throughput',
                    'execution_time', 'memory_peak_mb'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in successful_results:
                    writer.writerow({
                        'model_name': result.model_name,
                        'model_id': result.model_id,
                        'accuracy': result.metrics.accuracy,
                        'precision': result.metrics.precision,
                        'recall': result.metrics.recall,
                        'f1_score': result.metrics.f1_score,
                        'inference_time_mean': result.metrics.inference_time_mean,
                        'inference_time_std': result.metrics.inference_time_std,
                        'throughput': result.metrics.throughput,
                        'execution_time': result.execution_time,
                        'memory_peak_mb': result.memory_peak_mb or 0
                    })
            
            logger.info(f"CSV summary generated: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate CSV summary: {str(e)}")

# Convenience functions
def create_visualizer(output_dir: str = "benchmark_reports") -> BenchmarkVisualizer:
    """Create a benchmark visualizer"""
    return BenchmarkVisualizer(output_dir)

def generate_quick_report(summary: BenchmarkSummary, 
                         output_dir: str = "benchmark_reports") -> Path:
    """Generate a quick benchmark report"""
    visualizer = create_visualizer(output_dir)
    return visualizer.generate_comprehensive_report(summary)