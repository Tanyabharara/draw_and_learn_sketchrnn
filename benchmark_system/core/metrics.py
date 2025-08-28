"""
Comprehensive Metrics and Evaluation System
Advanced metrics calculation, statistical analysis, and performance evaluation
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import BenchmarkMetrics, Prediction, ModelInfo

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

@dataclass
class StatisticalMetrics:
    """Statistical analysis of model performance"""
    confidence_interval_95: Tuple[float, float]
    confidence_mean: float
    confidence_std: float
    prediction_entropy: float
    calibration_error: float
    reliability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'confidence_interval_95': self.confidence_interval_95,
            'confidence_mean': self.confidence_mean,
            'confidence_std': self.confidence_std,
            'prediction_entropy': self.prediction_entropy,
            'calibration_error': self.calibration_error,
            'reliability_score': self.reliability_score
        }

@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics"""
    throughput_samples_per_sec: float
    latency_percentiles: Dict[str, float]  # P50, P90, P95, P99
    memory_efficiency: float  # Accuracy per MB
    scalability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'latency_percentiles': self.latency_percentiles,
            'memory_efficiency': self.memory_efficiency,
            'scalability_score': self.scalability_score
        }

@dataclass
class CategoryAnalysis:
    """Per-category detailed analysis"""
    category_name: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_details: Dict[str, int]
    avg_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category_name': self.category_name,
            'sample_count': self.sample_count,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_details': self.confusion_details,
            'avg_confidence': self.avg_confidence
        }

@dataclass
class ComprehensiveMetrics:
    """Complete metrics package"""
    basic_metrics: BenchmarkMetrics
    statistical_metrics: StatisticalMetrics
    performance_metrics: PerformanceMetrics
    category_analysis: List[CategoryAnalysis]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'basic_metrics': self.basic_metrics.to_dict(),
            'statistical_metrics': self.statistical_metrics.to_dict(),
            'performance_metrics': self.performance_metrics.to_dict(),
            'category_analysis': [ca.to_dict() for ca in self.category_analysis],
            'recommendations': self.recommendations
        }

class MetricsCalculator:
    """Advanced metrics calculator with comprehensive evaluation capabilities"""
    
    def __init__(self, categories: Optional[List[str]] = None):
        self.categories = categories or []
        self.sklearn_available = self._check_sklearn()
        
    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available"""
        try:
            import sklearn
            return True
        except ImportError:
            logger.warning("scikit-learn not available - some metrics will be limited")
            return False
    
    def calculate_comprehensive_metrics(self,
                                      predictions: List[Prediction],
                                      ground_truth: np.ndarray,
                                      model_info: ModelInfo,
                                      execution_times: Optional[List[float]] = None,
                                      memory_usage: Optional[float] = None) -> ComprehensiveMetrics:
        """Calculate comprehensive metrics from predictions and ground truth"""
        logger.info(f"Calculating comprehensive metrics for {len(predictions)} predictions")
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(predictions, ground_truth, model_info)
        
        # Statistical metrics
        statistical_metrics = self._calculate_statistical_metrics(predictions, ground_truth)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            predictions, execution_times, memory_usage, basic_metrics.accuracy
        )
        
        # Category analysis
        category_analysis = self._calculate_category_analysis(predictions, ground_truth)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            basic_metrics, statistical_metrics, performance_metrics, category_analysis
        )
        
        return ComprehensiveMetrics(
            basic_metrics=basic_metrics,
            statistical_metrics=statistical_metrics,
            performance_metrics=performance_metrics,
            category_analysis=category_analysis,
            recommendations=recommendations
        )
    
    def _calculate_basic_metrics(self,
                               predictions: List[Prediction],
                               ground_truth: np.ndarray,
                               model_info: ModelInfo) -> BenchmarkMetrics:
        """Calculate basic classification metrics"""
        # Extract predicted classes and timing
        predicted_classes = []
        inference_times = []
        
        for pred in predictions:
            if pred.predicted_classes and len(pred.predicted_classes) > 0:
                try:
                    predicted_classes.append(int(pred.predicted_classes[0]))
                except (ValueError, TypeError):
                    predicted_classes.append(0)
            else:
                predicted_classes.append(0)
            inference_times.append(pred.inference_time)
        
        predicted_classes = np.array(predicted_classes)
        
        # Align arrays
        min_length = min(len(predicted_classes), len(ground_truth))
        predicted_classes = predicted_classes[:min_length]
        ground_truth = ground_truth[:min_length]
        
        # Calculate metrics
        if self.sklearn_available:
            metrics = self._calculate_sklearn_metrics(predicted_classes, ground_truth)
        else:
            metrics = self._calculate_manual_metrics(predicted_classes, ground_truth)
        
        # Add timing information
        if inference_times:
            metrics.inference_time_mean = float(np.mean(inference_times))
            metrics.inference_time_std = float(np.std(inference_times))
            metrics.throughput = len(predictions) / sum(inference_times) if sum(inference_times) > 0 else 0.0
        
        metrics.model_name = model_info.name
        return metrics
    
    def _calculate_sklearn_metrics(self, predicted_classes: np.ndarray, ground_truth: np.ndarray) -> BenchmarkMetrics:
        """Calculate metrics using scikit-learn"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        
        accuracy = accuracy_score(ground_truth, predicted_classes)
        precision = precision_score(ground_truth, predicted_classes, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predicted_classes, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predicted_classes, average='weighted', zero_division=0)
        
        # Per-class metrics
        try:
            report = classification_report(ground_truth, predicted_classes, output_dict=True, zero_division=0)
            per_class_metrics = {
                str(k): v for k, v in report.items() 
                if isinstance(v, dict) and k not in ['accuracy', 'macro avg', 'weighted avg']
            }
        except Exception:
            per_class_metrics = None
        
        # Confusion matrix
        try:
            conf_matrix = confusion_matrix(ground_truth, predicted_classes)
        except Exception:
            conf_matrix = None
        
        return BenchmarkMetrics(
            model_name="",
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            inference_time_mean=0.0,
            inference_time_std=0.0,
            preprocessing_time_mean=0.0,
            preprocessing_time_std=0.0,
            throughput=0.0,
            per_class_metrics=per_class_metrics,
            confusion_matrix=conf_matrix
        )
    
    def _calculate_manual_metrics(self, predicted_classes: np.ndarray, ground_truth: np.ndarray) -> BenchmarkMetrics:
        """Calculate metrics manually without scikit-learn"""
        correct = (predicted_classes == ground_truth).sum()
        accuracy = correct / len(ground_truth) if len(ground_truth) > 0 else 0.0
        
        return BenchmarkMetrics(
            model_name="",
            accuracy=float(accuracy),
            precision=float(accuracy),  # Simplified
            recall=float(accuracy),     # Simplified
            f1_score=float(accuracy),   # Simplified
            inference_time_mean=0.0,
            inference_time_std=0.0,
            preprocessing_time_mean=0.0,
            preprocessing_time_std=0.0,
            throughput=0.0
        )
    
    def _calculate_statistical_metrics(self,
                                     predictions: List[Prediction],
                                     ground_truth: np.ndarray) -> StatisticalMetrics:
        """Calculate advanced statistical metrics"""
        # Extract confidence scores
        confidence_scores = []
        correct_predictions = []
        
        for i, pred in enumerate(predictions):
            if i >= len(ground_truth):
                break
                
            if pred.confidence_scores is not None and len(pred.confidence_scores) > 0:
                confidence_scores.append(float(pred.confidence_scores[0]))
            else:
                confidence_scores.append(0.0)
            
            # Check if prediction is correct
            if pred.predicted_classes and len(pred.predicted_classes) > 0:
                try:
                    predicted_class = int(pred.predicted_classes[0])
                    correct_predictions.append(predicted_class == ground_truth[i])
                except (ValueError, TypeError):
                    correct_predictions.append(False)
            else:
                correct_predictions.append(False)
        
        confidence_scores = np.array(confidence_scores)
        correct_predictions = np.array(correct_predictions)
        
        # Calculate statistics
        confidence_mean = float(np.mean(confidence_scores)) if len(confidence_scores) > 0 else 0.0
        confidence_std = float(np.std(confidence_scores)) if len(confidence_scores) > 0 else 0.0
        
        # Simple confidence interval
        margin = 1.96 * confidence_std / np.sqrt(len(confidence_scores)) if len(confidence_scores) > 0 else 0.0
        confidence_interval = (confidence_mean - margin, confidence_mean + margin)
        
        # Calculate entropy and calibration
        entropy = 0.0  # Simplified
        calibration_error = self._calculate_calibration_error(confidence_scores, correct_predictions)
        reliability_score = self._calculate_reliability_score(confidence_scores, correct_predictions)
        
        return StatisticalMetrics(
            confidence_interval_95=confidence_interval,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            prediction_entropy=entropy,
            calibration_error=calibration_error,
            reliability_score=reliability_score
        )
    
    def _calculate_calibration_error(self, confidence_scores: np.ndarray, correct_predictions: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        if len(confidence_scores) == 0:
            return 0.0
        
        try:
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            
            for i in range(n_bins):
                bin_mask = (confidence_scores > bin_boundaries[i]) & (confidence_scores <= bin_boundaries[i + 1])
                if bin_mask.sum() > 0:
                    bin_acc = correct_predictions[bin_mask].mean()
                    bin_conf = confidence_scores[bin_mask].mean()
                    bin_size = bin_mask.sum() / len(confidence_scores)
                    ece += np.abs(bin_acc - bin_conf) * bin_size
            
            return float(ece)
        except Exception:
            return 0.0
    
    def _calculate_reliability_score(self, confidence_scores: np.ndarray, correct_predictions: np.ndarray) -> float:
        """Calculate reliability score"""
        if len(confidence_scores) == 0:
            return 0.0
        
        try:
            high_conf_mask = confidence_scores > 0.8
            if high_conf_mask.sum() == 0:
                return 0.5
            
            high_conf_accuracy = correct_predictions[high_conf_mask].mean()
            overall_accuracy = correct_predictions.mean()
            
            reliability = high_conf_accuracy / (overall_accuracy + 1e-10)
            return min(float(reliability), 2.0)
        except Exception:
            return 0.0
    
    def _calculate_performance_metrics(self,
                                     predictions: List[Prediction],
                                     execution_times: Optional[List[float]],
                                     memory_usage: Optional[float],
                                     accuracy: float) -> PerformanceMetrics:
        """Calculate performance metrics"""
        inference_times = [pred.inference_time for pred in predictions if pred.inference_time > 0]
        
        if not inference_times:
            inference_times = execution_times or [0.0]
        
        # Throughput
        total_time = sum(inference_times)
        throughput = len(predictions) / total_time if total_time > 0 else 0.0
        
        # Latency percentiles
        latency_percentiles = {'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0}
        if inference_times:
            try:
                latency_percentiles = {
                    'p50': float(np.percentile(inference_times, 50)),
                    'p90': float(np.percentile(inference_times, 90)),
                    'p95': float(np.percentile(inference_times, 95)),
                    'p99': float(np.percentile(inference_times, 99))
                }
            except Exception:
                pass
        
        # Memory efficiency
        memory_efficiency = 0.0
        if memory_usage and memory_usage > 0:
            memory_efficiency = accuracy / memory_usage
        
        # Scalability score
        scalability_score = 0.0
        if len(inference_times) > 1:
            latency_cv = np.std(inference_times) / (np.mean(inference_times) + 1e-10)
            scalability_score = 1.0 / (1.0 + latency_cv)
        
        return PerformanceMetrics(
            throughput_samples_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            memory_efficiency=memory_efficiency,
            scalability_score=float(scalability_score)
        )
    
    def _calculate_category_analysis(self,
                                   predictions: List[Prediction],
                                   ground_truth: np.ndarray) -> List[CategoryAnalysis]:
        """Calculate per-category analysis"""
        if not self.categories:
            return []
        
        category_analyses = []
        unique_classes = np.unique(ground_truth)
        
        for class_idx in unique_classes:
            if class_idx >= len(self.categories):
                continue
                
            category_name = self.categories[class_idx]
            class_mask = ground_truth == class_idx
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Get predictions for this category
            class_predictions = [predictions[i] for i in class_indices if i < len(predictions)]
            
            # Calculate metrics
            predicted_classes = []
            confidence_scores = []
            
            for pred in class_predictions:
                if pred.predicted_classes and len(pred.predicted_classes) > 0:
                    try:
                        predicted_classes.append(int(pred.predicted_classes[0]))
                    except (ValueError, TypeError):
                        predicted_classes.append(-1)
                else:
                    predicted_classes.append(-1)
                
                if pred.confidence_scores is not None and len(pred.confidence_scores) > 0:
                    confidence_scores.append(float(pred.confidence_scores[0]))
                else:
                    confidence_scores.append(0.0)
            
            predicted_classes = np.array(predicted_classes)
            
            # Calculate category metrics
            correct_predictions = predicted_classes == class_idx
            accuracy = correct_predictions.mean() if len(correct_predictions) > 0 else 0.0
            
            tp = correct_predictions.sum()
            fp = (predicted_classes == class_idx).sum() - tp
            fn = len(class_indices) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Confusion analysis
            confusion_details = {}
            incorrect_predictions = predicted_classes[~correct_predictions]
            for wrong_class in np.unique(incorrect_predictions):
                if 0 <= wrong_class < len(self.categories):
                    count = (incorrect_predictions == wrong_class).sum()
                    confusion_details[self.categories[wrong_class]] = int(count)
            
            avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
            
            analysis = CategoryAnalysis(
                category_name=category_name,
                sample_count=len(class_predictions),
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1_score),
                confusion_details=confusion_details,
                avg_confidence=avg_confidence
            )
            category_analyses.append(analysis)
        
        return category_analyses
    
    def _generate_recommendations(self,
                                basic_metrics: BenchmarkMetrics,
                                statistical_metrics: StatisticalMetrics,
                                performance_metrics: PerformanceMetrics,
                                category_analysis: List[CategoryAnalysis]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Accuracy recommendations
        if basic_metrics.accuracy < 0.7:
            recommendations.append("Model accuracy is low (<70%). Consider retraining with more data or adjusting architecture.")
        elif basic_metrics.accuracy < 0.85:
            recommendations.append("Model accuracy is moderate. Consider hyperparameter tuning or data augmentation.")
        
        # Confidence calibration
        if statistical_metrics.calibration_error > 0.1:
            recommendations.append("Model confidence is poorly calibrated. Consider temperature scaling.")
        
        # Performance recommendations
        if performance_metrics.throughput_samples_per_sec < 10:
            recommendations.append("Low throughput detected. Consider model optimization or hardware acceleration.")
        
        # Memory efficiency
        if performance_metrics.memory_efficiency < 0.1:
            recommendations.append("Low memory efficiency. Consider model pruning or quantization.")
        
        # Category-specific recommendations
        low_performance_categories = [ca for ca in category_analysis if ca.accuracy < 0.5]
        if low_performance_categories:
            category_names = [ca.category_name for ca in low_performance_categories]
            recommendations.append(f"Poor performance on categories: {', '.join(category_names)}. Consider class-specific data augmentation.")
        
        return recommendations

# Convenience functions
def calculate_metrics(predictions: List[Prediction], 
                     ground_truth: np.ndarray, 
                     model_info: ModelInfo,
                     categories: Optional[List[str]] = None,
                     **kwargs) -> ComprehensiveMetrics:
    """Calculate comprehensive metrics with simple interface"""
    calculator = MetricsCalculator(categories)
    return calculator.calculate_comprehensive_metrics(predictions, ground_truth, model_info, **kwargs)