"""
Advanced Visualizations and Utility Functions for Fairness Metrics SDK
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from fairness_metrics import AdjustedIntersectionalNetBenefit


class FairnessVisualizer:
    """Advanced visualization tools for fairness analysis"""
    
    def __init__(self, ablni_metric: AdjustedIntersectionalNetBenefit):
        """
        Initialize visualizer with fitted ABLNI metric.
        
        Parameters
        ----------
        ablni_metric : AdjustedIntersectionalNetBenefit
            Fitted ABLNI metric instance.
        """
        if ablni_metric.subgroup_results_ is None:
            raise ValueError("ABLNI metric must be fitted before visualization")
        
        self.ablni = ablni_metric
        self.results = ablni_metric.subgroup_results_
    
    def plot_comprehensive_dashboard(self, figsize=(16, 12), save_path=None):
        """
        Create comprehensive fairness dashboard with multiple plots.
        
        Parameters
        ----------
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save figure.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Net Benefit Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_net_benefit_bars(ax1)
        
        # 2. ABLNI Score Gauge
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_ablni_gauge(ax2)
        
        # 3. Performance Metrics Heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_performance_heatmap(ax3)
        
        # 4. Sample Size Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_sample_sizes(ax4)
        
        # 5. Confusion Matrix Comparison
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_confusion_matrices(ax5)
        
        plt.suptitle('Fairness Analysis Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_net_benefit_bars(self, ax):
        """Plot net benefit bars with error indication"""
        df = self.results.sort_values('net_benefit')
        
        colors = ['#d62728' if nb == df['net_benefit'].min() 
                 else '#2ca02c' if nb == df['net_benefit'].max()
                 else '#1f77b4' for nb in df['net_benefit']]
        
        bars = ax.barh(range(len(df)), df['net_benefit'], color=colors, alpha=0.7)
        
        # Add median line
        median = df['net_benefit'].median()
        ax.axvline(median, color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {median:.3f}')
        
        # Add zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['subgroup'])
        ax.set_xlabel('Net Benefit')
        ax.set_title('Net Benefit by Subgroup')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_ablni_gauge(self, ax):
        """Plot ABLNI score as gauge chart"""
        score = self.ablni.overall_score_
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        
        # Color zones
        zones = [
            (0, 0.6, '#d62728', 'Critical'),
            (0.6, 0.8, '#ff7f0e', 'Poor'),
            (0.8, 0.9, '#ffbb78', 'Fair'),
            (0.9, 1.0, '#2ca02c', 'Good')
        ]
        
        for start, end, color, label in zones:
            theta_zone = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 50)
            ax.fill_between(np.cos(theta_zone), 0, np.sin(theta_zone), 
                           color=color, alpha=0.3)
        
        # Needle
        if not np.isnan(score):
            angle = np.pi * (1 - score)
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 
                   'k-', linewidth=3, marker='o', markersize=10)
        
        ax.text(0, -0.3, f'ABLNI\n{score:.3f}', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.2)
        ax.axis('off')
        ax.set_title('Overall Fairness Score')
    
    def _plot_performance_heatmap(self, ax):
        """Plot heatmap of performance metrics"""
        metrics = ['tpr', 'tnr', 'ppv', 'npv']
        metric_labels = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        
        data = self.results[metrics].values.T
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=self.results['subgroup'],
                   yticklabels=metric_labels,
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        
        ax.set_title('Performance Metrics Across Subgroups')
        ax.set_xlabel('Subgroup')
    
    def _plot_sample_sizes(self, ax):
        """Plot sample size distribution"""
        df = self.results.sort_values('n', ascending=False)
        
        colors = ['#ff7f0e' if n < 100 else '#1f77b4' for n in df['n']]
        
        ax.bar(range(len(df)), df['n'], color=colors, alpha=0.7)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['subgroup'], rotation=45, ha='right')
        ax.set_ylabel('Sample Size')
        ax.set_title('Sample Size Distribution')
        
        # Add minimum size line
        if hasattr(self.ablni, 'min_subgroup_size'):
            ax.axhline(self.ablni.min_subgroup_size, color='red', 
                      linestyle='--', label='Minimum Size')
            ax.legend()
        
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_confusion_matrices(self, ax):
        """Plot confusion matrices for each subgroup"""
        n_subgroups = len(self.results)
        
        # Create mini confusion matrices
        for i, (_, row) in enumerate(self.results.iterrows()):
            # Calculate position
            x_offset = i * 1.2
            
            # Confusion matrix values
            cm = np.array([[row['tn'], row['fp']], 
                          [row['fn'], row['tp']]])
            
            # Normalize
            cm_norm = cm / cm.sum()
            
            # Plot mini heatmap
            im = ax.imshow(cm_norm, cmap='Blues', aspect='auto',
                          extent=[x_offset, x_offset + 1, 0, 2],
                          vmin=0, vmax=1)
            
            # Add text
            for j in range(2):
                for k in range(2):
                    text = ax.text(x_offset + 0.5, 1.5 - j, 
                                 f'{int(cm[j, k])}',
                                 ha='center', va='center', 
                                 color='white' if cm_norm[j, k] > 0.5 else 'black',
                                 fontsize=8)
            
            # Add subgroup label
            ax.text(x_offset + 0.5, -0.3, row['subgroup'], 
                   ha='center', va='top', rotation=45, fontsize=8)
        
        ax.set_xlim(-0.2, n_subgroups * 1.2)
        ax.set_ylim(-0.5, 2.2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Positive', 'Negative'])
        ax.set_xticks([])
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrices by Subgroup')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.1, aspect=30)
        cbar.set_label('Normalized Count')
    
    def plot_calibration_curves(self, y_true, y_pred_proba, sensitive_attrs,
                                n_bins=10, figsize=(12, 8), save_path=None):
        """
        Plot calibration curves for each subgroup.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred_proba : array-like
            Predicted probabilities.
        sensitive_attrs : DataFrame, Series, or array
            Sensitive attributes.
        n_bins : int
            Number of bins for calibration.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create subgroups
        if isinstance(sensitive_attrs, pd.DataFrame):
            subgroups = sensitive_attrs.astype(str).agg('_'.join, axis=1)
        else:
            subgroups = pd.Series(sensitive_attrs).astype(str)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(subgroups.unique())))
        
        # Plot calibration curves
        for i, subgroup in enumerate(subgroups.unique()):
            mask = subgroups == subgroup
            y_true_sub = np.array(y_true)[mask]
            y_pred_sub = np.array(y_pred_proba)[mask]
            
            # Calculate calibration
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            observed_freq = []
            predicted_freq = []
            counts = []
            
            for j in range(n_bins):
                bin_mask = (y_pred_sub >= bins[j]) & (y_pred_sub < bins[j+1])
                if j == n_bins - 1:  # Include upper bound in last bin
                    bin_mask = (y_pred_sub >= bins[j]) & (y_pred_sub <= bins[j+1])
                
                if bin_mask.sum() > 0:
                    observed_freq.append(y_true_sub[bin_mask].mean())
                    predicted_freq.append(y_pred_sub[bin_mask].mean())
                    counts.append(bin_mask.sum())
                else:
                    observed_freq.append(np.nan)
                    predicted_freq.append(np.nan)
                    counts.append(0)
            
            # Plot calibration curve
            valid = ~np.isnan(observed_freq)
            ax1.plot(np.array(predicted_freq)[valid], np.array(observed_freq)[valid], 
                    'o-', label=subgroup, color=colors[i], linewidth=2, markersize=8)
            
            # Plot distribution
            ax2.hist(y_pred_sub, bins=bins, alpha=0.5, label=subgroup, 
                    color=colors[i], edgecolor='black')
        
        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Observed Frequency')
        ax1.set_title('Calibration Curves by Subgroup')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution by Subgroup')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_decision_curves(self, y_true, y_pred_proba, sensitive_attrs,
                           threshold_range=(0.0, 1.0), n_thresholds=100,
                           figsize=(14, 6), save_path=None):
        """
        Plot decision curves (net benefit across thresholds) for each subgroup.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred_proba : array-like
            Predicted probabilities.
        sensitive_attrs : DataFrame, Series, or array
            Sensitive attributes.
        threshold_range : tuple
            Range of thresholds to evaluate.
        n_thresholds : int
            Number of thresholds to evaluate.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create subgroups
        if isinstance(sensitive_attrs, pd.DataFrame):
            subgroups = sensitive_attrs.astype(str).agg('_'.join, axis=1)
        else:
            subgroups = pd.Series(sensitive_attrs).astype(str)
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        colors = plt.cm.tab10(np.linspace(0, 1, len(subgroups.unique())))
        
        # Calculate net benefit for each subgroup and threshold
        for i, subgroup in enumerate(subgroups.unique()):
            mask = subgroups == subgroup
            y_true_sub = np.array(y_true)[mask]
            y_pred_sub = np.array(y_pred_proba)[mask]
            
            net_benefits = []
            prevalences = []
            
            for threshold in thresholds:
                if threshold == 0:
                    threshold = 0.001  # Avoid division by zero
                
                y_pred_binary = (y_pred_sub >= threshold).astype(int)
                
                tp = np.sum((y_pred_binary == 1) & (y_true_sub == 1))
                fp = np.sum((y_pred_binary == 1) & (y_true_sub == 0))
                n = len(y_true_sub)
                
                weight = (1 - threshold) / threshold
                nb = (tp / n) - (fp / n) * weight
                
                net_benefits.append(nb)
                prevalences.append(y_true_sub.mean())
            
            # Plot net benefit curve
            ax1.plot(thresholds, net_benefits, label=subgroup, 
                    color=colors[i], linewidth=2)
            
            # Plot difference from median
            median_nb = np.median(net_benefits)
            differences = np.array(net_benefits) - median_nb
            ax2.plot(thresholds, differences, label=subgroup,
                    color=colors[i], linewidth=2)
        
        # Add reference lines
        # Treat all
        treat_all_nb = [np.mean(y_true) - (1-t)/t * (1-np.mean(y_true)) 
                       for t in thresholds]
        ax1.plot(thresholds, treat_all_nb, 'k--', linewidth=1.5, 
                label='Treat All', alpha=0.5)
        
        # Treat none
        ax1.axhline(0, color='gray', linestyle=':', linewidth=1.5, 
                   label='Treat None', alpha=0.5)
        
        ax1.set_xlabel('Risk Threshold')
        ax1.set_ylabel('Net Benefit')
        ax1.set_title('Decision Curves by Subgroup')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('Risk Threshold')
        ax2.set_ylabel('Net Benefit - Median')
        ax2.set_title('Net Benefit Deviation from Median')
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class FairnessComparator:
    """Compare fairness across multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, y_true, y_pred_proba, sensitive_attrs, 
                  ablni_kwargs=None):
        """
        Add a model for comparison.
        
        Parameters
        ----------
        name : str
            Model name.
        y_true : array-like
            True labels.
        y_pred_proba : array-like
            Predicted probabilities.
        sensitive_attrs : DataFrame, Series, or array
            Sensitive attributes.
        ablni_kwargs : dict, optional
            Additional arguments for ABLNI metric.
        """
        if ablni_kwargs is None:
            ablni_kwargs = {}
        
        ablni = AdjustedIntersectionalNetBenefit(**ablni_kwargs)
        score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        self.models[name] = ablni
        self.results[name] = {
            'score': score,
            'subgroup_results': ablni.subgroup_results_,
            'ci': ablni.confidence_interval_
        }
    
    def plot_comparison(self, figsize=(14, 8), save_path=None):
        """
        Plot comparison of models.
        
        Parameters
        ----------
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save figure.
        """
        if len(self.models) == 0:
            raise ValueError("No models added for comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Overall ABLNI scores
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        scores = [self.results[name]['score'] for name in model_names]
        
        colors = ['#2ca02c' if s == max(scores) else '#1f77b4' for s in scores]
        bars = ax1.bar(range(len(model_names)), scores, color=colors, alpha=0.7)
        
        # Add confidence intervals if available
        for i, name in enumerate(model_names):
            ci = self.results[name]['ci']
            if ci is not None and not np.isnan(ci[0]):
                ax1.plot([i, i], ci, 'k-', linewidth=2)
                ax1.plot([i-0.1, i+0.1], [ci[0], ci[0]], 'k-', linewidth=2)
                ax1.plot([i-0.1, i+0.1], [ci[1], ci[1]], 'k-', linewidth=2)
        
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel('ABLNI Score')
        ax1.set_title('Overall Fairness Comparison')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(0.8, color='orange', linestyle='--', alpha=0.5, 
                   label='Acceptable Threshold')
        ax1.legend()
        
        # 2. Net benefit ranges
        ax2 = axes[0, 1]
        for i, name in enumerate(model_names):
            nb_values = self.results[name]['subgroup_results']['net_benefit']
            ax2.boxplot([nb_values], positions=[i], widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors[i], alpha=0.7))
        
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('Net Benefit')
        ax2.set_title('Net Benefit Distribution Across Subgroups')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Worst-case performance
        ax3 = axes[1, 0]
        min_nb = [self.results[name]['subgroup_results']['net_benefit'].min() 
                 for name in model_names]
        
        colors_min = ['#d62728' if s == min(min_nb) else '#ff7f0e' for s in min_nb]
        ax3.bar(range(len(model_names)), min_nb, color=colors_min, alpha=0.7)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.set_ylabel('Minimum Net Benefit')
        ax3.set_title('Worst-Case Subgroup Performance')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Fairness-Performance tradeoff
        ax4 = axes[1, 1]
        
        # Calculate average net benefit across subgroups
        avg_nb = [self.results[name]['subgroup_results']['net_benefit'].mean() 
                 for name in model_names]
        fairness_scores = scores
        
        ax4.scatter(avg_nb, fairness_scores, s=200, alpha=0.6, c=colors)
        
        for i, name in enumerate(model_names):
            ax4.annotate(name, (avg_nb[i], fairness_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Average Net Benefit')
        ax4.set_ylabel('ABLNI Score')
        ax4.set_title('Fairness vs Performance Tradeoff')
        ax4.grid(alpha=0.3)
        
        # Add quadrant lines
        ax4.axhline(0.8, color='gray', linestyle='--', alpha=0.3)
        ax4.axvline(np.mean(avg_nb), color='gray', linestyle='--', alpha=0.3)
        
        plt.suptitle('Model Fairness Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table.
        
        Returns
        -------
        pd.DataFrame
            Comparison table with key metrics.
        """
        comparison_data = []
        
        for name in self.results.keys():
            results = self.results[name]['subgroup_results']
            
            comparison_data.append({
                'Model': name,
                'ABLNI_Score': self.results[name]['score'],
                'CI_Lower': self.results[name]['ci'][0] if self.results[name]['ci'] else np.nan,
                'CI_Upper': self.results[name]['ci'][1] if self.results[name]['ci'] else np.nan,
                'Mean_Net_Benefit': results['net_benefit'].mean(),
                'Min_Net_Benefit': results['net_benefit'].min(),
                'Max_Net_Benefit': results['net_benefit'].max(),
                'Range_Net_Benefit': results['net_benefit'].max() - results['net_benefit'].min(),
                'Std_Net_Benefit': results['net_benefit'].std(),
                'N_Subgroups': len(results)
            })
        
        return pd.DataFrame(comparison_data)


# Utility functions

def simulate_biased_predictions(y_true, sensitive_attrs, bias_strength=0.3, 
                                random_state=None):
    """
    Simulate biased predictions for testing.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    sensitive_attrs : DataFrame, Series, or array
        Sensitive attributes.
    bias_strength : float
        Strength of bias (0 = no bias, 1 = maximum bias).
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    np.ndarray
        Biased predictions.
    """
    np.random.seed(random_state)
    
    y_pred = np.zeros(len(y_true))
    
    if isinstance(sensitive_attrs, pd.DataFrame):
        subgroups = sensitive_attrs.astype(str).agg('_'.join, axis=1)
    else:
        subgroups = pd.Series(sensitive_attrs).astype(str)
    
    unique_groups = subgroups.unique()
    
    # Assign different quality predictions to different groups
    for i, group in enumerate(unique_groups):
        mask = subgroups == group
        bias_factor = 1 - (i / len(unique_groups)) * bias_strength
        
        for j in np.where(mask)[0]:
            if y_true[j] == 1:
                alpha, beta = 5 * bias_factor, 2
            else:
                alpha, beta = 2, 5 * bias_factor
            
            y_pred[j] = np.random.beta(alpha, beta)
    
    return y_pred


def generate_fairness_report_html(ablni_metric, output_path='fairness_report.html'):
    """
    Generate HTML report of fairness analysis.
    
    Parameters
    ----------
    ablni_metric : AdjustedIntersectionalNetBenefit
        Fitted ABLNI metric.
    output_path : str
        Path to save HTML report.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fairness Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .score-good {{ color: #27ae60; font-weight: bold; }}
            .score-warning {{ color: #e67e22; font-weight: bold; }}
            .score-critical {{ color: #e74c3c; font-weight: bold; }}
            .summary-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Clinical AI Fairness Analysis Report</h1>
        
        <div class="summary-box">
            <h2>Overall Fairness Score</h2>
            <p>ABLNI Score: <span class="{score_class}">{score:.4f}</span></p>
            <p>Interpretation: {interpretation}</p>
            {ci_html}
        </div>
        
        <h2>Subgroup Performance Details</h2>
        {table_html}
        
        <h2>Recommendations</h2>
        {recommendations_html}
        
        <p style="margin-top: 40px; color: #7f8c8d; font-size: 12px;">
            Generated on {timestamp}
        </p>
    </body>
    </html>
    """
    
    # Determine score class and interpretation
    score = ablni_metric.overall_score_
    if score >= 0.9:
        score_class = "score-good"
        interpretation = "EXCELLENT - Model demonstrates high fairness across subgroups"
    elif score >= 0.8:
        score_class = "score-good"
        interpretation = "GOOD - Acceptable fairness with minor disparities"
    elif score >= 0.7:
        score_class = "score-warning"
        interpretation = "MODERATE - Notable disparities warrant review"
    else:
        score_class = "score-critical"
        interpretation = "NEEDS ATTENTION - Significant disparities detected"
    
    # CI HTML
    ci_html = ""
    if ablni_metric.confidence_interval_ is not None:
        ci_lower, ci_upper = ablni_metric.confidence_interval_
        ci_html = f"<p>95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})</p>"
    
    # Table HTML
    df = ablni_metric.subgroup_results_
    table_html = df.to_html(index=False, float_format=lambda x: f'{x:.4f}')
    
    # Recommendations
    recommendations = []
    min_nb_row = df.loc[df['net_benefit'].idxmin()]
    recommendations.append(f"• Focus attention on subgroup '{min_nb_row['subgroup']}' which shows the lowest net benefit ({min_nb_row['net_benefit']:.4f})")
    
    if df['n'].min() < 100:
        recommendations.append("• Some subgroups have limited sample sizes. Consider collecting more data for robust evaluation.")
    
    if score < 0.8:
        recommendations.append("• Consider implementing fairness-aware training methods or post-processing techniques")
        recommendations.append("• Consult with domain experts and affected stakeholders")
    
    recommendations_html = "<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>"
    
    # Generate timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Fill template
    html_content = html_template.format(
        score=score,
        score_class=score_class,
        interpretation=interpretation,
        ci_html=ci_html,
        table_html=table_html,
        recommendations_html=recommendations_html,
        timestamp=timestamp
    )
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"[OK] HTML report generated: {output_path}")