"""
Visualization module for keylogger detection results
Generates plots and reports from detection analysis
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

class DetectionVisualizer:
    def __init__(self, output_dir="visualization_output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10
    
    def visualize_performance_metrics(self, metrics_data):
        """Visualize performance evaluation metrics"""
        if not metrics_data:
            print("No metrics data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Detection System Performance Metrics', fontsize=16, fontweight='bold')
        
        # 1. Metrics Bar Chart
        ax1 = axes[0, 0]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics_data['accuracy'] * 100,
            metrics_data['precision'] * 100,
            metrics_data['recall'] * 100,
            metrics_data['f1'] * 100
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Percentage (%)', fontweight='bold')
        ax1.set_title('Performance Metrics', fontweight='bold')
        ax1.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Confusion Matrix
        ax2 = axes[0, 1]
        results_df = pd.DataFrame(metrics_data['results'])
        
        tp = sum((results_df['actual'] == 1) & (results_df['predicted'] == 1))
        fp = sum((results_df['actual'] == 0) & (results_df['predicted'] == 1))
        tn = sum((results_df['actual'] == 0) & (results_df['predicted'] == 0))
        fn = sum((results_df['actual'] == 1) & (results_df['predicted'] == 0))
        
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
                   xticklabels=['Normal', 'Keylogger'], yticklabels=['Normal', 'Keylogger'])
        ax2.set_ylabel('Actual', fontweight='bold')
        ax2.set_xlabel('Predicted', fontweight='bold')
        ax2.set_title('Confusion Matrix', fontweight='bold')
        
        # 3. Detection Results Pie Chart
        ax3 = axes[1, 0]
        detection_counts = pd.Series([tn, fp, fn, tp], index=['TN', 'FP', 'FN', 'TP'])
        colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        wedges, texts, autotexts = ax3.pie(detection_counts, labels=detection_counts.index, autopct='%1.0f',
                                            colors=colors_pie, startangle=90, textprops={'fontweight': 'bold'})
        ax3.set_title('Detection Distribution', fontweight='bold')
        
        # 4. Results Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['True Positives', str(tp)],
            ['False Positives', str(fp)],
            ['True Negatives', str(tn)],
            ['False Negatives', str(fn)],
            ['Total Samples', str(len(results_df))],
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"performance_metrics_{self.timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def visualize_anomaly_scores(self, detection_results):
        """Visualize anomaly scores for detected samples"""
        if not detection_results or not isinstance(detection_results, list):
            print("No detection results to visualize")
            return
        
        # Extract data
        files = []
        scores = []
        classifications = []
        
        for result in detection_results:
            if isinstance(result, dict) and 'file' in result:
                files.append(os.path.basename(result.get('file', 'unknown')))
                scores.append(result.get('anomaly_score', 0))
                classifications.append('Keylogger' if result.get('is_keylogger', False) else 'Normal')
        
        if not scores:
            print("No anomaly scores to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color by classification
        colors = ['#e74c3c' if c == 'Keylogger' else '#2ecc71' for c in classifications]
        
        bars = ax.bar(range(len(files)), scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add threshold line
        threshold = np.mean(scores) - np.std(scores)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
        
        ax.set_xlabel('Sample', fontweight='bold', fontsize=12)
        ax.set_ylabel('Anomaly Score', fontweight='bold', fontsize=12)
        ax.set_title('Anomaly Scores by Sample', fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(files)))
        ax.set_xticklabels(files, rotation=45, ha='right')
        
        # Add legend
        normal_patch = mpatches.Patch(color='#2ecc71', label='Normal', alpha=0.7)
        keylogger_patch = mpatches.Patch(color='#e74c3c', label='Keylogger', alpha=0.7)
        ax.legend(handles=[normal_patch, keylogger_patch, ax.get_lines()[0]], loc='upper right')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"anomaly_scores_{self.timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def visualize_feature_comparison(self, detection_results):
        """Compare detected features"""
        if not detection_results or not isinstance(detection_results, list):
            print("No detection results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Analysis: Detected Samples', fontsize=16, fontweight='bold')
        
        feature_groups = {
            'CPU Metrics': ['cpu_mean', 'cpu_std', 'cpu_max'],
            'Memory Metrics': ['memory_mean', 'memory_std', 'memory_growth'],
            'Thread Metrics': ['threads_mean', 'threads_std', 'threads_max'],
            'Handle Metrics': ['handles_mean', 'handles_std', 'handles_max']
        }
        
        axes_flat = axes.flatten()
        
        for idx, (group_name, features) in enumerate(feature_groups.items()):
            ax = axes_flat[idx]
            
            # Initialize feature_data with all features in the group
            feature_data = {feature: [] for feature in features}
            labels = []
            
            for result in detection_results:
                if isinstance(result, dict) and 'features' in result:
                    features_dict = result['features']
                    label = os.path.basename(result.get('file', 'unknown'))
                    labels.append(label)
                    
                    # Iterate through features and collect data
                    for feature in features:
                        if feature in features_dict:
                            feature_data[feature].append(features_dict[feature])
                        else:
                            feature_data[feature].append(0)
            
            if not labels:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(group_name, fontweight='bold')
                continue
            
            # Create grouped bar chart
            x = np.arange(len(labels))
            width = 0.25
            
            for i, feature in enumerate(features):
                ax.bar(x + (i - 1) * width, feature_data[feature], width, label=feature, alpha=0.8)
            
            ax.set_ylabel('Value', fontweight='bold')
            ax.set_title(group_name, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"feature_analysis_{self.timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def visualize_rule_violations(self, detection_results):
        """Visualize rule violations across samples"""
        if not detection_results or not isinstance(detection_results, list):
            print("No detection results to visualize")
            return
        
        violations_count = {}
        sample_labels = []
        
        for result in detection_results:
            if isinstance(result, dict):
                label = os.path.basename(result.get('file', 'unknown'))
                sample_labels.append(label)
                violations = result.get('rule_violations', [])
                
                for violation in violations:
                    violations_count[violation] = violations_count.get(violation, 0) + 1
        
        if not violations_count:
            print("No rule violations to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        violation_types = list(violations_count.keys())
        violation_values = list(violations_count.values())
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#e67e22']
        
        bars = ax.barh(violation_types, violation_values, color=colors[:len(violation_types)], 
                      alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Count', fontweight='bold', fontsize=12)
        ax.set_title('Rule Violations Distribution', fontweight='bold', fontsize=14)
        
        # Add value labels
        for bar, value in zip(bars, violation_values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(value)}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"rule_violations_{self.timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def generate_html_report(self, metrics_data, detection_results):
        """Generate comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Keylogger Detection Report</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #ecf0f1; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }}
                h1 {{ color: #2c3e50; margin-bottom: 10px; text-align: center; }}
                .timestamp {{ text-align: center; color: #7f8c8d; margin-bottom: 30px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 32px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; opacity: 0.9; margin-top: 10px; }}
                .results-table {{ width: 100%; border-collapse: collapse; margin: 30px 0; }}
                .results-table th {{ background: #34495e; color: white; padding: 15px; text-align: left; }}
                .results-table td {{ padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }}
                .results-table tr:hover {{ background: #f5f5f5; }}
                .status-keylogger {{ background: #e74c3c; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
                .status-normal {{ background: #2ecc71; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
                .violations {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 4px; font-size: 12px; }}
                .section {{ margin: 40px 0; }}
                .section-title {{ font-size: 20px; color: #2c3e50; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #3498db; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
                .stat-box {{ background: #ecf0f1; padding: 15px; border-radius: 4px; text-align: center; }}
                .stat-number {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔒 Keylogger Detection System - Final Report</h1>
                <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                
                <div class="section">
                    <div class="section-title">📊 Performance Metrics</div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{metrics_data['accuracy']*100:.1f}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{metrics_data['precision']*100:.1f}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{metrics_data['recall']*100:.1f}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{metrics_data['f1']*100:.1f}%</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">📈 Confusion Matrix Summary</div>
                    <div class="summary-stats">
        """
        
        results_df = pd.DataFrame(metrics_data['results'])
        tp = sum((results_df['actual'] == 1) & (results_df['predicted'] == 1))
        fp = sum((results_df['actual'] == 0) & (results_df['predicted'] == 1))
        tn = sum((results_df['actual'] == 0) & (results_df['predicted'] == 0))
        fn = sum((results_df['actual'] == 1) & (results_df['predicted'] == 0))
        
        html_content += f"""
                        <div class="stat-box">
                            <div class="stat-number" style="color: #27ae60;">{tp}</div>
                            <div class="stat-label">True Positives</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number" style="color: #e74c3c;">{fp}</div>
                            <div class="stat-label">False Positives</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number" style="color: #27ae60;">{tn}</div>
                            <div class="stat-label">True Negatives</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number" style="color: #e74c3c;">{fn}</div>
                            <div class="stat-label">False Negatives</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">🔍 Detection Results</div>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Sample</th>
                                <th>Classification</th>
                                <th>Anomaly Score</th>
                                <th>Rule Violations</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for result in detection_results:
            if isinstance(result, dict):
                filename = os.path.basename(result.get('file', 'unknown'))
                is_keylogger = result.get('is_keylogger', False)
                status_class = 'status-keylogger' if is_keylogger else 'status-normal'
                status_text = '🚨 Keylogger' if is_keylogger else '✓ Normal'
                anomaly_score = result.get('anomaly_score', 0)
                violations = result.get('rule_violations', [])
                
                violations_html = '<div class="violations">' + '<br>'.join(violations) + '</div>' if violations else '<span style="color: #27ae60;">None</span>'
                
                html_content += f"""
                        <tr>
                            <td><strong>{filename}</strong></td>
                            <td><span class="{status_class}">{status_text}</span></td>
                            <td><code>{anomaly_score:.4f}</code></td>
                            <td>{violations_html}</td>
                        </tr>
                """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <div class="section-title">📁 Generated Visualizations</div>
                    <ul>
                        <li>performance_metrics_*.png - Performance dashboard</li>
                        <li>anomaly_scores_*.png - Anomaly score distribution</li>
                        <li>feature_analysis_*.png - Feature comparison</li>
                        <li>rule_violations_*.png - Violation patterns</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        filename = os.path.join(self.output_dir, f"report_{self.timestamp}.html")
        with open(filename, 'w') as f:
            f.write(html_content)
        print(f"✓ Saved: {filename}")
    
    def create_summary_document(self, metrics_data, detection_results):
        """Create a text summary document"""
        summary = f"""
{'='*80}
KEYLOGGER DETECTION SYSTEM - FINAL REPORT
{'='*80}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PERFORMANCE METRICS
{'-'*80}
Accuracy:  {metrics_data['accuracy']*100:.2f}%
Precision: {metrics_data['precision']*100:.2f}%
Recall:    {metrics_data['recall']*100:.2f}%
F1 Score:  {metrics_data['f1']*100:.2f}%

CONFUSION MATRIX
{'-'*80}
"""
        
        results_df = pd.DataFrame(metrics_data['results'])
        tp = sum((results_df['actual'] == 1) & (results_df['predicted'] == 1))
        fp = sum((results_df['actual'] == 0) & (results_df['predicted'] == 1))
        tn = sum((results_df['actual'] == 0) & (results_df['predicted'] == 0))
        fn = sum((results_df['actual'] == 1) & (results_df['predicted'] == 0))
        
        summary += f"""
True Positives:  {tp}
False Positives: {fp}
True Negatives:  {tn}
False Negatives: {fn}

DETECTION RESULTS
{'-'*80}
"""
        
        for result in detection_results:
            if isinstance(result, dict):
                filename = os.path.basename(result.get('file', 'unknown'))
                is_keylogger = result.get('is_keylogger', False)
                status = 'KEYLOGGER DETECTED' if is_keylogger else 'NORMAL BEHAVIOR'
                anomaly_score = result.get('anomaly_score', 0)
                violations = result.get('rule_violations', [])
                
                summary += f"\nFile: {filename}\n"
                summary += f"Classification: {status}\n"
                summary += f"Anomaly Score: {anomaly_score:.4f}\n"
                if violations:
                    summary += f"Violations: {', '.join(violations)}\n"
                else:
                    summary += "Violations: None\n"
        
        summary += f"\n{'='*80}\n"
        summary += "VISUALIZATIONS GENERATED:\n"
        summary += f"  - Performance metrics dashboard\n"
        summary += f"  - Anomaly scores chart\n"
        summary += f"  - Feature analysis plots\n"
        summary += f"  - Rule violations distribution\n"
        summary += f"  - HTML report\n"
        summary += f"\nAll files saved in: {self.output_dir}\n"
        summary += f"{'='*80}\n"
        
        filename = os.path.join(self.output_dir, f"summary_{self.timestamp}.txt")
        with open(filename, 'w') as f:
            f.write(summary)
        print(f"✓ Saved: {filename}")
        print(summary)