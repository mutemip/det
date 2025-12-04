"""
Behavioral Keylogger Detection System
Uses anomaly detection to identify keylogger behavior patterns
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil
import time
import threading
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from visualization import DetectionVisualizer



class KeyloggerDetector:
    def __init__(self, baseline_file="baseline_data.json"):
        self.baseline_file = baseline_file
        self.model = None
        self.scaler = StandardScaler()
        self.is_monitoring = False
        self.alert_callback = None
        self.baseline_stats = {}
        
    def load_baseline(self):
        """Load and process baseline data"""
        print("Loading baseline data...")
        
        with open(self.baseline_file, 'r') as f:
            data = json.load(f)
        
        metrics = data['metrics']
        df = pd.DataFrame(metrics)
        
        print(f"Loaded {len(df)} baseline samples")
        return df
    
    def extract_features(self, metrics_list):
        """Extract features from metrics for detection"""
        if not metrics_list or len(metrics_list) == 0:
            return None
        
        try:
            df = pd.DataFrame(metrics_list)
            
            # Ensure all required columns exist
            required_cols = {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'threads': 3.0,
                'handles': 0.0
            }
            
            for col, default_val in required_cols.items():
                if col not in df.columns:
                    df[col] = default_val
            
            # Calculate features with type conversion
            features = {
                # CPU metrics
                'cpu_mean': float(df['cpu_percent'].mean()) if len(df) > 0 else 0.0,
                'cpu_std': float(df['cpu_percent'].std()) if len(df) > 1 else 0.0,
                'cpu_max': float(df['cpu_percent'].max()) if len(df) > 0 else 0.0,
                'cpu_min': float(df['cpu_percent'].min()) if len(df) > 0 else 0.0,
                
                # Memory metrics
                'memory_mean': float(df['memory_mb'].mean()) if len(df) > 0 else 0.0,
                'memory_std': float(df['memory_mb'].std()) if len(df) > 1 else 0.0,
                'memory_max': float(df['memory_mb'].max()) if len(df) > 0 else 0.0,
                'memory_min': float(df['memory_mb'].min()) if len(df) > 0 else 0.0,
                'memory_growth': float(df['memory_mb'].max() - df['memory_mb'].min()) if len(df) > 0 else 0.0,
                
                # Thread metrics
                'threads_mean': float(df['threads'].mean()) if len(df) > 0 else 0.0,
                'threads_std': float(df['threads'].std()) if len(df) > 1 else 0.0,
                'threads_max': float(df['threads'].max()) if len(df) > 0 else 0.0,
                
                # Handle metrics
                'handles_mean': float(df['handles'].mean()) if len(df) > 0 else 0.0,
                'handles_std': float(df['handles'].std()) if len(df) > 1 else 0.0,
            }
            
            # Replace any NaN or inf values
            for key in features:
                if pd.isna(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0
            
            return features
        
        except Exception as e:
            print(f"[WARNING] Error extracting features: {e}")
            return None
    
    def _check_rules(self, features):
        """Rule-based detection for known patterns"""
        violations = []
        
        # Rule 1: High CPU usage
        if 'cpu_mean' in self.baseline_stats and 'cpu_mean' in features:
            if features['cpu_mean'] > self.baseline_stats['cpu_mean']['mean'] + 2 * self.baseline_stats['cpu_mean']['std']:
                violations.append("High CPU usage")
        
        # Rule 2: Excessive memory growth
        if features.get('memory_growth', 0) > 10:  # More than 10 MB growth
            violations.append("Rapid memory growth")
        
        # Rule 3: High thread count
        if 'threads_mean' in self.baseline_stats and 'threads_max' in features:
            if features['threads_max'] > self.baseline_stats['threads_mean']['mean'] + 2 * self.baseline_stats['threads_mean']['std']:
                violations.append("Unusual thread count")
        
        # Rule 4: Suspicious handle count
        if 'handles_mean' in self.baseline_stats and 'handles_mean' in features:
            if features['handles_mean'] > self.baseline_stats['handles_mean']['mean'] + 2 * self.baseline_stats['handles_mean']['std']:
                violations.append("Elevated handle count")
        
        return violations
    
    def train_model(self, baseline_file="baseline_data.json", contamination=0.1):
        """Train anomaly detection model on baseline data"""
        print("Loading baseline data...")
        
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Baseline file not found: {baseline_file}")
            return
        
        metrics = data.get('metrics', [])
        
        if len(metrics) < 10:
            print(f"⚠️  WARNING: Very few samples ({len(metrics)}). Minimum recommended: 50+")
            # Continue anyway with available data
        
        df = pd.DataFrame(metrics)
        
        print(f"Loaded {len(df)} baseline samples")
        
        # Extract features using sliding windows
        feature_list = []
        window_size = max(5, len(df) // 3)  # Adaptive window size
        
        for i in range(0, len(df), max(1, window_size // 2)):  # 50% overlap
            window = df.iloc[i:i+window_size]
            if len(window) >= 3:  # Minimum window size
                features = self.extract_features(window.to_dict('records'))
                if features:
                    feature_list.append(features)
        
        if not feature_list:
            print("[ERROR] Could not extract features from baseline")
            return
        
        # Convert feature list to DataFrame then to numpy array
        features_df = pd.DataFrame(feature_list)
        
        # Fill any NaN values with 0
        features_df = features_df.fillna(0)
        
        X = features_df.values  # Convert to numpy array
        
        print(f"Extracted {len(X)} feature vectors")
        
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(contamination=min(contamination, 0.5), random_state=42)
        self.model.fit(X_scaled)
        
        # Store baseline stats for rule-based detection
        self.baseline_stats = {
            'cpu_mean': {
                'mean': df['cpu_percent'].mean() if 'cpu_percent' in df.columns else 0,
                'std': df['cpu_percent'].std() if 'cpu_percent' in df.columns else 1
            },
            'memory_mean': {
                'mean': df['memory_mb'].mean() if 'memory_mb' in df.columns else 0,
                'std': df['memory_mb'].std() if 'memory_mb' in df.columns else 1
            },
            'threads_mean': {
                'mean': df['threads'].mean() if 'threads' in df.columns else 3,
                'std': df['threads'].std() if 'threads' in df.columns else 0.5
            }
        }
        
        print("✓ Model trained successfully")

        
    def detect_from_file(self, file_path):
        """Detect if a file contains keylogger data"""
        print(f"\nAnalyzing: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load file: {e}")
            return None
        
        try:
            # Get metrics from the file
            metrics = data.get('system_metrics', [])
            keystrokes = data.get('keystrokes', [])
            
            if not metrics:
                print(f"[WARNING] No system metrics found in {file_path}")
                return None
            
            # Convert metrics to DataFrame
            df_metrics = pd.DataFrame(metrics)
            
            # Extract features from metrics
            features = self.extract_features(df_metrics.to_dict('records'))
            
            if not features:
                print(f"[ERROR] Could not extract features from {file_path}")
                return None
            
            # Ensure all feature values are numeric
            features = {k: float(v) if v is not None else 0.0 for k, v in features.items()}
            
            # Check if model is trained
            if self.model is None:
                print("[WARNING] Model not trained yet. Training now...")
                self.train_model()
            
            # Make prediction
            X_test = pd.DataFrame([features])
            X_test_scaled = self.scaler.transform(X_test)
            prediction = self.model.predict(X_test_scaled)[0]
            anomaly_score = self.model.score_samples(X_test_scaled)[0]
            
            # Check rule violations
            rule_violations = self._check_rules(features)
            
            # Determine if keylogger
            is_keylogger = (prediction == -1) or (len(rule_violations) >= 2)
            
            result = {
                'file': file_path,
                'is_keylogger': is_keylogger,
                'ml_prediction': 'ANOMALY' if prediction == -1 else 'NORMAL',
                'anomaly_score': float(anomaly_score),
                'rule_violations': rule_violations,
                'keystroke_count': len([k for k in keystrokes if isinstance(k, dict) and 'timestamp' in k]),
                'features': features
            }
            
            self._print_detection_result(result)
            return result
        
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _check_rules(self, features):
        """Rule-based detection for known patterns"""
        violations = []
        
        # Rule 1: High CPU usage
        if 'cpu_mean' in self.baseline_stats and features['cpu_mean'] > self.baseline_stats['cpu_mean']['mean'] + 2 * self.baseline_stats['cpu_mean']['std']:
            violations.append("High CPU usage")
        
        # Rule 2: Excessive memory growth
        if features['memory_growth'] > 10:  # More than 10 MB growth
            violations.append("Rapid memory growth")
        
        # Rule 3: High thread count
        if 'threads_mean' in self.baseline_stats and features['threads_max'] > self.baseline_stats['threads_mean']['mean'] + 2 * self.baseline_stats['threads_mean']['std']:
            violations.append("Unusual thread count")
        
        # Rule 4: Suspicious handle count
        if 'handles_mean' in self.baseline_stats and features['handles_mean'] > self.baseline_stats['handles_mean']['mean'] + 2 * self.baseline_stats['handles_mean']['std']:
            violations.append("Elevated handle count")
        
        return violations
    
    def _print_detection_result(self, result):
        """Print detection result in formatted way"""
        print("\n" + "="*70)
        print("DETECTION RESULT")
        print("="*70)
        print(f"File: {result['file']}")
        print(f"Classification: {'🚨 KEYLOGGER DETECTED' if result['is_keylogger'] else '✓ Normal Behavior'}")
        print(f"ML Prediction: {result['ml_prediction']}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"\nRule Violations ({len(result['rule_violations'])}):")
        if result['rule_violations']:
            for violation in result['rule_violations']:
                print(f"  - {violation}")
        else:
            print("  None")
        print("="*70)
    
    def real_time_monitor(self, duration_seconds=60, check_interval=5):
        """Monitor system in real-time for keylogger behavior"""
        print(f"\n{'='*70}")
        print("REAL-TIME MONITORING")
        print(f"{'='*70}")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Check interval: {check_interval} seconds")
        print("Monitoring started...\n")
        
        # Ensure model and scaler are trained
        if self.model is None or not hasattr(self.scaler, 'mean_'):
            print("[INFO] Model or scaler not fitted. Training now...")
            self.train_model()
        
        self.is_monitoring = True
        start_time = time.time()
        metrics_buffer = []
        
        try:
            while time.time() - start_time < duration_seconds and self.is_monitoring:
                # Collect current metrics
                try:
                    current_process = psutil.Process()
                    metrics = {
                        "timestamp": time.time(),
                        "cpu_percent": current_process.cpu_percent(interval=0.5),
                        "memory_mb": current_process.memory_info().rss / (1024 * 1024),
                        "threads": current_process.num_threads(),
                        "handles": current_process.num_handles() if hasattr(current_process, 'num_handles') else 0,
                    }
                    metrics_buffer.append(metrics)
                except Exception as e:
                    pass
                
                # Check if we have enough data to analyze
                if len(metrics_buffer) >= 10:
                    features = self.extract_features(metrics_buffer[-30:])  # Use last 30 samples
                    
                    if features:
                        try:
                            # Ensure all feature values are numeric
                            features = {k: float(v) if v is not None else 0.0 for k, v in features.items()}
                            
                            # Convert to 2D array for sklearn
                            X_test = pd.DataFrame([features])
                            X_test_scaled = self.scaler.transform(X_test)
                            prediction = self.model.predict(X_test_scaled)[0]
                            anomaly_score = self.model.score_samples(X_test_scaled)[0]
                            rule_violations = self._check_rules(features)
                            
                            if prediction == -1 or len(rule_violations) >= 2:
                                alert = {
                                    'timestamp': datetime.now().isoformat(),
                                    'prediction': 'ANOMALY',
                                    'anomaly_score': float(anomaly_score),
                                    'violations': rule_violations,
                                    'features': features
                                }
                                self._trigger_alert(alert)
                        except Exception as e:
                            pass
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.is_monitoring = False
            print("\nMonitoring completed")

    def _trigger_alert(self, alert):
        """Trigger detection alert"""
        print(f"\n🚨 ALERT: Suspicious behavior detected at {alert['timestamp']}")
        print(f"Anomaly Score: {alert['anomaly_score']:.4f}")
        if alert['violations']:
            print(f"Violations: {', '.join(alert['violations'])}")
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
    
    def evaluate_performance(self, keylogger_files, normal_files=None):
        """Evaluate detector performance with visualizations"""
        print("\n" + "="*70)
        print("PERFORMANCE EVALUATION")
        print("="*70)
        
        results = []
        detection_results = []  # For visualizations
        
        # Test on keylogger files
        print("\nTesting keylogger samples...")
        for kf in keylogger_files:
            result = self.detect_from_file(kf)
            if result:
                results.append({
                    'file': kf,
                    'actual': 1,  # Keylogger
                    'predicted': 1 if result.get('is_keylogger') else 0
                })
                detection_results.append(result)
        
        # Test on normal files if provided
        if normal_files:
            print("\nTesting normal samples...")
            for nf in normal_files:
                result = self.detect_from_file(nf)
                if result:
                    results.append({
                        'file': nf,
                        'actual': 0,  # Normal
                        'predicted': 1 if result.get('is_keylogger') else 0
                    })
                    detection_results.append(result)
        
        # Calculate metrics
        if results:
            df_results = pd.DataFrame(results)
            accuracy = accuracy_score(df_results['actual'], df_results['predicted'])
            precision = precision_score(df_results['actual'], df_results['predicted'], zero_division=0)
            recall = recall_score(df_results['actual'], df_results['predicted'], zero_division=0)
            f1 = f1_score(df_results['actual'], df_results['predicted'], zero_division=0)
            
            print("\n" + "="*70)
            print("PERFORMANCE METRICS")
            print("="*70)
            print(f"Accuracy:  {accuracy*100:.2f}%")
            print(f"Precision: {precision*100:.2f}%")
            print(f"Recall:    {recall*100:.2f}%")
            print(f"F1 Score:  {f1*100:.2f}%")
            print(f"\nTrue Positives:  {sum((df_results['actual'] == 1) & (df_results['predicted'] == 1))}")
            print(f"False Positives: {sum((df_results['actual'] == 0) & (df_results['predicted'] == 1))}")
            print(f"True Negatives:  {sum((df_results['actual'] == 0) & (df_results['predicted'] == 0))}")
            print(f"False Negatives: {sum((df_results['actual'] == 1) & (df_results['predicted'] == 0))}")
            print("="*70)
            
            metrics_data = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'results': results
            }
            
            # Generate visualizations
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
            try:
                visualizer = DetectionVisualizer()
                visualizer.visualize_performance_metrics(metrics_data)
                visualizer.visualize_anomaly_scores(detection_results)
                visualizer.visualize_feature_comparison(detection_results)
                visualizer.visualize_rule_violations(detection_results)
                visualizer.generate_html_report(metrics_data, detection_results)
                visualizer.create_summary_document(metrics_data, detection_results)
                
                print("\n✓ All visualizations generated successfully!")
                print(f"📁 Output directory: {visualizer.output_dir}\n")
            except Exception as e:
                print(f"[WARNING] Visualization error: {e}")
            
            return metrics_data
        else:
            print("❌ No results to evaluate")
            return None


# ============================================================================
# Main Testing Interface
# ============================================================================
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("KEYLOGGER DETECTION SYSTEM")
    print("="*70)
    
    detector = KeyloggerDetector("baseline_data.json")
    
    # Train the model
    detector.train_model(contamination=0.1)
    
    # Test on keylogger variants
    keylogger_files = [
        "keylog_variant1.json",
        "keylog_variant2.json",
        "keylog_variant3.json"
    ]
    
    # Evaluate performance
    detector.evaluate_performance(keylogger_files)
    
    # Optional: Real-time monitoring
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        detector.real_time_monitor(duration_seconds=60, check_interval=5)
