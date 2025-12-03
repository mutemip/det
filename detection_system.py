"""
Behavioral Keylogger Detection System
Uses anomaly detection to identify keylogger behavior patterns
"""

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
        if not metrics_list:
            return None
        
        df = pd.DataFrame(metrics_list)
        
        # Calculate features with safe column access - match actual column names
        features = {
            # CPU metrics
            'cpu_mean': df['cpu_percent'].mean() if 'cpu_percent' in df.columns else 0,
            'cpu_std': df['cpu_percent'].std() if 'cpu_percent' in df.columns else 0,
            'cpu_max': df['cpu_percent'].max() if 'cpu_percent' in df.columns else 0,
            
            # Memory metrics
            'memory_mean': df['memory_mb'].mean() if 'memory_mb' in df.columns else 0,
            'memory_std': df['memory_mb'].std() if 'memory_mb' in df.columns else 0,
            'memory_growth': (df['memory_mb'].iloc[-1] - df['memory_mb'].iloc[0]) if ('memory_mb' in df.columns and len(df) > 1) else 0,
            
            # Thread metrics
            'threads_mean': df['threads'].mean() if 'threads' in df.columns else 0,
            'threads_std': df['threads'].std() if 'threads' in df.columns else 0,
            'threads_max': df['threads'].max() if 'threads' in df.columns else 0,
            
            # Handle metrics
            'handles_mean': df['handles'].mean() if 'handles' in df.columns else 0,
            'handles_std': df['handles'].std() if 'handles' in df.columns else 0,
            'handles_max': df['handles'].max() if 'handles' in df.columns else 0,
        }
        
        return features
    
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
    
    def train_model(self, contamination=0.1):
        """Train anomaly detection model on baseline data"""
        print("\nTraining detection model...")
        
        # Load baseline data
        baseline_df = self.load_baseline()
        
        if baseline_df is None or len(baseline_df) == 0:
            print("❌ No baseline data available for training")
            return None
        
        print(f"Baseline data shape: {baseline_df.shape}")
        
        # Split into windows for feature extraction
        window_size = 10
        features_list = []
        
        for i in range(0, len(baseline_df), window_size):
            window = baseline_df.iloc[i:i+window_size]
            if len(window) >= 5:
                features = self.extract_features(window.to_dict('records'))
                if features:
                    features_list.append(features)
        
        print(f"Extracted {len(features_list)} feature windows")
        
        if len(features_list) < 10:
            print("⚠️  WARNING: Very few feature windows. Increase baseline duration!")
            return None
        
        # Convert to DataFrame
        X_train = pd.DataFrame(features_list)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Features: {list(X_train.columns)}")
        
        # Calculate baseline statistics BEFORE scaling
        self.baseline_stats = {
            col: {
                'mean': X_train[col].mean(),
                'std': X_train[col].std(),
                'min': X_train[col].min(),
                'max': X_train[col].max()
            }
            for col in X_train.columns
        }
        
        # Initialize and fit StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=150
        )
        self.model.fit(X_train_scaled)
        
        print(f"✓ Model trained successfully on {len(X_train)} feature windows")
        print(f"✓ Contamination parameter: {contamination}")
        
        return X_train
    
    def detect_from_file(self, keylogger_file):
        """Detect keylogger from logged data file"""
        print(f"\nAnalyzing: {keylogger_file}")
        
        with open(keylogger_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('system_metrics', [])
        
        if not metrics:
            print("No metrics found in file")
            return None
        
        # Debug: print available columns
        if metrics:
            print(f"Available columns: {list(metrics[0].keys())}")
            print(f"Total metrics collected: {len(metrics)}")  # ADD THIS
            print(f"Keystrokes: {data.get('keystroke_count', 0)}")  # ADD THIS
        
        # Extract features
        features = self.extract_features(metrics)
        
        if not features:
            print("Could not extract features")
            return None
        
        # Convert to DataFrame
        X_test = pd.DataFrame([features])
        
        # Normalize
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        prediction = self.model.predict(X_test_scaled)[0]
        anomaly_score = self.model.score_samples(X_test_scaled)[0]
        
        # Rule-based detection
        rule_violations = self._check_rules(features)
        
        # FIX: Change detection logic - keyloggers SHOULD have anomalies
        # Negative anomaly score = ANOMALY, Positive = NORMAL
        is_keylogger = prediction == -1 or len(rule_violations) >= 2  # CHANGED from >= 3
        
        result = {
            'file': keylogger_file,
            'is_keylogger': is_keylogger,
            'ml_prediction': 'ANOMALY' if prediction == -1 else 'NORMAL',
            'anomaly_score': float(anomaly_score),
            'rule_violations': rule_violations,
            'features': features
        }
        
        self._print_detection_result(result)
        
        return result
    
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
                        "num_threads": current_process.num_threads(),
                        "num_handles": current_process.num_handles() if hasattr(current_process, 'num_handles') else 0,
                        "io_counters": {
                            "read_bytes": current_process.io_counters().read_bytes,
                            "write_bytes": current_process.io_counters().write_bytes,
                            "read_count": current_process.io_counters().read_count,
                            "write_count": current_process.io_counters().write_count
                        },
                        "open_files": len(current_process.open_files()),
                        "connections": len(current_process.connections()),
                        "keystroke_count": 0  # Would need keyboard hook to get actual count
                    }
                    metrics_buffer.append(metrics)
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                
                # Check if we have enough data to analyze
                if len(metrics_buffer) >= 10:
                    features = self.extract_features(metrics_buffer[-30:])  # Use last 30 samples
                    
                    if features:
                        # Check for anomalies
                        X_test = pd.DataFrame([features])
                        X_test_scaled = self.scaler.transform(X_test)
                        prediction = self.model.predict(X_test_scaled)[0]
                        rule_violations = self._check_rules(features)
                        
                        if prediction == -1 or len(rule_violations) >= 2:
                            alert = {
                                'timestamp': datetime.now().isoformat(),
                                'prediction': 'ANOMALY',
                                'violations': rule_violations,
                                'features': features
                            }
                            self._trigger_alert(alert)
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.is_monitoring = False
            print("\nMonitoring completed")
    
    def _trigger_alert(self, alert):
        """Trigger detection alert"""
        print(f"\n🚨 ALERT: Suspicious behavior detected at {alert['timestamp']}")
        print(f"Violations: {', '.join(alert['violations'])}")
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
    
    def evaluate_performance(self, keylogger_files, normal_files=None):
        """Evaluate detector performance"""
        print("\n" + "="*70)
        print("PERFORMANCE EVALUATION")
        print("="*70)
        
        results = []
        
        # Test on keylogger files
        print("\nTesting keylogger samples...")
        for kf in keylogger_files:
            result = self.detect_from_file(kf)
            if result:
                results.append({
                    'file': kf,
                    'actual': 1,  # Keylogger
                    'predicted': 1 if result['is_keylogger'] else 0
                })
        
        # Test on normal files if provided
        if normal_files:
            print("\nTesting normal samples...")
            for nf in normal_files:
                result = self.detect_from_file(nf)
                if result:
                    results.append({
                        'file': nf,
                        'actual': 0,  # Normal
                        'predicted': 1 if result['is_keylogger'] else 0
                    })
        
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
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'results': results
            }
        
        return None
    
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
                        'predicted': 1 if result['is_keylogger'] else 0
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
                            'predicted': 1 if result['is_keylogger'] else 0
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
                visualizer = DetectionVisualizer()
                visualizer.visualize_performance_metrics(metrics_data)
                visualizer.visualize_anomaly_scores(detection_results)
                visualizer.visualize_feature_comparison(detection_results)
                visualizer.visualize_rule_violations(detection_results)
                visualizer.generate_html_report(metrics_data, detection_results)
                visualizer.create_summary_document(metrics_data, detection_results)
                
                print("\n✓ All visualizations generated successfully!")
                print(f"📁 Output directory: {visualizer.output_dir}\n")
                
                return metrics_data
            
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
