"""
Master System: Complete Keylogger Creation and Detection Research Framework
Integrates all three phases of the research methodology
"""

import os
import sys
import json
from datetime import datetime


class KeyloggerResearchSystem:
    def __init__(self):
        self.baseline_file = "baseline_data.json"
        self.results_dir = "research_results"
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.results_dir}/reports", exist_ok=True)
    
    def print_menu(self):
        """Display main menu"""
        print("\n" + "="*70)
        print("KEYLOGGER CREATION AND DETECTION RESEARCH SYSTEM")
        print("="*70)
        print("\nPHASE 1: Baseline Collection")
        print("  1. Collect baseline behavior data (normal typing)")
        print("  2. View baseline statistics")
        print("\nPHASE 2: Attack Simulation (VM ONLY!)")
        print("  3. Run keylogger variants (Global Hook)")
        print("  4. Run keylogger variants (Polling)")
        print("  5. Run keylogger variants (Simple Hook)")
        print("  6. Run all keylogger variants")
        print("\nPHASE 3: Detection and Analysis")
        print("  7. Train detection model")
        print("  8. Detect keyloggers from files")
        print("  9. Real-time monitoring")
        print("  10. Full performance evaluation")
        print("\nCOMPLETE WORKFLOW")
        print("  11. Run complete research workflow")
        print("\n  0. Exit")
        print("="*70)
    
    def collect_baseline(self):
        """Phase 1: Collect baseline data"""
        print("\n--- PHASE 1: BASELINE COLLECTION ---")
        print("This will monitor your normal typing behavior.")
        
        duration = input("Enter collection duration in minutes [default: 5]: ")
        duration = int(duration) if duration.strip() else 5
        
        # ENSURE MINIMUM DURATION
        if duration < 3:
            print("⚠️  Warning: Minimum 3 minutes recommended for accurate baseline")
            duration = 3
        
        from baseline_collector import CrossPlatformBaselineCollector
        collector = CrossPlatformBaselineCollector(self.baseline_file)
        collector.start_collection(duration_minutes=duration)
    
    def view_baseline_stats(self):
        """View baseline statistics"""
        if not os.path.exists(self.baseline_file):
            print("❌ No baseline data found. Run baseline collection first.")
            return
        
        with open(self.baseline_file, 'r') as f:
            data = json.load(f)
        
        if not data.get('metrics'):
            print("❌ No metrics in baseline data")
            return
        
        import pandas as pd
        df = pd.DataFrame(data['metrics'])
        
        print("\n" + "="*70)
        print("BASELINE STATISTICS")
        print("="*70)
        print(f"\nTotal keystrokes: {data['keystroke_count']}")
        print(f"Metrics collected: {len(df)}")
        print(f"Duration: {data.get('start_time', 'N/A')} to {data.get('end_time', 'N/A')}")
        
        # CPU metrics
        print(f"\n📊 CPU Usage:")
        print(f"  Mean: {df['cpu_percent'].mean():.2f}%")
        print(f"  Std:  {df['cpu_percent'].std():.2f}%")
        print(f"  Max:  {df['cpu_percent'].max():.2f}%")
        
        # Memory metrics
        print(f"\n📊 Memory Usage:")
        print(f"  Mean: {df['memory_mb'].mean():.2f} MB")
        print(f"  Std:  {df['memory_mb'].std():.2f} MB")
        print(f"  Max:  {df['memory_mb'].max():.2f} MB")
        
        # Thread metrics (CHANGED from 'num_threads' to 'threads')
        print(f"\n📊 Threads:")
        print(f"  Mean: {df['threads'].mean():.1f}")
        print(f"  Std:  {df['threads'].std():.1f}")
        print(f"  Max:  {df['threads'].max():.1f}")
        
        # Handle metrics
        if 'handles' in df.columns:
            print(f"\n📊 Handles:")
            print(f"  Mean: {df['handles'].mean():.1f}")
            print(f"  Max:  {df['handles'].max():.1f}")
        
        print(f"\n📊 System Metrics:")
        print(f"  System CPU Mean: {df['system_cpu'].mean():.2f}%")
        print(f"  System Memory Mean: {df['system_memory'].mean():.2f}%")
        
        print("="*70 + "\n")
    
    def run_keylogger_variant(self, variant_num):
        """Phase 2: Run specific keylogger variant"""
        print(f"\n--- PHASE 2: KEYLOGGER VARIANT {variant_num} ---")
        print("⚠️  WARNING: Use only in isolated VM environment!")
        
        confirm = input("Are you in an isolated VM? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Aborted. Run only in VM.")
            return
        
        duration = input("Enter test duration in seconds [default: 30]: ")
        duration = int(duration) if duration.strip() else 30
        
        from keylogger_variants import GlobalHookKeylogger, PollingKeylogger, SimpleHookKeylogger
        
        if variant_num == 1:
            keylogger = GlobalHookKeylogger(f"{self.results_dir}/logs/keylog_variant1.json")
        elif variant_num == 2:
            keylogger = PollingKeylogger(f"{self.results_dir}/logs/keylog_variant2.json")
        elif variant_num == 3:
            keylogger = SimpleHookKeylogger(f"{self.results_dir}/logs/keylog_variant3.json")
        else:
            print("Invalid variant")
            return
        
        keylogger.start(duration_seconds=duration)
    
    def diagnose_baseline(self):
        """Diagnose baseline data quality"""
        if not os.path.exists(self.baseline_file):
            print("No baseline data")
            return
        
        with open(self.baseline_file, 'r') as f:
            data = json.load(f)
        
        print("\n" + "="*70)
        print("BASELINE DIAGNOSTIC")
        print("="*70)
        print(f"Keystrokes captured: {data['keystroke_count']}")
        print(f"Metrics samples: {len(data['metrics'])}")
        
        if data['metrics']:
            import pandas as pd
            df = pd.DataFrame(data['metrics'])
            print(f"\nCPU variance: {df['cpu_percent'].std():.2f}")
            print(f"Memory variance: {df['memory_mb'].std():.2f}")
            print(f"Thread variance: {df['threads'].std():.2f}")
        
        print("="*70)
    
    def run_all_variants(self):
        """Run all keylogger variants"""
        print("\n--- RUNNING ALL VARIANTS ---")
        print("⚠️  WARNING: Use only in isolated VM environment!")
        
        confirm = input("Are you in an isolated VM? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Aborted. Run only in VM.")
            return
        
        duration = input("Enter test duration per variant in seconds [default: 30]: ")
        duration = int(duration) if duration.strip() else 30
        
        from keylogger_variants import GlobalHookKeylogger, PollingKeylogger, SimpleHookKeylogger
        
        variants = [
            ("Global Hook", GlobalHookKeylogger(f"{self.results_dir}/logs/keylog_variant1.json")),
            ("Polling", PollingKeylogger(f"{self.results_dir}/logs/keylog_variant2.json")),
            ("Simple Hook", SimpleHookKeylogger(f"{self.results_dir}/logs/keylog_variant3.json"))
        ]
        
        for name, keylogger in variants:
            print(f"\n--- Testing {name} ---")
            keylogger.start(duration_seconds=duration)
            print("Waiting 5 seconds before next variant...")
            import time
            time.sleep(5)
    
    def train_detector(self):
        """Phase 3: Train detection model"""
        print("\n--- PHASE 3: TRAINING DETECTION MODEL ---")
        
        if not os.path.exists(self.baseline_file):
            print("❌ No baseline data found. Run baseline collection first.")
            return
        
        from detection_system import KeyloggerDetector
        detector = KeyloggerDetector(self.baseline_file)
        detector.train_model(contamination=0.1)
        
        # Save model
        import pickle
        with open(f"{self.results_dir}/detector_model.pkl", 'wb') as f:
            pickle.dump({'model': detector.model, 'scaler': detector.scaler, 
                        'baseline_stats': detector.baseline_stats}, f)
        
        print("✓ Model trained and saved successfully")
    
    def detect_from_files(self):
        """Detect keyloggers from log files"""
        print("\n--- DETECTING KEYLOGGERS FROM FILES ---")
        
        from detection_system import KeyloggerDetector
        detector = KeyloggerDetector(self.baseline_file)
        
        # Load trained model
        try:
            import pickle
            with open(f"{self.results_dir}/detector_model.pkl", 'rb') as f:
                saved = pickle.load(f)
                detector.model = saved['model']
                detector.scaler = saved['scaler']
                detector.baseline_stats = saved['baseline_stats']
        except:
            print("No trained model found. Training now...")
            detector.train_model()
        
        # Find keylogger log files
        log_dir = f"{self.results_dir}/logs"
        log_files = [f"{log_dir}/{f}" for f in os.listdir(log_dir) if f.endswith('.json')]
        
        if not log_files:
            print("No keylogger log files found.")
            return
        
        print(f"Found {len(log_files)} log files")
        
        for log_file in log_files:
            detector.detect_from_file(log_file)
    
    def real_time_monitoring(self):
        """Run real-time monitoring"""
        print("\n--- REAL-TIME MONITORING ---")
        
        from detection_system import KeyloggerDetector
        detector = KeyloggerDetector(self.baseline_file)
        
        # Load trained model
        try:
            import pickle
            with open(f"{self.results_dir}/detector_model.pkl", 'rb') as f:
                saved = pickle.load(f)
                detector.model = saved['model']
                detector.scaler = saved['scaler']
                detector.baseline_stats = saved['baseline_stats']
        except:
            print("No trained model found. Training now...")
            detector.train_model()
        
        duration = input("Enter monitoring duration in seconds [default: 60]: ")
        duration = int(duration) if duration.strip() else 60
        
        detector.real_time_monitor(duration_seconds=duration, check_interval=5)
    
    def full_evaluation(self):
        """Complete performance evaluation"""
        print("\n--- FULL PERFORMANCE EVALUATION ---")
        
        from detection_system import KeyloggerDetector
        detector = KeyloggerDetector(self.baseline_file)
        
        # Train model
        detector.train_model()
        
        # Find all keylogger files
        log_dir = f"{self.results_dir}/logs"
        keylogger_files = [f"{log_dir}/{f}" for f in os.listdir(log_dir) if f.endswith('.json')]
        
        if not keylogger_files:
            print("No keylogger files found. Run keylogger variants first.")
            return
        
        # Evaluate
        results = detector.evaluate_performance(keylogger_files)
        
        # Save results
        if results:
            report_file = f"{self.results_dir}/reports/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {report_file}")
    
    def complete_workflow(self):
        """Run complete research workflow"""
        print("\n" + "="*70)
        print("COMPLETE RESEARCH WORKFLOW")
        print("="*70)
        print("\nThis will run all three phases:")
        print("1. Collect baseline (5 min)")
        print("2. Run keylogger variants (30s each)")
        print("3. Train detector and evaluate")
        
        confirm = input("\nProceed? (yes/no): ")
        if confirm.lower() != 'yes':
            return
        
        # Phase 1
        print("\n>>> PHASE 1: Collecting baseline...")
        self.collect_baseline()
        
        # Phase 2
        print("\n>>> PHASE 2: Running keylogger variants...")
        self.run_all_variants()
        
        # Phase 3
        print("\n>>> PHASE 3: Training and evaluating...")
        self.full_evaluation()
        
        print("\n" + "="*70)
        print("✓ COMPLETE WORKFLOW FINISHED")
        print(f"✓ Results saved in: {self.results_dir}/")
        print("="*70)
    
    def run(self):
        """Main system loop"""
        while True:
            self.print_menu()
            choice = input("\nEnter choice: ").strip()
            
            if choice == '0':
                print("\nExiting system. Goodbye!")
                break
            elif choice == '1':
                self.collect_baseline()
            elif choice == '2':
                self.view_baseline_stats()
            elif choice == '3':
                self.run_keylogger_variant(1)
            elif choice == '4':
                self.run_keylogger_variant(2)
            elif choice == '5':
                self.run_keylogger_variant(3)
            elif choice == '6':
                self.run_all_variants()
            elif choice == '7':
                self.train_detector()
            elif choice == '8':
                self.detect_from_files()
            elif choice == '9':
                self.real_time_monitoring()
            elif choice == '10':
                self.full_evaluation()
            elif choice == '11':
                self.complete_workflow()
            else:
                print("❌ Invalid choice")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    system = KeyloggerResearchSystem()
    system.run()
