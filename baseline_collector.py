"""
Cross-Platform Baseline Behavior Collector
Works on Windows, macOS, and Linux
"""

import time
import psutil
import json
import threading
import platform
from datetime import datetime
from collections import defaultdict
import os
from pynput import keyboard

import os

# Add display debugging
if platform.system() == "Linux":
    DISPLAY = os.environ.get('DISPLAY', 'Not set')
    if DISPLAY == 'Not set':
        print("[WARNING] No X11 display detected!")
        print("[TIP] Run with: xvfb-run -a python master_system.py")

class CrossPlatformBaselineCollector:
    def __init__(self, output_file="baseline_data.json"):
        self.output_file = output_file
        self.platform = platform.system()
        self.keystroke_count = 0  # ADD THIS LINE - Initialize instance variable
        self.data = {
            "session_id": datetime.now().isoformat(),
            "platform": self.platform,
            "metrics": [],
            "keystroke_count": 0,
            "start_time": None,
            "end_time": None
        }
        self.is_collecting = False
        self.listener = None
        self.monitor_thread = None
        
        print(f"Running on: {self.platform}")
    
    def on_press(self, key):
        """Callback for keystroke events"""
        if self.is_collecting:
            self.keystroke_count += 1
            self.data["keystroke_count"] = self.keystroke_count  # SYNC WITH DATA
    
    def collect_system_metrics(self):
        """Collect system-level behavioral metrics (cross-platform)"""
        try:
            current_process = psutil.Process()
            
            # Get current process information
            metrics = {
                "timestamp": time.time(),
                "cpu_percent": current_process.cpu_percent(interval=0.1),
                "memory_mb": current_process.memory_info().rss / (1024 * 1024),
                "threads": current_process.num_threads(),  # CHANGED from num_threads
                "handles": self._get_windows_handles(current_process),  # CHANGED from num_handles
                "keystroke_count": self.keystroke_count  # USE INSTANCE VARIABLE
            }
            
            # Platform-specific metrics
            if self.platform == "Windows":
                metrics["handles"] = self._get_windows_handles(current_process)
            
            # Get system-wide metrics
            metrics["system_cpu"] = psutil.cpu_percent(interval=0.1)
            metrics["system_memory"] = psutil.virtual_memory().percent
            
            # Count suspicious processes
            metrics["suspicious_processes"] = self._count_suspicious_processes()
            
            return metrics
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return None
    
    def _get_io_counters(self, process):
        """Get I/O counters for process - cross-platform"""
        try:
            io_counters = process.io_counters()
            return {
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count
            }
        except:
            return {'read_count': 0, 'write_count': 0}
    
    def _get_open_files_count(self, process):
        """Safely get open files count"""
        try:
            return len(process.open_files())
        except:
            return 0
    
    def _get_windows_handles(self, process):
        """Get handle count on Windows"""
        try:
            if hasattr(process, 'num_handles'):
                return process.num_handles()
            return 0
        except:
            return 0
    
    def _count_suspicious_processes(self):
        """Count suspicious processes"""
        try:
            # Use net_connections instead of deprecated connections
            suspicious = 0
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if hasattr(proc, 'net_connections'):
                        conns = proc.net_connections()
                        if len(conns) > 10:  # Many connections = suspicious
                            suspicious += 1
                except:
                    pass
            return suspicious
        except:
            return 0
    
    def monitor_loop(self):
        """Continuously monitor system metrics"""
        while self.is_collecting:
            metrics = self.collect_system_metrics()
            if metrics:
                self.data["metrics"].append(metrics)
            time.sleep(1)  # Collect metrics every second
    
    def start_collection(self, duration_minutes=5):
        """Start collecting baseline data"""
        print(f"\n{'='*60}")
        print("CROSS-PLATFORM BASELINE COLLECTOR")
        print(f"Platform: {self.platform}")
        print(f"{'='*60}")
        print(f"\nStarting data collection for {duration_minutes} minutes...")
        print("Please perform normal typing activities:")
        print("  - Write documents")
        print("  - Browse websites")
        print("  - Fill forms")
        print("  - Send emails")
        print("\nPress Ctrl+C to stop early\n")
        
        self.is_collecting = True
        self.data["start_time"] = datetime.now().isoformat()
        
        # Start keyboard listener
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
        except Exception as e:
            print(f"Warning: Could not start keyboard listener: {e}")
            print("Continuing without keystroke counting...")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.start()
        
        try:
            # Run for specified duration
            time.sleep(duration_minutes * 60)
        except KeyboardInterrupt:
            print("\n\nCollection stopped by user.")
        finally:
            self.stop_collection()
    
    def stop_collection(self):
        """Stop collecting data and save results"""
        self.is_collecting = False
        self.data["end_time"] = datetime.now().isoformat()
        
        if self.listener:
            try:
                self.listener.stop()
            except:
                pass
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.save_data()
        self.print_summary()
    
    def save_data(self):
        """Save collected data to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"\nData saved to: {self.output_file}")
    
    def print_summary(self):
        """Print summary of collected data"""
        print(f"\n{'='*60}")
        print("COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Platform: {self.data['platform']}")
        print(f"Total keystrokes: {self.data['keystroke_count']}")
        print(f"Total metrics collected: {len(self.data['metrics'])}")
        print(f"Duration: {self.data['start_time']} to {self.data['end_time']}")
        
        if self.data['metrics']:
            # Calculate statistics
            cpu_values = [m['cpu_percent'] for m in self.data['metrics']]
            mem_values = [m['memory_mb'] for m in self.data['metrics']]
            thread_values = [m['threads'] for m in self.data['metrics']]  # CHANGED from 'num_threads'
            
            print(f"\nAverage CPU usage: {sum(cpu_values)/len(cpu_values):.2f}%")
            print(f"Average Memory usage: {sum(mem_values)/len(mem_values):.2f} MB")
            print(f"Average threads: {sum(thread_values)/len(thread_values):.1f}")
            
            if self.platform == "Windows":
                handle_values = [m.get('handles', 0) for m in self.data['metrics']]  # CHANGED from 'num_handles'
                print(f"Average handles: {sum(handle_values)/len(handle_values):.1f}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    collector = CrossPlatformBaselineCollector("baseline_data.json")
    
    # Collect for 5 minutes (adjust as needed)
    # For quick testing, use duration_minutes=1
    collector.start_collection(duration_minutes=5)