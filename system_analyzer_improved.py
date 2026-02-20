"""
System Analyzer Module - Enhanced Version
Detects and reports system hardware specifications including CPU, RAM, GPU, and disk space.
Enhanced GPU detection for NVIDIA GPUs including DGX systems with GB10 (Blackwell) GPUs.
"""

import psutil
import platform
import subprocess
import sys
import re
from typing import Dict, List, Any, Optional


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information including cores, frequency, and model name."""
    try:
        cpu_freq = psutil.cpu_freq()
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency_ghz': round(cpu_freq.max / 1000, 2) if cpu_freq else None,
            'current_frequency_ghz': round(cpu_freq.current / 1000, 2) if cpu_freq else None,
            'model': platform.processor() or "Unknown",
        }
    except Exception as e:
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'error': str(e)
        }


def get_ram_info() -> Dict[str, Any]:
    """Get RAM information including total, available, and used memory."""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'used_percent': round(memory.percent, 1),
        }
    except Exception as e:
        return {'error': str(e)}


def get_nvidia_gpu_info_via_smi() -> List[Dict[str, Any]]:
    """
    Get NVIDIA GPU information using nvidia-smi with comprehensive query.
    This method is more reliable for detecting all NVIDIA GPUs including newer models.
    """
    gpus = []
    try:
        # Enhanced query to get more detailed information
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pci.bus_id,compute_cap',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        try:
                            gpu_info = {
                                'index': int(parts[0]) if parts[0].isdigit() else None,
                                'name': parts[1],
                                'vram_total_gb': round(float(parts[2]) / 1024, 2) if parts[2] else None,
                                'vram_used_gb': round(float(parts[3]) / 1024, 2) if len(parts) > 3 and parts[3] else None,
                                'vram_free_gb': round(float(parts[4]) / 1024, 2) if len(parts) > 4 and parts[4] else None,
                                'driver': parts[5] if len(parts) > 5 else 'Unknown',
                                'pci_bus_id': parts[6] if len(parts) > 6 else 'Unknown',
                                'compute_capability': parts[7] if len(parts) > 7 else 'Unknown',
                                'vendor': 'NVIDIA'
                            }
                            gpus.append(gpu_info)
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Failed to parse GPU line: {line}, Error: {e}", file=sys.stderr)
                            continue
    except FileNotFoundError:
        print("Warning: nvidia-smi not found. NVIDIA drivers may not be installed.", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("Warning: nvidia-smi command timed out.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error running nvidia-smi: {e}", file=sys.stderr)
    
    return gpus


def get_nvidia_gpu_info_via_gputil() -> List[Dict[str, Any]]:
    """Get NVIDIA GPU information using GPUtil library."""
    gpus = []
    try:
        import GPUtil
        nvidia_gpus = GPUtil.getGPUs()
        for gpu in nvidia_gpus:
            gpus.append({
                'index': gpu.id,
                'name': gpu.name,
                'vram_total_gb': round(gpu.memoryTotal / 1024, 2),
                'vram_used_gb': round(gpu.memoryUsed / 1024, 2),
                'vram_free_gb': round(gpu.memoryFree / 1024, 2),
                'driver': gpu.driver,
                'uuid': gpu.uuid if hasattr(gpu, 'uuid') else 'Unknown',
                'vendor': 'NVIDIA'
            })
    except ImportError:
        pass  # GPUtil not installed
    except Exception as e:
        print(f"Warning: GPUtil failed: {e}", file=sys.stderr)
    
    return gpus


def get_nvidia_gpu_info_via_pynvml() -> List[Dict[str, Any]]:
    """Get NVIDIA GPU information using pynvml (NVML) library."""
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            
            # Decode if bytes
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_info = {
                'index': i,
                'name': name,
                'vram_total_gb': round(memory_info.total / (1024**3), 2),
                'vram_used_gb': round(memory_info.used / (1024**3), 2),
                'vram_free_gb': round(memory_info.free / (1024**3), 2),
                'vendor': 'NVIDIA'
            }
            
            # Try to get additional info
            try:
                gpu_info['uuid'] = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(gpu_info['uuid'], bytes):
                    gpu_info['uuid'] = gpu_info['uuid'].decode('utf-8')
            except:
                pass
            
            try:
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                gpu_info['pci_bus_id'] = pci_info.busId
                if isinstance(gpu_info['pci_bus_id'], bytes):
                    gpu_info['pci_bus_id'] = gpu_info['pci_bus_id'].decode('utf-8')
            except:
                pass
            
            try:
                compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                gpu_info['compute_capability'] = f"{compute_cap[0]}.{compute_cap[1]}"
            except:
                pass
            
            gpus.append(gpu_info)
        
        pynvml.nvmlShutdown()
    except ImportError:
        pass  # pynvml not installed
    except Exception as e:
        print(f"Warning: pynvml failed: {e}", file=sys.stderr)
    
    return gpus


def get_nvidia_gpu_info() -> List[Dict[str, Any]]:
    """
    Get NVIDIA GPU information using multiple methods.
    Tries pynvml first (most reliable), then nvidia-smi, then GPUtil.
    """
    # Try pynvml first (most comprehensive and reliable)
    gpus = get_nvidia_gpu_info_via_pynvml()
    if gpus:
        return gpus
    
    # Try nvidia-smi (works without Python bindings)
    gpus = get_nvidia_gpu_info_via_smi()
    if gpus:
        return gpus
    
    # Fall back to GPUtil
    gpus = get_nvidia_gpu_info_via_gputil()
    return gpus


def get_amd_gpu_info() -> List[Dict[str, Any]]:
    """Get AMD GPU information (Windows only using WMI)."""
    gpus = []
    if sys.platform != 'win32':
        return gpus
    
    try:
        import wmi
        w = wmi.WMI()
        for gpu in w.Win32_VideoController():
            # Check if it's an AMD GPU
            if gpu.Name and ('AMD' in gpu.Name.upper() or 'RADEON' in gpu.Name.upper()):
                vram_bytes = gpu.AdapterRAM if gpu.AdapterRAM else 0
                gpus.append({
                    'name': gpu.Name,
                    'vram_total_gb': round(vram_bytes / (1024**3), 2) if vram_bytes > 0 else None,
                    'vendor': 'AMD',
                    'driver': gpu.DriverVersion if gpu.DriverVersion else 'Unknown'
                })
    except Exception:
        pass
    
    return gpus


def get_integrated_gpu_info() -> List[Dict[str, Any]]:
    """Get integrated GPU information (Windows only using WMI)."""
    gpus = []
    if sys.platform != 'win32':
        return gpus
    
    try:
        import wmi
        w = wmi.WMI()
        for gpu in w.Win32_VideoController():
            if gpu.Name:
                name_upper = gpu.Name.upper()
                # Check for integrated graphics (Intel, AMD APU, etc.)
                if any(keyword in name_upper for keyword in ['INTEL', 'UHD', 'IRIS', 'VEGA']) and \
                   'NVIDIA' not in name_upper and \
                   'RADEON RX' not in name_upper:
                    vram_bytes = gpu.AdapterRAM if gpu.AdapterRAM else 0
                    gpus.append({
                        'name': gpu.Name,
                        'vram_total_gb': round(vram_bytes / (1024**3), 2) if vram_bytes > 0 else None,
                        'vendor': 'Integrated',
                        'type': 'Integrated Graphics'
                    })
    except Exception:
        pass
    
    return gpus


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get all GPU information (NVIDIA, AMD, and integrated)."""
    all_gpus = []
    
    # Try to get NVIDIA GPUs
    nvidia_gpus = get_nvidia_gpu_info()
    all_gpus.extend(nvidia_gpus)
    
    # Try to get AMD GPUs
    amd_gpus = get_amd_gpu_info()
    all_gpus.extend(amd_gpus)
    
    # Try to get integrated GPUs (only if no discrete GPUs found)
    if not all_gpus:
        integrated_gpus = get_integrated_gpu_info()
        all_gpus.extend(integrated_gpus)
    
    return all_gpus


def get_disk_info() -> Dict[str, Any]:
    """Get disk space information for the main system drive."""
    try:
        # Get the root partition (where Python is running)
        disk = psutil.disk_usage('/')
        return {
            'total_gb': round(disk.total / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'used_percent': round(disk.percent, 1),
        }
    except Exception as e:
        return {'error': str(e)}


def get_system_info() -> Dict[str, Any]:
    """Get complete system information."""
    return {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
        },
        'cpu': get_cpu_info(),
        'ram': get_ram_info(),
        'gpu': get_gpu_info(),
        'disk': get_disk_info(),
    }


def print_system_info():
    """Print system information in a readable format."""
    info = get_system_info()
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Platform
    print("\n[PLATFORM]")
    for key, value in info['platform'].items():
        print(f"  {key}: {value}")
    
    # CPU
    print("\n[CPU]")
    for key, value in info['cpu'].items():
        print(f"  {key}: {value}")
    
    # RAM
    print("\n[RAM]")
    for key, value in info['ram'].items():
        print(f"  {key}: {value}")
    
    # GPU
    print("\n[GPU]")
    if info['gpu']:
        for i, gpu in enumerate(info['gpu']):
            print(f"  GPU {i}:")
            for key, value in gpu.items():
                print(f"    {key}: {value}")
    else:
        print("  No GPUs detected")
    
    # Disk
    print("\n[DISK]")
    for key, value in info['disk'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    import json
    
    # Check if we should print in JSON format
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        info = get_system_info()
        print(json.dumps(info, indent=2))
    else:
        print_system_info()
