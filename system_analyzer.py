"""
System Analyzer Module
Detects and reports system hardware specifications including CPU, RAM, GPU, and disk space.
"""

import psutil
import platform
import subprocess
import sys
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


def get_nvidia_gpu_info() -> List[Dict[str, Any]]:
    """Get NVIDIA GPU information using nvidia-smi."""
    gpus = []
    try:
        import GPUtil
        nvidia_gpus = GPUtil.getGPUs()
        for gpu in nvidia_gpus:
            gpus.append({
                'name': gpu.name,
                'vram_total_gb': round(gpu.memoryTotal / 1024, 2),
                'vram_used_gb': round(gpu.memoryUsed / 1024, 2),
                'vram_free_gb': round(gpu.memoryFree / 1024, 2),
                'driver': gpu.driver,
                'vendor': 'NVIDIA'
            })
    except Exception:
        # If GPUtil fails, try nvidia-smi directly
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            gpus.append({
                                'name': parts[0].strip(),
                                'vram_total_gb': round(float(parts[1].strip()) / 1024, 2),
                                'vendor': 'NVIDIA'
                            })
        except Exception:
            pass
    
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
    
    # Try to get integrated GPUs
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


if __name__ == '__main__':
    # Test the system analyzer
    import json
    info = get_system_info()
    print(json.dumps(info, indent=2))
