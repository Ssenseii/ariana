"""
System Analyzer Module
Detects and reports system hardware specifications including CPU, RAM, GPU, and disk space.
"""

import psutil
import platform
import subprocess
import sys
import json
import os
import re
from typing import Dict, List, Any, Optional


def _debug_log(message: str) -> None:
    """Print debug output when AI_ANALYZER_DEBUG is enabled."""
    if os.environ.get('AI_ANALYZER_DEBUG', '').lower() in {'1', 'true', 'yes'}:
        print(f"[system_analyzer] {message}", file=sys.stderr)


def _is_apple_silicon_hardware() -> bool:
    """
    Detect Apple Silicon hardware even when Python runs under Rosetta.
    """
    if sys.platform != 'darwin':
        return False

    if platform.machine().lower() == 'arm64':
        return True

    # Rosetta case: x86_64 process on ARM hardware.
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'hw.optional.arm64'],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0 and result.stdout.strip() == '1'
    except Exception as exc:
        _debug_log(f"Unable to query Apple Silicon hardware flag: {exc}")
        return False


def _is_metal_supported(value: str) -> bool:
    """Parse system_profiler metal support values safely."""
    normalized = value.strip().lower()
    if not normalized:
        return False

    # Negative markers first to avoid false positives such as "supported".
    if 'not supported' in normalized or 'unsupported' in normalized:
        return False
    if re.search(r'\b(no|none|false|unavailable)\b', normalized):
        return False

    # Positive markers.
    if 'supported' in normalized or 'metal' in normalized:
        return True
    if normalized in {'yes', 'true'}:
        return True

    return False


def _extract_gpu_core_count(raw_value: Any) -> Optional[int]:
    """Extract integer GPU core count from system_profiler values."""
    if raw_value is None:
        return None
    match = re.search(r'\d+', str(raw_value))
    return int(match.group()) if match else None


def _looks_like_apple_silicon_name(name: str) -> bool:
    """Heuristic for Apple Silicon GPU names."""
    normalized = name.strip().lower()
    if not normalized:
        return False
    if 'apple' in normalized:
        return True
    return bool(re.match(r'^(m\d+)(\s|$)', normalized))


def _normalize_gpu_name(name: str) -> str:
    """Normalize GPU names for tolerant matching across parsers."""
    return re.sub(r'\s+', ' ', name.strip().lower())


def _parse_apple_gpu_sections_from_text(output: str) -> List[Dict[str, Any]]:
    """Parse Apple GPU sections from text system_profiler output."""
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith('Chipset Model:'):
            name = stripped.split(':', 1)[1].strip()
            current = {
                'name': name,
                'metal_support': None,
                'gpu_cores': None,
                'vendor': '',
                'is_apple': _looks_like_apple_silicon_name(name),
            }
            sections.append(current)
            continue

        if current is None:
            continue

        if (
            stripped.startswith('Metal Support:')
            or stripped.startswith('Metal Family:')
            or stripped.startswith('Metal:')
        ):
            value = stripped.split(':', 1)[1].strip() if ':' in stripped else ''
            current['metal_support'] = _is_metal_supported(value)
        elif stripped.startswith('Total Number of Cores:'):
            current['gpu_cores'] = _extract_gpu_core_count(
                stripped.split(':', 1)[1].strip() if ':' in stripped else None
            )
        elif stripped.startswith('Vendor:'):
            vendor = stripped.split(':', 1)[1].strip() if ':' in stripped else ''
            current['vendor'] = vendor
            if 'apple' in vendor.lower():
                current['is_apple'] = True

    return sections


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


def get_apple_gpu_info(total_ram_gb: Optional[float] = None) -> List[Dict[str, Any]]:
    """Get Apple Silicon GPU information (macOS Apple Silicon hardware only).

    Parses system_profiler output per-section so that non-Apple display
    adapters (DisplayLink, virtual displays) don't shadow the Apple GPU.
    """
    gpus = []
    if not _is_apple_silicon_hardware():
        return gpus

    sections: List[Dict[str, Any]] = []
    try:
        # Prefer JSON output to avoid locale- and format-sensitive parsing.
        json_result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType', '-json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if json_result.returncode == 0:
            data = json.loads(json_result.stdout or '{}')
            for entry in data.get('SPDisplaysDataType', []):
                name = entry.get('sppci_model') or entry.get('_name') or ''
                if not name:
                    continue

                vendor_value = (
                    entry.get('spdisplays_vendor')
                    or entry.get('spdisplays_vendor-id')
                    or ''
                )
                is_apple = (
                    ('apple' in str(name).lower())
                    or ('apple' in str(vendor_value).lower())
                    or _looks_like_apple_silicon_name(str(name))
                )
                metal_value = (
                    entry.get('spdisplays_metal')
                    or entry.get('spdisplays_metalfamily')
                    or entry.get('spdisplays_mtlgpufamily')
                    or ''
                )
                metal_support: Optional[bool] = None
                if str(metal_value).strip():
                    metal_support = _is_metal_supported(str(metal_value))
                sections.append({
                    'name': str(name),
                    'metal_support': metal_support,
                    'gpu_cores': _extract_gpu_core_count(
                        entry.get('sppci_cores') or entry.get('spdisplays_cores')
                    ),
                    'vendor': str(vendor_value),
                    'is_apple': is_apple,
                })
        else:
            _debug_log(
                f"system_profiler -json failed (code {json_result.returncode}); "
                "falling back to text parser."
            )
    except Exception as exc:
        _debug_log(f"Failed parsing JSON system_profiler output: {exc}")

    # Enrich incomplete JSON sections (missing metal/core details) via text output.
    if sections and any(
        section.get('metal_support') is None or section.get('gpu_cores') is None
        for section in sections
    ):
        try:
            text_result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if text_result.returncode == 0:
                text_sections = _parse_apple_gpu_sections_from_text(text_result.stdout)
                text_by_name = {
                    _normalize_gpu_name(section.get('name', '')): section
                    for section in text_sections
                }
                for section in sections:
                    matched = text_by_name.get(_normalize_gpu_name(section.get('name', '')))
                    if not matched:
                        continue
                    if section.get('metal_support') is None and matched.get('metal_support') is not None:
                        section['metal_support'] = matched.get('metal_support')
                    if section.get('gpu_cores') is None and matched.get('gpu_cores') is not None:
                        section['gpu_cores'] = matched.get('gpu_cores')
                    if not section.get('is_apple') and matched.get('is_apple'):
                        section['is_apple'] = True
                    if not section.get('vendor') and matched.get('vendor'):
                        section['vendor'] = matched.get('vendor')
            else:
                _debug_log(
                    "Could not enrich JSON GPU sections from text parser "
                    f"(code {text_result.returncode})."
                )
        except Exception as exc:
            _debug_log(f"Failed GPU text enrichment for JSON sections: {exc}")

    if not sections:
        try:
            text_result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if text_result.returncode == 0:
                sections = _parse_apple_gpu_sections_from_text(text_result.stdout)
            else:
                _debug_log(
                    f"system_profiler text mode failed with code {text_result.returncode}."
                )
        except Exception as exc:
            _debug_log(f"Failed parsing text system_profiler output: {exc}")

    if total_ram_gb is None:
        # Apple Silicon uses unified memory — GPU can access all system RAM.
        memory = psutil.virtual_memory()
        unified_memory_gb = round(memory.total / (1024**3), 2)
    else:
        unified_memory_gb = total_ram_gb

    apple_sections = [section for section in sections if section.get('is_apple')]
    if not apple_sections and sections:
        heuristic_matches = [
            section for section in sections
            if _looks_like_apple_silicon_name(str(section.get('name', '')))
        ]
        if heuristic_matches:
            _debug_log("Using Apple Silicon name heuristic for GPU section matching.")
            apple_sections = heuristic_matches
        else:
            _debug_log("No Apple GPU sections identified; skipping Apple GPU detection.")
            return gpus

    for section in apple_sections:
        metal_support = section.get('metal_support')
        if metal_support is None:
            # All Apple Silicon GPUs support Metal; this handles parser gaps.
            metal_support = True
            _debug_log(
                "Metal support value missing from system_profiler; "
                f"defaulting to True for {section.get('name', 'Unknown')}."
            )

        gpu_info: Dict[str, Any] = {
            'name': section['name'],
            'vram_total_gb': unified_memory_gb,
            'vendor': 'Apple',
            'type': 'Apple Silicon (Unified Memory)',
            'metal_support': bool(metal_support),
            'unified_memory': True,
        }
        if section['gpu_cores'] is not None:
            gpu_info['gpu_cores'] = section['gpu_cores']

        gpus.append(gpu_info)

    return gpus


def get_gpu_info(total_ram_gb: Optional[float] = None) -> List[Dict[str, Any]]:
    """Get all GPU information (NVIDIA, AMD, Apple Silicon, and integrated)."""
    all_gpus = []

    # Try to get NVIDIA GPUs
    nvidia_gpus = get_nvidia_gpu_info()
    all_gpus.extend(nvidia_gpus)

    # Try to get AMD GPUs
    amd_gpus = get_amd_gpu_info()
    all_gpus.extend(amd_gpus)

    # Try to get Apple Silicon GPUs
    apple_gpus = get_apple_gpu_info(total_ram_gb=total_ram_gb)
    all_gpus.extend(apple_gpus)

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
    ram_info = get_ram_info()
    total_ram_gb = ram_info.get('total_gb') if isinstance(ram_info.get('total_gb'), (int, float)) else None

    return {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
        },
        'cpu': get_cpu_info(),
        'ram': ram_info,
        'gpu': get_gpu_info(total_ram_gb=total_ram_gb),
        'disk': get_disk_info(),
    }


if __name__ == '__main__':
    # Test the system analyzer
    import json
    info = get_system_info()
    print(json.dumps(info, indent=2))
