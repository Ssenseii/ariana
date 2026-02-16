"""
GPU Detection Diagnostic Tool
Helps troubleshoot why GPUs (especially NVIDIA GB10/Blackwell GPUs) are not being detected.
"""

import subprocess
import sys
import os


def check_nvidia_smi():
    """Check if nvidia-smi is available and working."""
    print("=" * 60)
    print("CHECKING NVIDIA-SMI")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ nvidia-smi is available and working")
            print("\nOutput:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi returned an error")
            print("Error:", result.stderr)
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found in PATH")
        print("  This usually means NVIDIA drivers are not installed")
        return False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi command timed out")
        return False
    except Exception as e:
        print(f"✗ Error running nvidia-smi: {e}")
        return False


def check_nvidia_smi_detailed():
    """Run detailed nvidia-smi queries."""
    print("\n" + "=" * 60)
    print("DETAILED NVIDIA-SMI QUERIES")
    print("=" * 60)
    
    queries = [
        ("GPU Count", ['nvidia-smi', '--list-gpus']),
        ("GPU Details", ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version,pci.bus_id', '--format=csv']),
        ("Full Query", ['nvidia-smi', '-q']),
    ]
    
    for name, cmd in queries:
        print(f"\n[{name}]")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Failed to run: {e}")


def check_python_libraries():
    """Check if GPU-related Python libraries are installed."""
    print("\n" + "=" * 60)
    print("CHECKING PYTHON LIBRARIES")
    print("=" * 60)
    
    libraries = [
        ('pynvml', 'nvidia-ml-py3 or nvidia-ml-py'),
        ('GPUtil', 'gputil'),
        ('torch', 'pytorch'),
    ]
    
    for lib_name, package_name in libraries:
        try:
            __import__(lib_name)
            print(f"✓ {lib_name} is installed (package: {package_name})")
        except ImportError:
            print(f"✗ {lib_name} is NOT installed (install with: pip install {package_name})")


def check_pynvml_detection():
    """Try to detect GPUs using pynvml."""
    print("\n" + "=" * 60)
    print("CHECKING PYNVML GPU DETECTION")
    print("=" * 60)
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✓ pynvml detected {device_count} GPU(s)")
        
        for i in range(device_count):
            print(f"\n  GPU {i}:")
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                print(f"    Name: {name}")
                
                # Get memory info
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"    Total VRAM: {memory.total / (1024**3):.2f} GB")
                print(f"    Used VRAM: {memory.used / (1024**3):.2f} GB")
                print(f"    Free VRAM: {memory.free / (1024**3):.2f} GB")
                
                # Get UUID
                try:
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    if isinstance(uuid, bytes):
                        uuid = uuid.decode('utf-8')
                    print(f"    UUID: {uuid}")
                except:
                    pass
                
                # Get PCI info
                try:
                    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                    bus_id = pci_info.busId
                    if isinstance(bus_id, bytes):
                        bus_id = bus_id.decode('utf-8')
                    print(f"    PCI Bus ID: {bus_id}")
                except:
                    pass
                
                # Get compute capability
                try:
                    compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    print(f"    Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                except:
                    pass
                
            except Exception as e:
                print(f"    Error getting GPU {i} details: {e}")
        
        pynvml.nvmlShutdown()
        
    except ImportError:
        print("✗ pynvml is not installed")
        print("  Install with: pip install nvidia-ml-py3")
    except Exception as e:
        print(f"✗ pynvml failed: {e}")


def check_gputil_detection():
    """Try to detect GPUs using GPUtil."""
    print("\n" + "=" * 60)
    print("CHECKING GPUTIL GPU DETECTION")
    print("=" * 60)
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if gpus:
            print(f"✓ GPUtil detected {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"\n  GPU {gpu.id}:")
                print(f"    Name: {gpu.name}")
                print(f"    Total VRAM: {gpu.memoryTotal / 1024:.2f} GB")
                print(f"    Used VRAM: {gpu.memoryUsed / 1024:.2f} GB")
                print(f"    Free VRAM: {gpu.memoryFree / 1024:.2f} GB")
                print(f"    Driver: {gpu.driver}")
                if hasattr(gpu, 'uuid'):
                    print(f"    UUID: {gpu.uuid}")
        else:
            print("✗ GPUtil found no GPUs")
            
    except ImportError:
        print("✗ GPUtil is not installed")
        print("  Install with: pip install gputil")
    except Exception as e:
        print(f"✗ GPUtil failed: {e}")


def check_environment():
    """Check environment variables that might affect GPU detection."""
    print("\n" + "=" * 60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    relevant_vars = [
        'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES',
        'LD_LIBRARY_PATH',
        'PATH',
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}={value}")
        else:
            print(f"  {var}=<not set>")


def check_cuda():
    """Check CUDA installation and availability."""
    print("\n" + "=" * 60)
    print("CHECKING CUDA")
    print("=" * 60)
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ nvcc (CUDA compiler) is available")
            print(result.stdout)
        else:
            print("✗ nvcc returned an error")
    except FileNotFoundError:
        print("✗ nvcc not found (CUDA toolkit may not be installed)")
    except Exception as e:
        print(f"✗ Error checking nvcc: {e}")
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ PyTorch detects CUDA")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("\n✗ PyTorch does not detect CUDA")
    except ImportError:
        print("\nPyTorch is not installed")
    except Exception as e:
        print(f"\nError checking PyTorch CUDA: {e}")


def main():
    """Run all diagnostic checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "GPU DETECTION DIAGNOSTIC TOOL" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # Basic check
    nvidia_smi_works = check_nvidia_smi()
    
    if nvidia_smi_works:
        # Detailed nvidia-smi queries
        check_nvidia_smi_detailed()
    
    # Check Python libraries
    check_python_libraries()
    
    # Try different detection methods
    check_pynvml_detection()
    check_gputil_detection()
    
    # Check environment
    check_environment()
    
    # Check CUDA
    check_cuda()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)
    
    if not nvidia_smi_works:
        print("\n⚠ CRITICAL: nvidia-smi is not working")
        print("  Recommendations:")
        print("  1. Ensure NVIDIA drivers are properly installed")
        print("  2. Check if the GPU is properly seated in the PCIe slot")
        print("  3. Verify the GPU is enabled in BIOS/UEFI")
        print("  4. Try: sudo nvidia-smi")
        print("  5. Check system logs: dmesg | grep -i nvidia")
    else:
        print("\n✓ nvidia-smi is working")
        print("  Your NVIDIA driver is properly installed")
        print("  If GPUs are still not detected in Python:")
        print("  1. Install pynvml: pip install nvidia-ml-py3")
        print("  2. Use the improved system_analyzer_improved.py script")
        print("  3. Check CUDA_VISIBLE_DEVICES environment variable")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
