"""
Report Generator Module
Creates formatted text reports with system analysis and model compatibility results.
"""

from typing import Dict, List, Any
from datetime import datetime
from analyzer import can_run_now, can_run, can_run_after_memory_cleanup


def format_system_section(system_info: Dict[str, Any]) -> str:
    """Format system information section of the report."""
    lines = []
    lines.append("=" * 80)
    lines.append("SYSTEM SPECIFICATIONS")
    lines.append("=" * 80)
    lines.append("")
    
    # Platform info
    platform = system_info.get('platform', {})
    lines.append(f"Operating System: {platform.get('system', 'Unknown')} {platform.get('release', '')}")
    lines.append(f"Machine Type: {platform.get('machine', 'Unknown')}")
    lines.append("")
    
    # CPU info
    cpu = system_info.get('cpu', {})
    lines.append("CPU Information:")
    if 'model' in cpu:
        lines.append(f"  Model: {cpu['model']}")
    lines.append(f"  Physical Cores: {cpu.get('physical_cores', 'Unknown')}")
    lines.append(f"  Logical Cores: {cpu.get('logical_cores', 'Unknown')}")
    if 'max_frequency_ghz' in cpu and cpu['max_frequency_ghz']:
        lines.append(f"  Max Frequency: {cpu['max_frequency_ghz']} GHz")
    lines.append("")
    
    # RAM info
    ram = system_info.get('ram', {})
    lines.append("Memory (RAM):")
    lines.append(f"  Total: {ram.get('total_gb', 'Unknown')} GB")
    lines.append(f"  Available: {ram.get('available_gb', 'Unknown')} GB")
    lines.append(f"  Used: {ram.get('used_percent', 'Unknown')}%")
    lines.append("")
    
    # GPU info
    gpu_list = system_info.get('gpu', [])
    if gpu_list:
        lines.append("Graphics Processing Unit(s):")
        for i, gpu in enumerate(gpu_list, 1):
            lines.append(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
            if gpu.get('unified_memory') and gpu.get('vram_total_gb'):
                lines.append(f"    Unified Memory: {gpu['vram_total_gb']} GB (shared CPU/GPU)")
            elif 'vram_total_gb' in gpu and gpu['vram_total_gb']:
                lines.append(f"    VRAM: {gpu['vram_total_gb']} GB")
            if 'vendor' in gpu:
                lines.append(f"    Vendor: {gpu['vendor']}")
            if gpu.get('metal_support'):
                lines.append(f"    Metal: Supported")
            if gpu.get('gpu_cores'):
                lines.append(f"    GPU Cores: {gpu['gpu_cores']}")
            if 'driver' in gpu:
                lines.append(f"    Driver: {gpu['driver']}")
            lines.append("")
    else:
        lines.append("Graphics Processing Unit(s): None detected")
        lines.append("")
    
    # Disk info
    disk = system_info.get('disk', {})
    lines.append("Disk Space:")
    lines.append(f"  Total: {disk.get('total_gb', 'Unknown')} GB")
    lines.append(f"  Free: {disk.get('free_gb', 'Unknown')} GB")
    lines.append(f"  Used: {disk.get('used_percent', 'Unknown')}%")
    lines.append("")
    
    return "\n".join(lines)


def format_model_entry(model: Dict[str, Any], rank: int) -> List[str]:
    """Format a single model entry."""
    lines = []
    compat = model.get('compatibility', {})
    
    # Status indicator
    status = compat.get('status', 'unknown')
    score = compat.get('score', 0)
    
    status_emoji = {
        'excellent': '✓✓✓',
        'good': '✓✓',
        'marginal': '✓',
        'cannot_run': '✗'
    }
    
    indicator = status_emoji.get(status, '?')
    
    lines.append(f"{rank}. {indicator} {model.get('name', 'Unknown')} - {score}% Compatible")
    lines.append(f"   Parameters: {model.get('parameters_billions', 'Unknown')}B")
    lines.append(f"   Quantization: {model.get('quantization', 'Unknown')}")
    lines.append(f"   Required RAM: {model.get('ram_required_gb', 'Unknown')} GB")
    lines.append(f"   Recommended VRAM: {model.get('vram_recommended_gb', 'Unknown')} GB")
    lines.append(f"   Disk Space Needed: {model.get('disk_required_gb', 'Unknown')} GB")
    
    # Bottleneck info
    bottleneck = compat.get('bottleneck', 'none')
    if bottleneck == 'ram':
        lines.append("   ⚠ Bottleneck: Insufficient RAM")
    elif bottleneck == 'unified_memory':
        lines.append("   ⚠ Bottleneck: Model exceeds total unified memory")
    elif bottleneck == 'unified_memory_pressure':
        lines.append("   ⚠ Bottleneck: High unified-memory pressure (close apps for better performance)")
    elif bottleneck == 'vram':
        lines.append("   ⚠ Bottleneck: Limited VRAM (will use CPU offloading)")
    elif bottleneck == 'no_gpu':
        lines.append("   ⚠ Note: Will run on CPU only (slower)")
    
    lines.append("")
    return lines


def format_models_section(analyzed_models: List[Dict[str, Any]]) -> str:
    """Format model compatibility section of the report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COMPATIBILITY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    
    # Categorize models
    runnable = [m for m in analyzed_models if can_run_now(m.get('compatibility', {}))]
    runnable_after_cleanup = [
        m for m in analyzed_models
        if can_run_after_memory_cleanup(m.get('compatibility', {}))
    ]
    cannot_run = [m for m in analyzed_models if not can_run(m.get('compatibility', {}))]
    
    excellent = [m for m in runnable if m['compatibility']['status'] == 'excellent']
    good = [m for m in runnable if m['compatibility']['status'] == 'good']
    marginal = [m for m in runnable if m['compatibility']['status'] == 'marginal']
    
    # Summary statistics
    lines.append("SUMMARY:")
    lines.append(f"  Total Models Analyzed: {len(analyzed_models)}")
    lines.append(f"  Can Run: {len(runnable)} ({len(runnable)*100//len(analyzed_models) if analyzed_models else 0}%)")
    lines.append(f"  Can Run After Memory Cleanup: {len(runnable_after_cleanup)} ({len(runnable_after_cleanup)*100//len(analyzed_models) if analyzed_models else 0}%)")
    lines.append(f"    - Excellent Performance: {len(excellent)}")
    lines.append(f"    - Good Performance: {len(good)}")
    lines.append(f"    - Marginal Performance: {len(marginal)}")
    lines.append(f"  Cannot Run: {len(cannot_run)} ({len(cannot_run)*100//len(analyzed_models) if analyzed_models else 0}%)")
    lines.append("")
    lines.append("")
    
    # Models you CAN run
    if runnable:
        lines.append("-" * 80)
        lines.append("MODELS YOU CAN RUN")
        lines.append("-" * 80)
        lines.append("")
        
        if excellent:
            lines.append("▸ EXCELLENT PERFORMANCE (90-100% Compatible)")
            lines.append("")
            for i, model in enumerate(excellent, 1):
                lines.extend(format_model_entry(model, i))
        
        if good:
            lines.append("▸ GOOD PERFORMANCE (75-89% Compatible)")
            lines.append("")
            for i, model in enumerate(good, 1):
                lines.extend(format_model_entry(model, i))
        
        if marginal:
            lines.append("▸ MARGINAL PERFORMANCE (60-74% Compatible)")
            lines.append("")
            for i, model in enumerate(marginal, 1):
                lines.extend(format_model_entry(model, i))

    # Models that may run after memory cleanup
    if runnable_after_cleanup:
        lines.append("-" * 80)
        lines.append("MODELS REQUIRING MEMORY CLEANUP")
        lines.append("-" * 80)
        lines.append("")

        for i, model in enumerate(runnable_after_cleanup, 1):
            lines.extend(format_model_entry(model, i))
    
    # Models you CANNOT run
    if cannot_run:
        lines.append("-" * 80)
        lines.append("MODELS YOU CANNOT RUN")
        lines.append("-" * 80)
        lines.append("")
        
        for i, model in enumerate(cannot_run[:10], 1):  # Limit to 10 to avoid clutter
            lines.extend(format_model_entry(model, i))
        
        if len(cannot_run) > 10:
            lines.append(f"   ... and {len(cannot_run) - 10} more models")
            lines.append("")
    
    return "\n".join(lines)


def format_recommendations_section(recommendations: List[str]) -> str:
    """Format recommendations section of the report."""
    lines = []
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS & NEXT STEPS")
    lines.append("=" * 80)
    lines.append("")
    
    for rec in recommendations:
        lines.append(rec)
    
    lines.append("")
    return "\n".join(lines)


def generate_report(
    system_info: Dict[str, Any],
    analyzed_models: List[Dict[str, Any]],
    recommendations: List[str],
    output_file: str = "ai_capability_report.txt"
) -> str:
    """
    Generate complete AI capability report.
    
    Args:
        system_info: System information
        analyzed_models: Models with compatibility analysis
        recommendations: List of recommendations
        output_file: Path to output file
    
    Returns:
        Path to generated report file
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("AI MODEL COMPATIBILITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("")
    
    # System section
    report_lines.append(format_system_section(system_info))
    report_lines.append("")
    
    # Models section
    report_lines.append(format_models_section(analyzed_models))
    report_lines.append("")
    
    # Recommendations section
    report_lines.append(format_recommendations_section(recommendations))
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write to file
    report_content = "\n".join(report_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return output_file


if __name__ == '__main__':
    # Test the report generator
    print("Report generator module - use via main.py for full report generation")
