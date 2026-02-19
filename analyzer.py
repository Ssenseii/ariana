"""
Analyzer Module
Compares system capabilities against model requirements and calculates compatibility.
"""

from typing import Dict, List, Any, Optional


STATUS_RANK = {
    'cannot_run': 0,
    'marginal': 1,
    'good': 2,
    'excellent': 3,
}

# Unified-memory pressure thresholds map to intentionally conservative scores:
# - >=75% of required memory currently available: likely runnable with pressure
# - >=50%: may run but with significant pressure/swap risk
# - <50%: very constrained; still potentially runnable after freeing memory
UNIFIED_PRESSURE_HIGH_THRESHOLD = 0.75
UNIFIED_PRESSURE_MEDIUM_THRESHOLD = 0.5
UNIFIED_PRESSURE_HIGH_SCORE = 70
UNIFIED_PRESSURE_MEDIUM_SCORE = 62
UNIFIED_PRESSURE_LOW_SCORE = 55


def calculate_compatibility_score(
    system_available_ram_gb: float,
    system_total_ram_gb: float,
    system_vram_gb: float,
    required_ram_gb: float,
    recommended_vram_gb: float,
    has_unified_memory: bool = False,
) -> tuple[int, str, str]:
    """
    Calculate compatibility score and determine if model can run.
    
    Args:
        system_available_ram_gb: Currently available system RAM in GB
        system_total_ram_gb: Total installed system RAM in GB
        system_vram_gb: Available GPU VRAM in GB (0 if no GPU)
        required_ram_gb: Required RAM for model in GB
        recommended_vram_gb: Recommended VRAM for model in GB
        has_unified_memory: Whether CPU and GPU share a unified memory pool
    
    Returns:
        Tuple of (score, status, bottleneck)
        - score: 0-100 compatibility percentage
        - status: 'excellent', 'good', 'marginal', 'cannot_run'
        - bottleneck: What's limiting

    Unified memory behavior:
        A unified-memory system can run models up to total installed RAM,
        even when currently available RAM is lower. We therefore use total RAM
        as the hard feasibility gate, and available RAM as a performance signal.
    """
    system_available_ram_gb = max(system_available_ram_gb, 0.0)
    system_total_ram_gb = max(system_total_ram_gb, 0.0)
    system_vram_gb = max(system_vram_gb, 0.0)
    required_ram_gb = max(required_ram_gb, 0.01)
    recommended_vram_gb = max(recommended_vram_gb, 0.01)

    # Unified memory uses total pool for hard fit, available memory for
    # near-term performance expectations.
    if has_unified_memory:
        if system_total_ram_gb < required_ram_gb:
            pool_ratio = (system_total_ram_gb / required_ram_gb) * 100
            return (int(pool_ratio), 'cannot_run', 'unified_memory')

        if system_available_ram_gb < required_ram_gb:
            pressure_ratio = system_available_ram_gb / required_ram_gb
            if pressure_ratio >= UNIFIED_PRESSURE_HIGH_THRESHOLD:
                return (UNIFIED_PRESSURE_HIGH_SCORE, 'good', 'unified_memory_pressure')
            if pressure_ratio >= UNIFIED_PRESSURE_MEDIUM_THRESHOLD:
                return (UNIFIED_PRESSURE_MEDIUM_SCORE, 'marginal', 'unified_memory_pressure')
            return (UNIFIED_PRESSURE_LOW_SCORE, 'marginal', 'unified_memory_pressure')

    # Non-unified systems require currently available RAM for feasibility.
    elif system_available_ram_gb < required_ram_gb:
        ram_ratio = (system_available_ram_gb / required_ram_gb) * 100
        return (int(ram_ratio), 'cannot_run', 'ram')

    # Once unified-memory pressure checks pass, use total pool headroom for
    # scoring. This avoids transient free-memory snapshots under-scoring
    # otherwise healthy unified-memory systems.
    if has_unified_memory:
        ram_ratio = system_total_ram_gb / required_ram_gb
    else:
        ram_ratio = system_available_ram_gb / required_ram_gb
    
    # Check GPU situation
    has_gpu = system_vram_gb > 0
    
    if has_gpu:
        vram_ratio = system_vram_gb / recommended_vram_gb
        
        # If we have enough VRAM, excellent performance expected
        if vram_ratio >= 1.0:
            if ram_ratio >= 2.0:
                return (100, 'excellent', 'none')
            elif ram_ratio >= 1.5:
                return (95, 'excellent', 'none')
            else:
                return (90, 'excellent', 'none')
        
        # Partial VRAM (can offload some layers)
        elif vram_ratio >= 0.5:
            if ram_ratio >= 1.5:
                return (80, 'good', 'vram')
            else:
                return (75, 'good', 'vram')
        
        # Very limited VRAM (mostly CPU inference)
        else:
            if ram_ratio >= 1.5:
                return (65, 'marginal', 'vram')
            else:
                return (60, 'marginal', 'vram')
    
    # No GPU - CPU only inference
    else:
        if ram_ratio >= 2.0:
            return (70, 'good', 'no_gpu')
        elif ram_ratio >= 1.5:
            return (65, 'marginal', 'no_gpu')
        else:
            return (60, 'marginal', 'no_gpu')


def can_run_now(compatibility: Dict[str, Any]) -> bool:
    """
    Determine if a model is runnable now (without first reclaiming memory).
    Falls back to historical `can_run` semantics for backward compatibility.
    """
    if 'can_run_now' in compatibility:
        return bool(compatibility.get('can_run_now'))
    return can_run(compatibility)


def can_run(compatibility: Dict[str, Any]) -> bool:
    """
    Determine if a model is runnable in principle.
    Accepts both legacy and newer compatibility payload variants.
    """
    if 'can_run' in compatibility:
        return bool(compatibility.get('can_run'))

    status = compatibility.get('status')
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {'excellent', 'good', 'marginal'}:
            return True
        if normalized == 'cannot_run':
            return False

    if 'can_run_now' in compatibility:
        return bool(compatibility.get('can_run_now'))

    return False


def can_run_after_memory_cleanup(compatibility: Dict[str, Any]) -> bool:
    """
    Determine if a model can run after reclaiming memory.
    """
    if 'can_run_after_memory_cleanup' in compatibility:
        return bool(compatibility.get('can_run_after_memory_cleanup'))
    return can_run(compatibility) and not can_run_now(compatibility)


def analyze_model_compatibility(
    system_info: Dict[str, Any],
    models: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Analyze which models are compatible with the system.
    
    Args:
        system_info: System information from system_analyzer
        models: List of models from model_scraper
    
    Returns:
        List of models with compatibility analysis added
    """
    # Extract system resources
    ram_info = system_info.get('ram', {})
    available_ram = ram_info.get('available_gb', 0) or 0
    total_ram = ram_info.get('total_gb', available_ram) or available_ram

    # Get GPU info
    gpu_list = system_info.get('gpu', [])
    compute_profiles: List[Dict[str, Any]] = []
    if gpu_list:
        for gpu in gpu_list:
            unified = bool(gpu.get('unified_memory', False))
            vram = gpu.get('vram_total_gb', 0) or 0
            if unified:
                # Unified memory GPU can address the full memory pool.
                vram = max(vram, total_ram)
            compute_profiles.append({
                'gpu_name': gpu.get('name', 'Unknown GPU'),
                'vram_gb': vram,
                'unified_memory': unified,
            })
    else:
        compute_profiles.append({
            'gpu_name': None,
            'vram_gb': 0,
            'unified_memory': False,
        })
    
    # Analyze each model
    analyzed_models = []
    for model in models:
        required_ram = model.get('ram_required_gb', 0)
        recommended_vram = model.get('vram_recommended_gb', 0)
        # Score each compute profile and keep the best one.
        best_result: Optional[Dict[str, Any]] = None
        for profile in compute_profiles:
            score, status, bottleneck = calculate_compatibility_score(
                available_ram,
                total_ram,
                profile['vram_gb'],
                required_ram,
                recommended_vram,
                has_unified_memory=profile['unified_memory'],
            )
            result = {
                'score': score,
                'status': status,
                'bottleneck': bottleneck,
                'gpu_used': profile['gpu_name'] if profile['vram_gb'] > 0 else None,
                'unified_memory': profile['unified_memory'],
            }
            if best_result is None:
                best_result = result
                continue

            candidate_key = (STATUS_RANK.get(result['status'], -1), result['score'])
            best_key = (STATUS_RANK.get(best_result['status'], -1), best_result['score'])
            if candidate_key > best_key:
                best_result = result

        assert best_result is not None

        can_run = best_result['status'] != 'cannot_run'
        can_run_now = can_run and best_result['bottleneck'] != 'unified_memory_pressure'

        # Add analysis to model
        analyzed_model = model.copy()
        analyzed_model['compatibility'] = {
            'score': best_result['score'],
            'status': best_result['status'],
            'bottleneck': best_result['bottleneck'],
            'can_run': can_run,
            'can_run_now': can_run_now,
            'can_run_after_memory_cleanup': can_run and not can_run_now,
            'gpu_used': best_result['gpu_used'],
            'unified_memory': best_result['unified_memory'],
        }
        
        analyzed_models.append(analyzed_model)
    
    # Sort by compatibility score (descending)
    analyzed_models.sort(key=lambda x: x['compatibility']['score'], reverse=True)
    
    return analyzed_models


def generate_recommendations(
    system_info: Dict[str, Any],
    analyzed_models: List[Dict[str, Any]]
) -> List[str]:
    """
    Generate recommendations based on system capabilities and model analysis.
    
    Args:
        system_info: System information
        analyzed_models: Models with compatibility analysis
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Extract system info
    ram_info = system_info.get('ram', {})
    available_ram = ram_info.get('available_gb', 0) or 0
    total_ram = ram_info.get('total_gb', 0) or 0
    
    gpu_list = system_info.get('gpu', [])
    has_gpu = len(gpu_list) > 0
    has_unified_memory = any(g.get('unified_memory', False) for g in gpu_list)
    max_vram = 0
    best_discrete_gpu_name = "Unknown GPU"
    unified_gpu_name = "Unknown GPU"

    for gpu in gpu_list:
        if gpu.get('unified_memory') and unified_gpu_name == "Unknown GPU":
            unified_gpu_name = gpu.get('name', 'Unknown GPU')
        vram = gpu.get('vram_total_gb', 0) or 0
        if not gpu.get('unified_memory') and vram > max_vram:
            max_vram = vram
            best_discrete_gpu_name = gpu.get('name', 'Unknown GPU')
    
    # Count runnable models
    runnable = [m for m in analyzed_models if can_run_now(m.get('compatibility', {}))]
    runnable_after_cleanup = [
        m for m in analyzed_models
        if can_run_after_memory_cleanup(m.get('compatibility', {}))
    ]
    excellent = [m for m in runnable if m['compatibility']['status'] == 'excellent']
    good = [m for m in runnable if m['compatibility']['status'] == 'good']
    marginal = [m for m in runnable if m['compatibility']['status'] == 'marginal']
    
    # Overall assessment
    if len(excellent) > 5:
        recommendations.append("✓ Great news! Your system can run many AI models with excellent performance.")
    elif len(runnable) > 5:
        recommendations.append("✓ Your system can run several AI models, though some may have reduced performance.")
    elif len(runnable) > 0:
        recommendations.append("✓ Your system can run smaller AI models, but larger models may struggle.")
    elif len(runnable_after_cleanup) > 0:
        recommendations.append("⚠ Several models may run after you close memory-heavy apps, but not reliably right now.")
    else:
        recommendations.append("✗ Your current system may struggle with most AI models.")
    
    # Specific model recommendations
    if excellent:
        top_models = [m['name'] for m in excellent[:3]]
        recommendations.append(f"✓ Recommended models to try first: {', '.join(top_models)}")
    elif runnable:
        top_models = [m['name'] for m in runnable[:3]]
        recommendations.append(f"✓ Best models for your system: {', '.join(top_models)}")
    elif runnable_after_cleanup:
        top_models = [m['name'] for m in runnable_after_cleanup[:3]]
        recommendations.append(f"⚠ Potentially runnable after freeing memory: {', '.join(top_models)}")
    
    # GPU recommendations
    if not has_gpu:
        recommendations.append("⚠ No dedicated GPU detected - consider adding one for 3-10x faster inference.")
        recommendations.append("  Recommended: NVIDIA RTX 3060 (12GB) or better for optimal AI performance.")
    else:
        if has_unified_memory:
            if total_ram < 16:
                recommendations.append(f"⚠ {unified_gpu_name} with {total_ram}GB unified memory — larger models need 16GB+.")
            elif total_ram >= 32:
                recommendations.append(f"✓ {unified_gpu_name} with {total_ram}GB unified memory — good capacity for larger models via Metal.")
            else:
                recommendations.append(f"✓ {unified_gpu_name} with {total_ram}GB unified memory — solid for many local models.")
            if available_ram < 8:
                recommendations.append("⚠ Low available memory right now — close other apps before loading larger models.")

        if max_vram > 0 and max_vram < 8:
            recommendations.append(f"⚠ Limited VRAM on {best_discrete_gpu_name} ({max_vram}GB) - consider upgrading GPU for larger models.")
            recommendations.append("  Recommended: GPU with 12GB+ VRAM for better model support.")
    
    # RAM recommendations
    if total_ram < 16:
        recommendations.append(f"⚠ Limited RAM ({total_ram}GB) - consider upgrading to 16GB+ for better performance.")
    elif available_ram < 8:
        recommendations.append("⚠ Low available RAM - close other applications before running AI models.")
    
    # Optimization tips
    recommendations.append("\n💡 Optimization Tips:")
    recommendations.append("  • Use quantized models (Q4, Q5) for better performance on limited hardware")
    recommendations.append("  • Start with smaller models (2B-7B parameters) to test your setup")
    recommendations.append("  • Use Ollama (https://ollama.com) for easy model management")
    if has_unified_memory:
        recommendations.append("  • Use MLX or llama.cpp with Metal backend for best Apple Silicon performance")
    elif has_gpu:
        recommendations.append("  • Enable GPU acceleration in your AI framework for faster inference")
    
    return recommendations


if __name__ == '__main__':
    # Test the analyzer
    print("Analyzer module - use via main.py for full analysis")
