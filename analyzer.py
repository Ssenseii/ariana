"""
Analyzer Module
Compares system capabilities against model requirements and calculates compatibility.
"""

from typing import Dict, List, Any


def calculate_compatibility_score(
    system_ram_gb: float,
    system_vram_gb: float,
    required_ram_gb: float,
    recommended_vram_gb: float,
) -> tuple[int, str, str]:
    """
    Calculate compatibility score and determine if model can run.
    
    Args:
        system_ram_gb: Available system RAM in GB
        system_vram_gb: Available GPU VRAM in GB (0 if no GPU)
        required_ram_gb: Required RAM for model in GB
        recommended_vram_gb: Recommended VRAM for model in GB
    
    Returns:
        Tuple of (score, status, bottleneck)
        - score: 0-100 compatibility percentage
        - status: 'excellent', 'good', 'marginal', 'cannot_run'
        - bottleneck: What's limiting ('ram', 'vram', 'none')
    """
    # Check if we have enough RAM (absolute requirement)
    if system_ram_gb < required_ram_gb:
        ram_ratio = (system_ram_gb / required_ram_gb) * 100
        return (int(ram_ratio), 'cannot_run', 'ram')
    
    # We have enough RAM, now calculate performance score
    ram_ratio = system_ram_gb / required_ram_gb
    
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
    available_ram = ram_info.get('available_gb', 0)
    
    # Get GPU info
    gpu_list = system_info.get('gpu', [])
    max_vram = 0
    gpu_name = "None"
    
    if gpu_list:
        # Use the GPU with most VRAM
        for gpu in gpu_list:
            vram = gpu.get('vram_total_gb', 0) or 0
            if vram > max_vram:
                max_vram = vram
                gpu_name = gpu.get('name', 'Unknown GPU')
    
    # Analyze each model
    analyzed_models = []
    for model in models:
        required_ram = model.get('ram_required_gb', 0)
        recommended_vram = model.get('vram_recommended_gb', 0)
        disk_required = model.get('disk_required_gb', 0)
        
        # Calculate compatibility
        score, status, bottleneck = calculate_compatibility_score(
            available_ram,
            max_vram,
            required_ram,
            recommended_vram
        )
        
        # Add analysis to model
        analyzed_model = model.copy()
        analyzed_model['compatibility'] = {
            'score': score,
            'status': status,
            'bottleneck': bottleneck,
            'can_run': status != 'cannot_run',
            'gpu_used': gpu_name if max_vram > 0 else None,
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
    available_ram = ram_info.get('available_gb', 0)
    total_ram = ram_info.get('total_gb', 0)
    
    gpu_list = system_info.get('gpu', [])
    has_gpu = len(gpu_list) > 0
    max_vram = 0
    
    for gpu in gpu_list:
        vram = gpu.get('vram_total_gb', 0) or 0
        if vram > max_vram:
            max_vram = vram
    
    # Count runnable models
    runnable = [m for m in analyzed_models if m['compatibility']['can_run']]
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
    else:
        recommendations.append("✗ Your current system may struggle with most AI models.")
    
    # Specific model recommendations
    if excellent:
        top_models = [m['name'] for m in excellent[:3]]
        recommendations.append(f"✓ Recommended models to try first: {', '.join(top_models)}")
    elif runnable:
        top_models = [m['name'] for m in runnable[:3]]
        recommendations.append(f"✓ Best models for your system: {', '.join(top_models)}")
    
    # GPU recommendations
    if not has_gpu:
        recommendations.append("⚠ No dedicated GPU detected - consider adding one for 3-10x faster inference.")
        recommendations.append("  Recommended: NVIDIA RTX 3060 (12GB) or better for optimal AI performance.")
    elif max_vram < 8:
        recommendations.append(f"⚠ Limited VRAM ({max_vram}GB) - consider upgrading GPU for larger models.")
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
    if has_gpu:
        recommendations.append("  • Enable GPU acceleration in your AI framework for faster inference")
    
    return recommendations


if __name__ == '__main__':
    # Test the analyzer
    print("Analyzer module - use via main.py for full analysis")
