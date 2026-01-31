"""
Model Scraper Module
Scrapes AI model data from online sources (primarily Ollama) and provides fallback data.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re


def estimate_model_requirements(param_count: float, quantization: str = "Q4") -> Dict[str, float]:
    """
    Estimate model requirements based on parameter count and quantization.
    
    Args:
        param_count: Number of parameters in billions (e.g., 7 for 7B model)
        quantization: Quantization level (Q4, Q8, F16, etc.)
    
    Returns:
        Dictionary with estimated RAM and disk requirements in GB
    """
    # Bytes per parameter based on quantization
    bits_per_param = {
        'Q2': 2.5,
        'Q3': 3.5,
        'Q4': 4.5,
        'Q5': 5.5,
        'Q8': 8.5,
        'F16': 16,
        'F32': 32,
    }
    
    bits = bits_per_param.get(quantization.upper(), 4.5)  # Default to Q4
    
    # Calculate model size in GB
    model_size_gb = (param_count * 1e9 * bits) / (8 * 1024**3)
    
    # RAM requirement: model size + overhead (context, activations)
    # Typically need 1.2-1.5x model size for inference
    ram_required = model_size_gb * 1.3
    
    # Disk space: slightly more than model size for downloads
    disk_required = model_size_gb * 1.1
    
    # VRAM for GPU: similar to RAM but can be less with offloading
    vram_recommended = model_size_gb * 1.2
    
    return {
        'model_size_gb': round(model_size_gb, 2),
        'ram_required_gb': round(ram_required, 2),
        'vram_recommended_gb': round(vram_recommended, 2),
        'disk_required_gb': round(disk_required, 2),
    }


def parse_model_size(model_name: str) -> tuple[float, str]:
    """
    Parse model parameter count and quantization from model name.
    
    Args:
        model_name: Model name (e.g., "llama2:7b-q4", "mistral:13b")
    
    Returns:
        Tuple of (parameter_count_in_billions, quantization)
    """
    # Extract parameter count (e.g., 7b, 13b, 70b)
    param_match = re.search(r'(\d+\.?\d*)b', model_name.lower())
    param_count = float(param_match.group(1)) if param_match else 7.0
    
    # Extract quantization (e.g., q4, q8, f16)
    quant_match = re.search(r'(q\d|f\d+)', model_name.lower())
    quantization = quant_match.group(1).upper() if quant_match else 'Q4'
    
    return param_count, quantization


def scrape_ollama_models() -> List[Dict[str, Any]]:
    """
    Scrape model data from Ollama library.
    
    Returns:
        List of model dictionaries with requirements
    """
    models = []
    
    try:
        # Try to fetch from Ollama library page
        url = "https://ollama.com/library"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find model names (this may need adjustment based on actual HTML structure)
        # For now, we'll look for common patterns
        model_links = soup.find_all('a', href=re.compile(r'/library/'))
        
        for link in model_links[:50]:  # Limit to first 50 to avoid overwhelming
            model_name = link.get('href', '').split('/')[-1]
            if model_name and model_name != 'library':
                # Try to get model variants
                try:
                    variant_url = f"https://ollama.com/library/{model_name}"
                    variant_response = requests.get(variant_url, headers=headers, timeout=5)
                    
                    if variant_response.status_code == 200:
                        variant_soup = BeautifulSoup(variant_response.text, 'html.parser')
                        # Look for model tags/variants
                        tags = variant_soup.find_all(text=re.compile(r'\d+b'))
                        
                        for tag_text in tags[:5]:  # Get first few variants
                            if 'b' in tag_text.lower():
                                variant_name = f"{model_name}:{tag_text.strip()}"
                                param_count, quantization = parse_model_size(variant_name)
                                requirements = estimate_model_requirements(param_count, quantization)
                                
                                models.append({
                                    'name': variant_name,
                                    'family': model_name,
                                    'parameters_billions': param_count,
                                    'quantization': quantization,
                                    **requirements
                                })
                except Exception:
                    # If we can't get variants, add a default variant
                    pass
                
                # Add at least one default variant if nothing found
                if not any(m['family'] == model_name for m in models):
                    default_variant = f"{model_name}:7b"
                    param_count, quantization = parse_model_size(default_variant)
                    requirements = estimate_model_requirements(param_count, quantization)
                    
                    models.append({
                        'name': default_variant,
                        'family': model_name,
                        'parameters_billions': param_count,
                        'quantization': quantization,
                        **requirements
                    })
        
    except Exception as e:
        print(f"Warning: Could not scrape Ollama website: {e}")
        print("Falling back to cached model data...")
    
    return models


def get_fallback_models() -> List[Dict[str, Any]]:
    """
    Get fallback model data when scraping fails.
    
    Returns:
        List of popular AI models with their requirements
    """
    popular_models = [
        # Llama models
        ("llama2:7b", 7, "Q4"),
        ("llama2:13b", 13, "Q4"),
        ("llama2:70b", 70, "Q4"),
        ("llama3:8b", 8, "Q4"),
        ("llama3:70b", 70, "Q4"),
        
        # Mistral models
        ("mistral:7b", 7, "Q4"),
        ("mixtral:8x7b", 47, "Q4"),  # Mixture of Experts
        
        # Other popular models
        ("phi:2.7b", 2.7, "Q4"),
        ("gemma:2b", 2, "Q4"),
        ("gemma:7b", 7, "Q4"),
        ("qwen:7b", 7, "Q4"),
        ("qwen:14b", 14, "Q4"),
        ("codellama:7b", 7, "Q4"),
        ("codellama:13b", 13, "Q4"),
        ("vicuna:7b", 7, "Q4"),
        ("vicuna:13b", 13, "Q4"),
        
        # Smaller models
        ("tinyllama:1.1b", 1.1, "Q4"),
        ("orca-mini:3b", 3, "Q4"),
    ]
    
    models = []
    for name, params, quant in popular_models:
        requirements = estimate_model_requirements(params, quant)
        family = name.split(':')[0]
        
        models.append({
            'name': name,
            'family': family,
            'parameters_billions': params,
            'quantization': quant,
            **requirements
        })
    
    return models


def get_model_data() -> List[Dict[str, Any]]:
    """
    Get AI model data, trying to scrape first, falling back to cached data.
    
    Returns:
        List of model dictionaries with requirements
    """
    # Try scraping first
    models = scrape_ollama_models()
    
    # If scraping failed or returned too few models, use fallback
    if len(models) < 5:
        print("Using fallback model database...")
        models = get_fallback_models()
    else:
        print(f"Successfully retrieved {len(models)} models from Ollama")
    
    # Sort by parameter count
    models.sort(key=lambda x: x['parameters_billions'])
    
    return models


if __name__ == '__main__':
    # Test the model scraper
    models = get_model_data()
    print(f"Found {len(models)} models")
    for model in models[:5]:
        print(f"\n{model['name']}:")
        print(f"  Parameters: {model['parameters_billions']}B")
        print(f"  RAM Required: {model['ram_required_gb']} GB")
        print(f"  VRAM Recommended: {model['vram_recommended_gb']} GB")
        print(f"  Disk Space: {model['disk_required_gb']} GB")
