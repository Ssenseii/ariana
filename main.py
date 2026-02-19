"""
AI Capability Analyzer
Main entry point for analyzing system capabilities for running AI models locally.

Usage:
    python main.py

This will:
1. Detect your system's hardware resources (CPU, RAM, GPU, disk)
2. Fetch AI model data (from Ollama and other sources)
3. Analyze which models you can run
4. Generate a detailed report saved as 'ai_capability_report.txt'
"""

import sys
from system_analyzer import get_system_info
from model_scraper import get_model_data
from analyzer import (
    analyze_model_compatibility,
    generate_recommendations,
    can_run_now,
    can_run_after_memory_cleanup,
)
from report_generator import generate_report


def main():
    """Main function to orchestrate the AI capability analysis."""
    print("=" * 80)
    print("AI MODEL CAPABILITY ANALYZER")
    print("=" * 80)
    print()
    
    # Step 1: Detect system resources
    print("[1/4] Detecting system resources...")
    try:
        system_info = get_system_info()
        ram_gb = system_info.get('ram', {}).get('available_gb', 0)
        gpu_list = system_info.get('gpu', [])
        
        print(f"  ✓ CPU: {system_info.get('cpu', {}).get('logical_cores', 'Unknown')} cores")
        print(f"  ✓ RAM: {ram_gb} GB available")
        
        if gpu_list:
            for gpu in gpu_list:
                vram = gpu.get('vram_total_gb', 0)
                if gpu.get('unified_memory'):
                    vram_str = f" ({vram} GB Unified Memory)" if vram else ""
                else:
                    vram_str = f" ({vram} GB VRAM)" if vram else ""
                print(f"  ✓ GPU: {gpu.get('name', 'Unknown')}{vram_str}")
        else:
            print(f"  ⚠ GPU: None detected (will use CPU only)")
        
        print()
    except Exception as e:
        print(f"  ✗ Error detecting system: {e}")
        print("  Continuing with limited information...")
        system_info = {}
        print()
    
    # Step 2: Fetch model data
    print("[2/4] Fetching AI model data...")
    try:
        models = get_model_data()
        print(f"  ✓ Retrieved {len(models)} AI models")
        print()
    except Exception as e:
        print(f"  ✗ Error fetching models: {e}")
        sys.exit(1)
    
    # Step 3: Analyze compatibility
    print("[3/4] Analyzing model compatibility...")
    try:
        analyzed_models = analyze_model_compatibility(system_info, models)
        
        runnable = [m for m in analyzed_models if can_run_now(m.get('compatibility', {}))]
        runnable_after_cleanup = [
            m for m in analyzed_models
            if can_run_after_memory_cleanup(m.get('compatibility', {}))
        ]
        print(f"  ✓ Analysis complete")
        print(f"  ✓ You can run {len(runnable)} out of {len(models)} models")
        if runnable_after_cleanup:
            print(f"  ⚠ {len(runnable_after_cleanup)} additional models may run after freeing memory")
        print()
    except Exception as e:
        print(f"  ✗ Error analyzing compatibility: {e}")
        sys.exit(1)
    
    # Step 4: Generate recommendations
    print("[4/4] Generating report...")
    try:
        recommendations = generate_recommendations(system_info, analyzed_models)
        output_file = generate_report(
            system_info,
            analyzed_models,
            recommendations,
            "ai_capability_report.txt"
        )
        print(f"  ✓ Report generated: {output_file}")
        print()
    except Exception as e:
        print(f"  ✗ Error generating report: {e}")
        sys.exit(1)
    
    # Display quick summary
    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print()
    
    # Show top 3 recommended models
    top_models = [m for m in analyzed_models if can_run_now(m.get('compatibility', {}))][:3]
    if top_models:
        print("Top models you can run:")
        for i, model in enumerate(top_models, 1):
            score = model['compatibility']['score']
            print(f"  {i}. {model['name']} ({score}% compatible)")
        print()
    else:
        print("⚠ No models can run with current hardware.")
        print()
    
    # Show first recommendation
    if recommendations:
        print(recommendations[0])
        print()
    
    print(f"Full report saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
