# AI Model Capability Analyzer

A Python tool that analyzes your system's hardware and determines which AI models you can run locally.

## Features

- **System Hardware Detection**: Automatically detects CPU, RAM, GPU, and disk space
- **AI Model Database**: Fetches data on 200+ AI models from Ollama and other sources
- **Compatibility Analysis**: Determines which models your system can run based on resource requirements
- **Detailed Reporting**: Generates a comprehensive report with recommendations
- **Support for Multiple GPUs**: Detects and accounts for NVIDIA and AMD GPUs

## Requirements

- Python 3.7+
- Windows/Linux/macOS

### Dependencies

```
psutil>=5.9.0
requests>=2.31.0
beautifulsoup4>=4.12.0
GPUtil>=1.4.0
wmi>=1.5.1 (Windows only)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ssenseii/ariana.git
cd ariana
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt

pip install nvidia-ml-py3 --break-system-packages
```

## Usage

Run the analyzer:
```bash
python main.py
```

The tool will:
1. Detect your system resources
2. Fetch available AI models
3. Analyze compatibility
4. Generate a detailed report (`ai_capability_report.txt`)

### Example Output

```
================================================================================
AI MODEL CAPABILITY ANALYZER
================================================================================

[1/4] Detecting system resources...
  ✓ CPU: 12 cores
  ✓ RAM: 31.11 GB available
  ✓ GPU: NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM)

[2/4] Fetching AI model data...
  ✓ Retrieved 217 AI models

[3/4] Analyzing model compatibility...
  ✓ Analysis complete
  ✓ You can run 158 out of 217 models

[4/4] Generating report...
  ✓ Report generated: ai_capability_report.txt
```

## How It Works

### System Analysis
The analyzer detects your system specifications:
- **CPU**: Number of cores and max frequency
- **RAM**: Total and available memory
- **GPU**: VRAM and driver information
- **Disk**: Available space

### Model Requirements
Each AI model has specific requirements:
- **Parameters**: Model size (in billions)
- **Quantization**: Compression level (Q4, Q5, F8, etc.)
- **RAM**: Minimum RAM needed
- **VRAM**: Recommended GPU VRAM
- **Disk**: Space required to download

### Compatibility Scoring
Models are scored based on:
- Available RAM vs. required RAM
- Available VRAM vs. recommended VRAM
- Available disk space
- Overall system capabilities

## Project Structure

```
ai-model-capability-analyzer/
├── main.py                    # Entry point
├── system_analyzer.py         # Hardware detection
├── model_scraper.py          # Fetches model data
├── analyzer.py               # Compatibility analysis
├── report_generator.py       # Report generation
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Supported Models

The analyzer supports 200+ models including:
- **Base Models**: Llama, Mistral, Qwen, Gemma, Phi, Falcon
- **Specialized Models**: CodeLlama, DeepSeek-Coder, StarCoder
- **Vision Models**: LLaVA, Qwen-VL, MiniCPM-V
- **Embedding Models**: nomic-embed-text, mxbai-embed-large

## Recommendations

After analyzing your system, the tool provides recommendations:
- Which models to try first
- Tips for optimizing performance
- GPU acceleration options
- Best practices for running models locally

## Tips for Running Models

1. **Start Small**: Begin with smaller models (0.5B-1B parameters)
2. **Use Quantization**: Q4/Q5 quantization significantly reduces memory usage
3. **GPU Acceleration**: Enable GPU acceleration for faster inference
4. **Use Ollama**: Simplifies model management and deployment
5. **Monitor Resources**: Watch CPU/RAM usage while running models

## Limitations

- VRAM detection may not be accurate on all systems
- Windows with WMI may have detection issues in some configurations
- GPU detection is optimized for NVIDIA and AMD GPUs
- Model requirements are estimates based on typical scenarios

## Troubleshooting

### GPU Not Detected
- Ensure your GPU drivers are up to date
- Install GPU detection libraries for your hardware

### High Memory Usage
- Use smaller models
- Apply higher quantization (Q4 instead of F16)
- Close other applications

### Inaccurate Compatibility Scores
- Results are estimates based on available specifications
- Actual performance depends on model optimization and driver versions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Model data sourced from [Ollama](https://ollama.com/)
- Inspired by the need to understand AI model compatibility with consumer hardware

## Disclaimer

This tool provides estimates based on theoretical requirements. Actual performance may vary depending on:
- System optimization
- Driver versions
- Background processes
- Model-specific optimizations

Always test models on your system before relying on them for production use.
