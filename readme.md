# SpectraBench

[![DOI](https://zenodo.org/badge/1029385925.svg)](https://doi.org/10.5281/zenodo.16735469)

**Intelligent Scheduling System for Large Language Model Benchmarking**

SpectraBench is an advanced benchmarking tool that optimizes LLM evaluation through smart scheduling. Instead of running evaluations sequentially, it uses machine learning to determine the best execution order and resource allocation, dramatically reducing evaluation time and preventing out-of-memory errors.

## ✨ Why SpectraBench?

- **🚀 34% faster evaluation** - Complete benchmarks in significantly less time
- **💾 67% fewer memory errors** - Smart resource management prevents crashes
- **📊 Better GPU utilization** - Make the most of your hardware
- **🤖 Adaptive learning** - Gets smarter with each evaluation

## 🎯 Supported Models

### Korean Language Models
- **EXAONE 3.5** (2.4B, 7.8B, 32B Instruct)
- **HyperCLOVA-X SEED** (0.5B, 1.5B Text Instruct)
- **Korean Bllossom** (3B, 8B)
- **Kanana 1.5** (2.1B base/instruct, 8B base)
- **KISTI-KONI** (8B Instruct)
- **KULLM3**
- **ETRI Eagle** (3B preview)
- **SaltLux Luxia** (21.4B alignment v1.0/v1.2, Ko-Llama3-8B)

### International Models
- **LLaMA 3.1** (8B)
- **LLaMA-DNA** (8B Instruct)
- **Gemma 3** (4B, 12B Instruct)
- **Mistral 7B** (v0.3)
- **Qwen 3** (8B)

## 📊 Supported Benchmarks

### Korean Tasks
- **KMMLU** - Korean language understanding
- **Haerae** - Korean cultural knowledge
- **KoBEST** - Korean benchmark suite
- **CSATQA** - Korean college entrance exam
- **KorMedMCQA** - Korean medical knowledge

### General Tasks
- **MMLU** - Multitask language understanding
- **BBH** - Big-Bench Hard reasoning
- **GSM8K** - Math reasoning
- **HumanEval** - Code generation
- **HellaSwag** - Commonsense reasoning
- **ARC** - Science questions
- And more...

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/gwleee/SpectraBench.git
cd SpectraBench
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with automatic scheduling
python spectrabench.py --models models.yaml --tasks tasks.yaml

# Compare with baseline
python spectrabench.py --mode comparison --models models.yaml --tasks tasks.yaml
```

### Configuration

Edit `models.yaml` and `tasks.yaml` to specify which models and benchmarks to run:

```yaml
# models.yaml example
models:
  - name: "LLaMA 3.1 8B"
    id: "meta-llama/Llama-3.1-8B"
    
# tasks.yaml example  
harness_tasks:
  - mmlu
  - gsm8k
```

## 🎮 How It Works

SpectraBench uses a **3-stage evolution approach**:

1. **🧠 Intelligent Stage** - Starts with smart rules
2. **🔄 Hybrid Stage** - Combines rules + ML predictions  
3. **🤖 Adaptive Stage** - Fully AI-driven scheduling

The system automatically learns from your evaluation history and gets better over time!

## 📈 Performance

Compared to traditional sequential evaluation:

| Metric | Improvement |
|--------|-------------|
| ⏱️ Execution Time | 34% faster |
| 💾 Memory Errors | 67% reduction |
| 🎯 GPU Utilization | 31% better |
| ✅ Success Rate | 92% vs 76% |

## 🛠️ Requirements

### Hardware
- NVIDIA GPU with 8GB+ VRAM (A100 recommended)
- 16GB+ RAM
- Multi-core CPU

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
- See `requirements.txt` for full dependencies

## 📝 Documentation

For detailed guides and examples:
- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🏛️ Citation

```bibtex
@software{spectrabench2025,
  title={SpectraBench: Adaptive Scheduling System for Efficient Large Language Model Evaluation},
  author={KISTI Large-scale AI Research Center},
  year={2025},
  doi={10.5281/zenodo.16735469},
  url={https://github.com/gwleee/SpectraBench},
  license={Apache-2.0}
}
```

## 🙋‍♂️ Contact

- **Issues**: [GitHub Issues](https://github.com/gwleee/SpectraBench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gwleee/SpectraBench/discussions)

---

*Developed with ❤️ by KISTI Large-scale AI Research Center*