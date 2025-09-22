[![DOI](https://zenodo.org/badge/1029385925.svg)](https://doi.org/10.5281/zenodo.16741725)

**Intelligent Scheduling System for Large Language Model Benchmarking**

SpectraBench is an advanced benchmarking tool developed by the **AI Platform Team** at **KISTI Large-scale AI Research Center** ([Korea Institute of Science and Technology Information](https://www.kisti.re.kr/)). The Large-scale AI Research Center was officially launched in March 2024, building upon KISTI's generative large language model 'KONI (KISTI Open Natural Intelligence)' unveiled in December 2023. The **AI Platform Team is responsible for developing AI model and Agent service technologies**.

SpectraBench optimizes LLM evaluation through smart scheduling, using machine learning to determine the best execution order and resource allocation, dramatically reducing evaluation time and preventing out-of-memory errors.

🔗 **Related Projects**: 
- [KISTI-KONI Models](https://huggingface.co/KISTI-KONI) - KISTI Large Language Models
- [KISTI-MCP](https://github.com/ansua79/kisti-mcp) - KISTI Model Context Protocol server

## ✨ Why SpectraBench?

- **🚀 Intelligent scheduling** - Optimal execution order through machine learning
- **💾 Smart resource management** - Prevents memory errors with adaptive allocation
- **📊 Maximized efficiency** - Better hardware utilization than sequential evaluation
- **🤖 Adaptive learning** - Continuously improves with usage
- **🔧 Flexible configuration** - Easy switching between test and production setups
- **📈 3-stage evolution** - Intelligent → Hybrid → Adaptive progression

## 🎯 Supported Models (22+ Models)

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

*All models from 0.5B to 32B parameters are fully supported with automatic size detection and resource optimization.*

## 📊 Supported Benchmarks (21+ Tasks)

### Korean Tasks
- **KMMLU** / **KMMLU Hard** - Korean language understanding
- **Haerae** - Korean cultural knowledge
- **KoBEST** - Korean benchmark suite
- **CSATQA** - Korean college entrance exam
- **KorMedMCQA** - Korean medical knowledge
- **Belebele** - Multilingual reading comprehension

### Reasoning & Knowledge
- **MMLU** / **MMLU Pro** - Multitask language understanding
- **BBH** - Big-Bench Hard reasoning
- **GPQA** - Graduate-level science questions
- **AGIEval** - Human-centric benchmark

### Math & Science
- **GSM8K** - Math reasoning
- **HRM8K** - Advanced math problems
- **ARC Challenge/Easy** - Science questions

### Language Understanding
- **HellaSwag** - Commonsense reasoning
- **PIQA** - Physical interaction reasoning
- **TriviaQA** - Factual knowledge
- **Natural Questions Open** - Open-domain QA

### Code Generation
- **HumanEval** - Python code generation

*Total: 462 possible model-task combinations (22 models × 21 tasks)*

## 🚀 Quick Start

### Automated Installation & Execution

For users who want to quickly run experiments:

```bash
# 1. Clone repository
git clone https://github.com/gwleee/SpectraBench.git
cd SpectraBench

# 2. Automated installation (recommended)
chmod +x install.sh
./install.sh

# 3. Run experiments
chmod +x run_experiments.sh

# Quick test (15-30 minutes)
./run_experiments.sh --quick

# Full comparison (4-8 hours)
./run_experiments.sh --comparison
```

### Manual Installation

If you prefer manual setup:

```bash
git clone https://github.com/gwleee/SpectraBench.git
cd SpectraBench

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables (optional)
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

#### Interactive Mode (Standard Approach)
```bash
# Activate environment
source venv/bin/activate

# Run interactive benchmark
python entrance.py

# Choose execution mode:
# 1. Baseline mode (Traditional sequential)
# 2. Optimized mode (Intelligent scheduling) 
# 3. Comparison mode (Both modes for analysis)

# Choose dataset size:
# - Test mode (limit=2): Quick validation for new models/tasks
# - Full mode (limit=None): Complete benchmark evaluation
```

#### Automated Execution Scripts

The repository includes automated scripts for easy experiment execution:

| Script | Purpose | Duration | Best For |
|--------|---------|----------|----------|
| `install.sh` | Complete environment setup | 5-15 min | First-time setup |
| `run_experiments.sh` | Basic test run | 15-30 min | Quick validation |
| `run_experiments.sh --quick` | Fast test (limit=2) | 15-30 min | Development/testing |
| `run_experiments.sh --comparison` | Full comparison | 4-8 hours | Comprehensive evaluation |

#### Dataset Limit Configuration

For faster testing and validation, SpectraBench supports configurable data limits:

```python
# In evaluation_lm.py or entrance.py
FULL_RUN = False  # Uses limit=2 for quick testing
FULL_RUN = True   # Uses full dataset

# You can modify the limit value in the code:
limit = 2    # Quick test (default for development)
limit = 10   # Small validation 
limit = 50   # Medium test
limit = None # Full dataset (production)
```

**When to use different limits:**
- **limit=2**: New model testing, debugging, proof-of-concept (minutes)
- **limit=10-50**: Development validation, parameter tuning (hours)  
- **limit=None**: Production evaluation, research papers (full runtime)

#### Advanced Usage
```bash
# Use extended configuration (rename files first)
mv "code/config/models copy.yaml" code/config/models.yaml
mv "code/config/tasks copy.yaml" code/config/tasks.yaml
python entrance.py

# Run experimental validation
python code/experiments/phase1_validation_experiment.py  # Limit optimization
python code/experiments/phase2_threshold_optimization.py  # Threshold tuning

# Generate performance comparison report
python code/experiments/comparison_analyzer.py
```

### Configuration

SpectraBench provides two configuration sets for different use cases:

#### Standard Configuration (Faster Testing)
- **`models.yaml`** - 13 carefully selected models for efficient testing
- **`tasks.yaml`** - 10 core benchmarks for quick evaluation
- **Total combinations**: 130 (13 × 10)
- **Dataset limit**: Configurable (default limit=2 for testing)
- **Recommended for**: Development, testing, proof-of-concept, new model validation

#### Extended Configuration (Comprehensive Evaluation)
- **`models copy.yaml`** - 22 models covering the full Korean AI ecosystem
- **`tasks copy.yaml`** - 21 benchmarks for thorough assessment
- **Total combinations**: 462 (22 × 21)
- **Dataset limit**: Full dataset (limit=None) or configurable for testing
- **Recommended for**: Research, production evaluation, comprehensive comparison

```yaml
# Standard models.yaml (13 models)
models:
  - name: "LLaMA 3.1 8B"
    id: "meta-llama/Llama-3.1-8B"
  - name: "EXAONE-3.5-32B-Instruct"
    id: "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
  # ... 11 more models

# Extended models copy.yaml (22 models)  
models:
  - name: "LLaMA 3.1 8B"
    id: "meta-llama/Llama-3.1-8B"
  - name: "EXAONE-3.5-7.8B-Instruct"    # Additional model
    id: "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  - name: "KISTI-KONI"                  # Additional model
    id: "KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024"
  # ... 20 more models including Bllossom, Kanana variants, KULLM3, etc.
```

```yaml
# Standard tasks.yaml (10 tasks)
harness_tasks:
  - kmmlu
  - mmlu
  - arc_challenge
  - hellaswag
  # ... 6 more core tasks

# Extended tasks copy.yaml (21 tasks)
harness_tasks:
  - kmmlu
  - kmmlu_hard                         # Additional task
  - belebele                           # Additional task
  - mmlu_pro                           # Additional task
  - triviaqa                           # Additional task
  - nq_open                            # Additional task
  - agieval                            # Additional task
  # ... 14 more comprehensive tasks
```

#### Switching Between Configurations

To use the extended configuration, simply rename the files:

```bash
# Switch to comprehensive evaluation
mv code/config/models.yaml code/config/models_standard.yaml
mv code/config/tasks.yaml code/config/tasks_standard.yaml
mv code/config/models\ copy.yaml code/config/models.yaml
mv code/config/tasks\ copy.yaml code/config/tasks.yaml

# Switch back to standard configuration
mv code/config/models.yaml code/config/models_extended.yaml
mv code/config/tasks.yaml code/config/tasks_extended.yaml
mv code/config/models_standard.yaml code/config/models.yaml
mv code/config/tasks_standard.yaml code/config/tasks.yaml
```

### Advanced Configuration

The system uses `code/config/scheduler_config.yaml` for fine-tuning:

```yaml
# Dynamic threshold management
stage_transitions:
  min_learning_data: 50    # Intelligent → Hybrid
  stable_learning_data: 200  # Hybrid → Adaptive
  
# Domain-specific optimization
domain_thresholds:
  large_models:     # >8B parameters
    min_learning_data: 75
    stable_learning_data: 300
```

## 🎮 How It Works

SpectraBench uses a **3-stage evolution approach** with ConfigManager integration:

1. **🧠 Intelligent Stage** - Rule-based scheduling with smart heuristics
2. **🔄 Hybrid Stage** - ML predictions + rule-based decisions (weighted combination)
3. **🤖 Adaptive Stage** - Fully ML-driven with Random Forest scheduling

### Key Features:
- **Dynamic model size detection** - Automatically handles 0.5B to 32B models
- **Memory-aware scheduling** - Prevents OOM with intelligent batch sizing
- **Task-specific optimization** - Zero-shot detection for appropriate tasks
- **Real-time resource monitoring** - GPU memory, utilization tracking
- **Performance learning** - Builds prediction models from execution history

## ⚡ Configuration Comparison

### Standard Configuration (130 combinations)
- **Purpose**: Quick validation, development, testing
- **Models**: 13 carefully selected representative models
- **Tasks**: 10 core benchmarks
- **Best for**: Proof-of-concept, algorithm testing, development cycles

### Extended Configuration (462 combinations)  
- **Purpose**: Comprehensive research, production evaluation
- **Models**: 22 models covering the full Korean AI ecosystem
- **Tasks**: 21 benchmarks for thorough assessment
- **Best for**: Research papers, production deployment, comprehensive comparison

The intelligent scheduling system automatically optimizes both configurations for maximum efficiency.

## 📈 Key Benefits

SpectraBench's intelligent scheduling provides significant improvements over traditional sequential evaluation:

| Benefit | Description |
|---------|-------------|
| ⏱️ **Faster Execution** | Optimized task ordering reduces total time |
| 💾 **Memory Efficiency** | Smart resource management prevents crashes |
| 🎯 **Better Utilization** | Maximizes GPU usage through intelligent scheduling |
| ✅ **Higher Success Rate** | Adaptive batch sizing reduces failure rates |
| 🧠 **Learning System** | Gets smarter with each evaluation run |

## 🔬 Research Pipeline

SpectraBench development follows a systematic 3-phase research approach:

### Phase 1: Limit Convergence Analysis (Completed)
```bash
python code/experiments/phase1_validation_experiment.py
```
**Purpose**: Determines optimal data limits for accuracy convergence analysis.
**Results**: Integrated into system configurations for efficient testing.

### Phase 2: Threshold Optimization (Completed)  
```bash
python code/experiments/phase2_threshold_optimization.py
```
**Purpose**: Optimizes stage transition thresholds based on Phase 1 results.
**Results**: Applied to `code/config/scheduler_config.yaml` for intelligent scheduling.

### Main Experiments (Ready to Use)
```bash
./run_experiments.sh --comparison
```
**Purpose**: Executes comprehensive benchmarking using optimized configurations.
**Results**: Generates performance comparisons and evaluation metrics.

*Note: Phase 1 and Phase 2 optimizations have been completed and their results are integrated into the system. Users can directly run main experiments.*

## 🔬 Additional Analysis Tools

### Performance Comparison
```bash
python code/experiments/comparison_analyzer.py
```
Detailed baseline vs optimized analysis with visualizations.

### Advanced Threshold Testing (Experimental)
```bash
python code/experiments/threshold_optimization.py
```
Comprehensive threshold combination testing with simulation.

## 🛠️ System Requirements

This system has been successfully deployed and tested on the following environment:

### Hardware Environment
- **GPU**: 2× NVIDIA A100 (80GB VRAM)
- **RAM**: 444GB 
- **CPU**: Intel Xeon Processor (Icelake, 48 cores)
- **Storage**: 1.9TB ext4 for fast model loading and execution data storage

### Software Environment
- **OS**: Ubuntu 22.04.5 LTS (Kernel 5.15.0-139-generic)
- **Python**: 3.10.16
- **PyTorch**: 2.7.1
- **CUDA**: 12.8
- **Transformers**: 4.51.3

### ⚠️ Important Notes for Different Hardware

**For smaller GPU configurations:**
- **8-16GB VRAM**: Use small models only (0.5B-3B parameters)
- **24GB VRAM**: Support up to 8B models 
- **40GB+ VRAM**: Support most models, but large models (21.4B, 32B) may still cause OOM

**Recommendations:**
- Start with standard configuration (13 models) before trying extended (22 models)
- Monitor GPU memory usage and adjust model selection accordingly
- Large models (>20B parameters) require substantial VRAM and may need single-GPU execution

See `requirements.txt` for complete dependency list.

## 🚀 Quick Start Guide

To quickly run experiments and validate results:

### Quick Validation (15-30 minutes)
```bash
git clone https://github.com/gwleee/SpectraBench.git
cd SpectraBench
./install.sh
./run_experiments.sh --quick
```

### Full Evaluation (4-8 hours)
```bash
./run_experiments.sh --comparison
```

The automated scripts handle all configuration and provide detailed logs for analysis. Results are saved in `experiments_results/` with performance tracking in `data/performanceDB/performance_history.db`.

## 📄 License

Apache License 2.0 - see [LICENSE](license.md) for details.

## 🏛️ Citation

```bibtex
@software{spectrabench2025,
  title={SpectraBench:  A Three-Stage Evolution Framework for Intelligent Large Language Model Evaluation},
  author={KISTI Large-scale AI Research Center},
  year={2025},
  doi={10.5281/zenodo.16865650},
  url={https://github.com/gwleee/SpectraBench},
  license={Apache-2.0},
  note={Supports 22+ models and 21+ benchmarks with intelligent scheduling}
}
```

---

*Developed with ❤️ by Gunwoo Lee from the AI Platform Team (Leader: Ryong Lee) at KISTI Large-scale AI Research Center (Director: Kyong-Ha Lee)*

*Supporting the Korean AI ecosystem with intelligent benchmarking tools through automated Docker multi-version evaluation*
