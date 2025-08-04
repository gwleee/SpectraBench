# Dependency installation
install_dependencies() {
    log_header "Installing required dependencies..."
    
    # Check for requirements file
    if [ -f "requirements.txt" ]; then
        log_info "Installing packages from requirements.txt..."
        pip install -r requirements.txt
        log_success "Dependencies installed from requirements.txt"
    else
        log_warning "requirements.txt not found. Installing essential packages..."
        
        # Core ML/DL frameworks
        log_info "Installing PyTorch ecosystem..."
        pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        
        log_info "Installing Transformers and related packages..."
        pip install transformers>=4.30.0
        pip install tokenizers>=0.13.0
        pip install accelerate>=0.20.0
        pip install datasets>=2.12.0
        
        # Evaluation framework
        log_info "Installing evaluation framework..."
        pip install lm-eval>=0.4.0
        
        # Scientific computing
        log_info "Installing scientific computing packages..."
        pip install numpy>=1.21.0
        pip install pandas>=1.5.0
        pip install scipy>=1.9.0
        pip install scikit-learn>=1.3.0
        
        # Visualization and monitoring
        log_info "Installing visualization and monitoring tools..."
        pip install matplotlib>=3.6.0
        pip install seaborn>=0.12.0
        pip install plotly>=5.15.0
        pip install wandb>=0.15.0
        pip install tensorboard>=2.13.0
        
        # Configuration and utilities
        log_info "Installing configuration and utility packages..."
        pip install PyYAML>=6.0
        pip install python-dotenv>=1.0.0
        pip install joblib>=1.3.0
        pip install tqdm>=4.65.0
        pip install click>=8.1.0
        
        # System monitoring
        log_info "Installing system monitoring tools..."
        pip install GPUtil>=1.4.0
        pip install psutil>=5.9.0
        pip install nvidia-ml-py3>=7.352.0
        
        # Web and API utilities
        log_info "Installing web and API utilities..."
        pip install requests>=2.31.0
        pip install beautifulsoup4>=4.12.0
        
        # HuggingFace and model utilities
        log_info "Installing HuggingFace and model utilities..."
        pip install huggingface-hub>=0.16.0
        pip install safetensors>=0.3.0
        pip install sentencepiece>=0.1.99
        
        # Memory optimization and additional utilities
        log_info "Installing optimization and utility packages..."
        pip install bitsandbytes>=0.39.0
        pip install Pillow>=9.5.0
        pip install packaging>=23.0
        pip install protobuf>=4.23.0
        
        # Optional development and Jupyter support
        log_info "Installing optional development tools..."
        pip install jupyter>=1.0.0
        pip install ipywidgets>=8.0.0
        pip install pytest>=7.4.0
        pip install black>=23.3.0
        pip install flake8>=6.0.0
        pip install mypy>=1.4.0
        
        log_success "Essential packages installed successfully"
    fi