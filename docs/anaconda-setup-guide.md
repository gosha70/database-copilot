# Anaconda Setup Guide for Database Copilot Enhancements

This guide provides instructions for setting up and running the Database Copilot enhancements within an Anaconda virtual environment.

## 1. Create and Activate Anaconda Environment

First, create a new Anaconda environment for the Database Copilot:

```bash
# Create a new environment with Python 3.11
conda create -n LiquibaseQA python=3.11

# Activate the environment
conda activate LiquibaseQA
```

## 2. Install Dependencies

Install all required dependencies using the requirements.txt file:

```bash
# Activate the conda environment first
conda activate LiquibaseQA

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

This will install all the necessary packages in one command.

For PyTorch, you might want to install it with conda for better compatibility:

```bash
# Install PyTorch with conda (optional, but recommended)
conda install -c pytorch pytorch
```

Alternatively, you can use our dependency installer script:

```bash
# Activate the conda environment first
conda activate LiquibaseQA

# Run the dependency installer
python install_dependencies.py
```

## 3. Set PYTHONPATH

To ensure Python can find the modules in the project, set the PYTHONPATH environment variable:

```bash
# For Linux/macOS
export PYTHONPATH=$PYTHONPATH:/path/to/database-copilot

# For Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\database-copilot
```

Replace `/path/to/database-copilot` with the actual path to your project directory.

## 4. Running the Test Script

To run the test script:

```bash
# Activate the conda environment
conda activate LiquibaseQA

# Set PYTHONPATH (if not already set)
export PYTHONPATH=$PYTHONPATH:/path/to/database-copilot

# Run the test script
python test_enhanced_reviewer.py --migration examples/20250505-create-custom-table.yaml
```

## 5. Running the Benchmark Script

To run the benchmark script:

```bash
# Activate the conda environment
conda activate LiquibaseQA

# Set PYTHONPATH (if not already set)
export PYTHONPATH=$PYTHONPATH:/path/to/database-copilot

# Run the benchmark script
python docs/benchmark_example.py --migration examples/20250505-create-custom-table.yaml
```

## 6. Running the Application

To run the main application (now launches the optimized app):

```bash
# Activate the conda environment
conda activate LiquibaseQA

# Set PYTHONPATH (if not already set)
export PYTHONPATH=$PYTHONPATH:/path/to/database-copilot

# Run the application
python run_app.py
```

For advanced options (CPU/memory limits, lazy loading, etc.), use:

```bash
python run_app_optimized.py [--cpu-limit N] [--memory-limit MB] [--use-external-llm] [--disable-vector-store] [--lazy-load]
```

## Troubleshooting

### Module Not Found Errors

If you encounter "Module not found" errors, ensure:

1. The conda environment is activated
2. PYTHONPATH is set correctly
3. All dependencies are installed

You can check if a module is installed with:

```bash
conda list | grep module_name
# or
pip list | grep module_name
```

### PyTorch Issues

If you encounter issues with PyTorch:

```bash
# Uninstall existing PyTorch
pip uninstall torch

# Install PyTorch with conda
conda install -c pytorch pytorch
```

### Streamlit Compatibility Issues

If you encounter issues with Streamlit and PyTorch:

1. Make sure you're using our streamlit_compatibility module:
   ```python
   from backend.models.streamlit_compatibility import get_safe_llm, get_safe_embedding_model
   ```

2. Try running Streamlit with the `--browser.serverAddress=localhost` flag:
   ```bash
   streamlit run backend/app_optimized.py --browser.serverAddress=localhost
   ```

### CUDA Issues

If you're using a GPU and encounter CUDA issues:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install specific CUDA version if needed
conda install cudatoolkit=11.8
```

## One-Line Setup Script

For convenience, you can use this one-line setup script:

```bash
# Create and setup environment in one go
conda create -n LiquibaseQA python=3.11 -y && conda activate LiquibaseQA && pip install -r requirements.txt && conda install -c pytorch pytorch -y
```
