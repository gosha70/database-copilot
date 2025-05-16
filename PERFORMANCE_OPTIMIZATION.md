# Performance Optimization Guide for Database Copilot

This guide explains the performance optimizations implemented to address the issue of the application freezing the computer and making it unresponsive.

## Problem

The original application was causing the computer to freeze and become unresponsive until VSCode was killed. This was likely due to:

1. Eager loading of all models and components at startup
2. High resource usage from the embedding model and LLM
3. Inefficient vector store initialization
4. Streamlit's file watcher consuming excessive resources

## Solution

Two optimized files have been created:

1. `run_app_optimized.py` - An optimized launcher script with resource control options
2. `backend/app_optimized.py` - An optimized version of the Streamlit application

### Key Optimizations

1. **Lazy Loading of Components**
   - Models and components are only loaded when needed, not at startup
   - This significantly reduces initial memory usage and startup time

2. **Resource Usage Controls**
   - CPU usage limits to prevent the application from using all available cores
   - Memory usage limits to prevent excessive memory consumption
   - Option to use external LLM APIs instead of local models

3. **Vector Store Optimizations**
   - Option to disable vector store initialization at startup
   - Vector store is only loaded when needed for specific operations

4. **Streamlit Optimizations**
   - Using polling file watcher instead of the default watcher
   - Caching of expensive operations using `lru_cache`
   - Performance metrics display to monitor application performance

## Usage

### Running the Optimized Version

```bash
python run_app_optimized.py
```

### Command Line Options

The optimized launcher script supports several command line options:

```bash
python run_app_optimized.py --help
```

Available options:

- `--cpu-limit N`: Limit CPU usage to N cores
- `--memory-limit N`: Limit memory usage to N MB
- `--use-external-llm`: Use external LLM API instead of local model
- `--disable-vector-store`: Disable vector store initialization at startup
- `--lazy-load`: Lazy load models only when needed

### Examples

1. Run with limited resources (recommended for most systems):
   ```bash
   python run_app_optimized.py --cpu-limit 2 --memory-limit 2048
   ```

2. Run with external LLM (requires API key in `.streamlit/secrets.toml`):
   ```bash
   python run_app_optimized.py --use-external-llm
   ```

3. Run with minimal resource usage:
   ```bash
   python run_app_optimized.py --cpu-limit 1 --memory-limit 1024 --disable-vector-store --lazy-load
   ```

## In-App Performance Settings

The optimized app includes performance settings in the sidebar:

1. **Vector Store Toggle**
   - Enable/disable the vector store without restarting the application
   - Useful when you don't need RAG capabilities

2. **External LLM Toggle**
   - Switch between local and external LLM models
   - Select from different providers (OpenAI, Claude, Gemini, etc.)

## Monitoring Performance

The optimized app displays startup time in the footer, allowing you to monitor the impact of different settings on performance.

## Troubleshooting

If you still experience performance issues:

1. Try running with `--disable-vector-store` to skip vector store initialization
2. Use `--use-external-llm` to offload LLM processing to external APIs
3. Reduce `--cpu-limit` and `--memory-limit` values
4. Check if your system has enough available memory (at least 4GB recommended)
5. Consider using a more powerful machine for resource-intensive operations

## Technical Details

### Lazy Loading Implementation

Components are loaded on-demand using a component registry:

```python
# Lazy loading of models and components
_COMPONENTS = {}

def get_component(component_name: str) -> Any:
    """
    Lazily load and cache components to reduce startup time and memory usage.
    """
    if component_name in _COMPONENTS:
        return _COMPONENTS[component_name]
    
    # Load the component only when needed
    # ...
```

### Resource Limiting

Resource limits are implemented using environment variables:

```python
# Set CPU limit
os.environ["OMP_NUM_THREADS"] = str(cpu_count)
os.environ["MKL_NUM_THREADS"] = str(cpu_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)

# Set memory limit for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{memory_limit}"
```

### Streamlit Optimizations

Streamlit's file watcher is configured to use polling mode:

```python
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
```

This reduces CPU usage by avoiding constant filesystem monitoring.
