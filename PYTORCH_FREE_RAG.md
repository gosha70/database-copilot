# PyTorch-free RAG Stack for Mac M1/M2

This document explains the PyTorch-free implementation of the Database Copilot RAG system, which is designed to work reliably on Apple Silicon (M1/M2) Macs without any dependency on PyTorch or sentence-transformers.

## Why a PyTorch-free Implementation?

The original implementation using sentence-transformers (which depends on PyTorch) can encounter the "cannot import name 'Tensor' from 'torch'" error on Mac M1/M2. This happens because:

1. Even after pinning NumPy < 2 and reinstalling PyTorch, sentence-transformers still imports torch at import-time
2. If any stray copy of the torch C-extensions (compiled against NumPy 1.x) is found earlier on sys.path, your fresh build is ignored
3. On a long-lived Mac/conda setup, it's hard to guarantee that no other conda channel pre-loads an incompatible pytorch-cpu build

## The PyTorch-free Solution

The PyTorch-free implementation uses:

| Component | Technology | Benefits |
|-----------|------------|----------|
| **Embeddings** | nomic-embed-text-v1.5.Q4_K_M.gguf via llama-cpp-python | Pure C++ with no dynamic Python libs, Metal acceleration out-of-the-box |
| **Vector store** | Chroma | Works with any embedding callable |
| **LLM** | Mistral 7B Q4_K_M.gguf via llama-cpp-python | Metal acceleration for fast inference |
| **Framework** | LangChain + LangGraph | Unchanged from original implementation |

## Technical Implementation

### Embedding Model

The implementation uses the `nomic-embed-text-v1.5.Q4_K_M.gguf` model (~180 MB), which is:
- Quantized to 4-bit for efficiency
- Optimized for RAG/similarity search
- Comparable to all-mpnet-base-v2 on MTEB benchmarks
- Accelerated via Metal on Mac M1/M2

### LlamaCppEmbeddings Class

The core of the implementation is the `LlamaCppEmbeddings` class in `backend/models/embeddings_local.py`:

```python
class LlamaCppEmbeddings:
    def __init__(self, model_path=None, n_ctx=8192, n_threads=8, n_gpu_layers=-1):
        """Initialize the LlamaCppEmbeddings with the given model."""
        try:
            import llama_cpp
            
            if model_path is None:
                from backend.config import MODELS_DIR, DEFAULT_LLAMA_CPP_EMBEDDING_MODEL
                model_path = os.path.join(MODELS_DIR, DEFAULT_LLAMA_CPP_EMBEDDING_MODEL)
            
            self.model = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                embedding=True
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Initialized LlamaCppEmbeddings with model: {model_path}")
            
        except ImportError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error importing llama_cpp: {e}")
            logger.error("Please install llama-cpp-python with:")
            logger.error("pip install llama-cpp-python")
            raise ImportError(f"Failed to import llama_cpp: {e}")
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text):
        """Generate embeddings for a single query."""
        return self.model.embed(text)
```

### Configuration

The `backend/config.py` file has been updated to support both embedding types:

```python
# Embedding configuration
EMBEDDING_TYPE = "llama_cpp"  # Options: "sentence_transformers", "llama_cpp"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_LLAMA_CPP_EMBEDDING_MODEL = "nomic-embed-text-v1.5.Q4_K_M.gguf"
```

### Vector Store Integration

The `backend/models/vector_store.py` file has been modified to use the appropriate embedding model based on the configuration:

```python
def get_embeddings():
    """Get the embedding model based on configuration."""
    from backend.config import EMBEDDING_TYPE
    
    if EMBEDDING_TYPE == "llama_cpp":
        from backend.models.embeddings_local import get_embeddings as get_llama_embeddings
        return get_llama_embeddings()
    else:
        from langchain.embeddings import HuggingFaceEmbeddings
        from backend.config import DEFAULT_EMBEDDING_MODEL
        return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
```

## Setup and Usage

### Environment Setup

```bash
# Create a conda environment
conda create -n database_copilot_cpp python=3.11
conda activate database_copilot_cpp

# Install core dependencies
pip install langchain==0.1.14 langgraph chromadb==0.4.24 tqdm

# Install llama.cpp binding with Metal support
pip install llama-cpp-python==0.2.24
```

### Download the Embedding Model

```bash
python download_nomic_embed.py
```

This will download the `nomic-embed-text-v1.5.Q4_K_M.gguf` model (~180 MB) to the `data/hf_models/` directory.

### Build the Vector Store

```bash
python rebuild_vector_store.py
```

This will:
1. Check if the embedding model exists
2. Initialize the LlamaCppEmbeddings
3. Load the documents from the data directory
4. Create embeddings for each document
5. Store the embeddings in a Chroma vector store

### Run the Application

```bash
python run_app_optimized.py
```

Or use the convenience script:

```bash
./run_torch_free.sh run
```

## Benefits of the PyTorch-free Implementation

1. **Reliability**: No more "cannot import name 'Tensor'" errors or dependency conflicts
2. **Performance**: Metal acceleration for both embeddings and LLM
3. **Efficiency**: Lower memory footprint with a single C++ backend
4. **Simplicity**: Fewer dependencies to manage
5. **Flexibility**: Easy to switch between different embedding models by just dropping in a new .gguf file

## Rebuilding an Existing Vector Store

If you already have a vector store built with the original PyTorch-based implementation, you need to rebuild it with the llama.cpp embeddings, as the embedding spaces are different between the two models.

```bash
python rebuild_vector_store.py
```

This will create a new vector store using the llama.cpp embeddings. The original vector store will not be modified, so you can switch back to the PyTorch-based implementation if needed.

## Troubleshooting

### Metal Acceleration Issues

If you encounter issues with Metal acceleration:

```bash
export GGML_METAL=0
python run_app_optimized.py
```

### Model Loading Issues

If the model fails to load:

1. Check that the model file exists in the `data/hf_models/` directory
2. Verify that you have sufficient disk space
3. Try downloading the model manually from Hugging Face

### Performance Optimization

For better performance:

1. Adjust `n_threads` in the LlamaCppEmbeddings constructor to match your CPU core count
2. Set `n_gpu_layers=-1` to offload all layers to the GPU
3. Adjust `n_ctx` based on your memory constraints and document length requirements
