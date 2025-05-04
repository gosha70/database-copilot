"""
LLM and embedding model initialization and utilities.
"""
import os
import logging
from typing import Optional

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import torch

from backend.config import (
    MODELS_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPETITION_PENALTY
)

logger = logging.getLogger(__name__)

def get_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Initialize and return a HuggingFace embedding model.
    
    Args:
        model_name: Name of the embedding model to use. If None, uses the default model.
    
    Returns:
        An initialized HuggingFaceEmbeddings instance.
    """
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model_name}")
    
    # Check if model exists locally, otherwise use from HuggingFace
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        logger.info(f"Using local embedding model from: {model_path}")
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs
        )
    else:
        logger.info(f"Using embedding model from HuggingFace Hub: {model_name}")
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )

def get_llm(model_name: Optional[str] = None) -> HuggingFacePipeline:
    """
    Initialize and return a HuggingFace LLM pipeline.
    
    Args:
        model_name: Name of the LLM model to use. If None, uses the default model.
    
    Returns:
        An initialized HuggingFacePipeline instance.
    """
    model_name = model_name or DEFAULT_LLM_MODEL
    logger.info(f"Loading LLM model: {model_name}")
    
    # Check if model exists locally, otherwise use from HuggingFace
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        logger.info(f"Using local LLM model from: {model_path}")
        model_location = model_path
    else:
        logger.info(f"Using LLM model from HuggingFace Hub: {model_name}")
        model_location = model_name
    
    # Configure quantization if GPU is available
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using quantized model.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        logger.info("CUDA is not available. Using CPU.")
        quantization_config = None
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForCausalLM.from_pretrained(
        model_location,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Create LangChain pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
