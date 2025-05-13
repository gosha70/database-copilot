"""
Alternative embedding models that don't rely on PyTorch or sentence-transformers.

This module provides alternative embedding implementations that can be used
when sentence-transformers is not available or not working properly.
"""
import logging
import numpy as np
from typing import List, Union, Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class TensorflowEmbeddings(Embeddings):
    """
    TensorFlow-based embeddings using Universal Sentence Encoder.
    
    This class provides embeddings using TensorFlow's Universal Sentence Encoder,
    which works well on Apple Silicon and doesn't require PyTorch.
    """
    
    def __init__(self, model_name: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        """
        Initialize the TensorFlow embeddings.
        
        Args:
            model_name: The name or URL of the TensorFlow Hub model to use.
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # Disable GPU memory preallocation
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Error setting memory growth: {e}")
            
            # Load the model
            logger.info(f"Loading TensorFlow model: {model_name}")
            self.model = hub.load(model_name)
            logger.info("TensorFlow model loaded successfully")
            
            # Store the embedding dimension
            self.embedding_dim = 512  # Universal Sentence Encoder has 512 dimensions
            
        except ImportError as e:
            logger.error(f"Error importing TensorFlow: {e}")
            logger.error("Please install TensorFlow and TensorFlow Hub with:")
            logger.error("pip install tensorflow tensorflow-hub")
            raise ImportError(f"Failed to import TensorFlow: {e}")
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            raise ValueError(f"Failed to load TensorFlow model: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: A list of documents to embed.
            
        Returns:
            A list of embeddings, one for each document.
        """
        try:
            import tensorflow as tf
            
            # Convert to TensorFlow tensor
            embeddings = self.model(texts)
            
            # Convert to numpy array and then to list
            return embeddings.numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.embedding_dim for _ in range(len(texts))]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: The query to embed.
            
        Returns:
            The embedding for the query.
        """
        try:
            import tensorflow as tf
            
            # Convert to TensorFlow tensor
            embedding = self.model([text])
            
            # Convert to numpy array and then to list
            return embedding.numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            # Return zero embedding as fallback
            return [0.0] * self.embedding_dim


class TfidfEmbeddings(Embeddings):
    """
    TF-IDF based embeddings that don't require PyTorch.
    
    This class provides embeddings using scikit-learn's TfidfVectorizer,
    which is a simple but effective approach for many NLP tasks.
    """
    
    def __init__(self, max_features: Optional[int] = None):
        """
        Initialize the TF-IDF embeddings.
        
        Args:
            max_features: The maximum number of features to use. If None, all features are used.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Initialize the vectorizer
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            self.fitted = False
            self.documents = []
            
            # Default embedding dimension
            self.embedding_dim = max_features if max_features is not None else 1000
            
        except ImportError as e:
            logger.error(f"Error importing scikit-learn: {e}")
            logger.error("Please install scikit-learn with:")
            logger.error("pip install scikit-learn")
            raise ImportError(f"Failed to import scikit-learn: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: A list of documents to embed.
            
        Returns:
            A list of embeddings, one for each document.
        """
        try:
            # Fit the vectorizer if not already fitted
            if not self.fitted:
                self.vectorizer.fit(texts)
                self.fitted = True
                self.documents = texts
                # Update embedding dimension
                self.embedding_dim = len(self.vectorizer.get_feature_names_out())
            
            # Transform the documents
            embeddings = self.vectorizer.transform(texts)
            
            # Convert to dense array and then to list
            return embeddings.toarray().tolist()
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.embedding_dim for _ in range(len(texts))]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: The query to embed.
            
        Returns:
            The embedding for the query.
        """
        try:
            # Check if the vectorizer is fitted
            if not self.fitted:
                logger.error("Vectorizer not fitted. Call embed_documents first.")
                # Return zero embedding as fallback
                return [0.0] * self.embedding_dim
            
            # Transform the query
            embedding = self.vectorizer.transform([text])
            
            # Convert to dense array and then to list
            return embedding.toarray()[0].tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            # Return zero embedding as fallback
            return [0.0] * self.embedding_dim


# Factory function to create embeddings
def create_alternative_embeddings(embedding_type: str = "tensorflow") -> Embeddings:
    """
    Create alternative embeddings based on the specified type.
    
    Args:
        embedding_type: The type of embeddings to create. Must be one of "tensorflow" or "tfidf".
        
    Returns:
        An initialized embeddings instance.
    """
    if embedding_type.lower() == "tensorflow":
        return TensorflowEmbeddings()
    elif embedding_type.lower() == "tfidf":
        return TfidfEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Supported types are: tensorflow, tfidf")
