"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Embeddings Tool - Generates and manages vector embeddings for text and other data types, supporting various embedding models and similarity operations.
"""

import os
import json
import pickle
import time
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import tool decorator for registration
from .tool_decorator import register_tool

# Optional imports with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
    # Configure OpenAI
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if os.getenv('OPENAI_ORG_ID'):
        openai.organization = os.getenv('OPENAI_ORG_ID')
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

@dataclass
class EmbeddingConfig:
    """Embedding configuration settings."""
    model_name: str = os.getenv('OPENAI_MODEL', 'text-embedding-ada-002')
    model_type: str = 'sentence_transformers'  # sentence_transformers, openai, huggingface, custom
    dimensions: int = int(os.getenv('VECTOR_DB_DIMENSION', '384'))
    normalize: bool = True
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = 'auto'  # auto, cpu, cuda
    cache_embeddings: bool = True

@dataclass
class VectorSearchConfig:
    """Configuration for vector search."""
    index_type: str = os.getenv('VECTOR_DB_TYPE', 'faiss')  # faiss, annoy, chroma, simple
    similarity_metric: str = os.getenv('VECTOR_DB_METRIC', 'cosine')  # cosine, euclidean, dot_product, manhattan
    index_params: Dict[str, Any] = None
    search_params: Dict[str, Any] = None
    build_index_on_add: bool = True

@dataclass
class DocumentMetadata:
    """Metadata for documents in vector store."""
    doc_id: str
    content: str
    embedding_hash: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    content_length: int
    embedding_model: str

class EmbeddingGenerator:
    """
    Advanced embedding generator supporting multiple models and providers.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.models = {}
        self.tokenizers = {}
        self.embedding_cache = {}
        self.generation_stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'total_processing_time': 0
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models based on configuration."""
        try:
            if self.config.model_type == 'sentence_transformers' and SENTENCE_TRANSFORMERS_AVAILABLE:
                self.models['sentence_transformers'] = SentenceTransformer(self.config.model_name)
                if self.config.device != 'auto':
                    self.models['sentence_transformers'].to(self.config.device)
            
            elif self.config.model_type == 'huggingface' and TRANSFORMERS_AVAILABLE:
                self.tokenizers['huggingface'] = AutoTokenizer.from_pretrained(self.config.model_name)
                self.models['huggingface'] = AutoModel.from_pretrained(self.config.model_name)
                
                if self.config.device != 'auto':
                    device = self.config.device
                else:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                self.models['huggingface'].to(device)
            
            elif self.config.model_type == 'openai' and OPENAI_AVAILABLE:
                # OpenAI embeddings don't need local model loading
                pass
                
        except Exception as e:
            logging.warning(f"Failed to initialize some embedding models: {e}")
    
    def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with embeddings and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
                single_text = True
            else:
                single_text = False
            
            # Check cache if enabled
            if self.config.cache_embeddings:
                cached_embeddings, uncached_texts, cache_indices = self._check_cache(texts)
            else:
                cached_embeddings = []
                uncached_texts = texts
                cache_indices = list(range(len(texts)))
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if self.config.model_type == 'sentence_transformers':
                    new_embeddings = self._generate_sentence_transformers(uncached_texts)
                elif self.config.model_type == 'huggingface':
                    new_embeddings = self._generate_huggingface(uncached_texts)
                elif self.config.model_type == 'openai':
                    new_embeddings = self._generate_openai(uncached_texts)
                else:
                    return {'status': 'error', 'message': f'Unsupported model type: {self.config.model_type}'}
                
                if new_embeddings['status'] != 'success':
                    return new_embeddings
                
                # Cache new embeddings
                if self.config.cache_embeddings:
                    self._update_cache(uncached_texts, new_embeddings['embeddings'])
                
                embeddings = new_embeddings['embeddings']
            else:
                embeddings = []
            
            # Combine cached and new embeddings
            final_embeddings = self._combine_embeddings(cached_embeddings, embeddings, cache_indices)
            
            # Normalize if requested
            if self.config.normalize:
                final_embeddings = normalize(final_embeddings, norm='l2')
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.generation_stats['total_embeddings'] += len(texts)
            self.generation_stats['cache_hits'] += len(cached_embeddings)
            self.generation_stats['total_processing_time'] += processing_time
            
            result = {
                'status': 'success',
                'embeddings': final_embeddings[0] if single_text else final_embeddings,
                'dimensions': final_embeddings.shape[1],
                'count': len(texts),
                'processing_time': processing_time,
                'cache_hits': len(cached_embeddings),
                'model_info': {
                    'model_name': self.config.model_name,
                    'model_type': self.config.model_type,
                    'dimensions': self.config.dimensions
                }
            }
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Embedding generation failed: {str(e)}'}
    
    def _check_cache(self, texts: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
        """Check cache for existing embeddings."""
        cached_embeddings = []
        uncached_texts = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[text_hash])
            else:
                uncached_texts.append(text)
                cache_indices.append(i)
        
        return np.array(cached_embeddings) if cached_embeddings else np.empty((0, self.config.dimensions)), uncached_texts, cache_indices
    
    def _update_cache(self, texts: List[str], embeddings: np.ndarray):
        """Update embedding cache."""
        for text, embedding in zip(texts, embeddings):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self.embedding_cache[text_hash] = embedding
    
    def _combine_embeddings(self, cached: np.ndarray, new: np.ndarray, new_indices: List[int]) -> np.ndarray:
        """Combine cached and new embeddings in correct order."""
        if len(cached) == 0:
            return new
        if len(new) == 0:
            return cached
        
        total_count = len(cached) + len(new)
        combined = np.zeros((total_count, cached.shape[1] if len(cached) > 0 else new.shape[1]))
        
        cached_idx = 0
        new_idx = 0
        
        for i in range(total_count):
            if i in new_indices:
                combined[i] = new[new_idx]
                new_idx += 1
            else:
                combined[i] = cached[cached_idx]
                cached_idx += 1
        
        return combined
    
    def _generate_sentence_transformers(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings using SentenceTransformers."""
        try:
            model = self.models['sentence_transformers']
            embeddings = model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return {
                'status': 'success',
                'embeddings': embeddings,
                'model': 'sentence_transformers'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'SentenceTransformers generation failed: {str(e)}'}
    
    def _generate_huggingface(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings using Hugging Face transformers."""
        try:
            model = self.models['huggingface']
            tokenizer = self.tokenizers['huggingface']
            
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length,
                    return_tensors='pt'
                )
                
                # Move to device
                device = next(model.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**encoded)
                    # Use mean pooling of last hidden states
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(batch_embeddings.cpu().numpy())
            
            return {
                'status': 'success',
                'embeddings': np.vstack(embeddings),
                'model': 'huggingface'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Hugging Face generation failed: {str(e)}'}
    
    def _generate_openai(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings using OpenAI API."""
        try:
            embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                response = openai.Embedding.create(
                    input=batch_texts,
                    model=self.config.model_name
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            
            return {
                'status': 'success',
                'embeddings': np.array(embeddings),
                'model': 'openai'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'OpenAI generation failed: {str(e)}'}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'config': asdict(self.config),
            'loaded_models': list(self.models.keys()),
            'available_backends': {
                'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
                'huggingface': TRANSFORMERS_AVAILABLE,
                'openai': OPENAI_AVAILABLE
            },
            'cache_size': len(self.embedding_cache),
            'stats': self.generation_stats
        }

class VectorStore:
    """
    Advanced vector storage and indexing system supporting multiple backends.
    """
    
    def __init__(self, config: VectorSearchConfig = None, embedding_dim: int = 384):
        self.config = config or VectorSearchConfig()
        self.embedding_dim = embedding_dim
        self.documents = {}
        self.embeddings = None
        self.indices = {}
        self.metadata_db = {}
        
        # Storage statistics
        self.storage_stats = {
            'total_documents': 0,
            'total_embeddings': 0,
            'index_build_time': 0,
            'last_updated': None
        }
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize vector storage backends."""
        try:
            # Initialize FAISS index
            if self.config.index_type == 'faiss' and FAISS_AVAILABLE:
                if self.config.similarity_metric == 'cosine':
                    self.indices['faiss'] = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    self.indices['faiss'] = faiss.IndexFlatL2(self.embedding_dim)
            
            # Initialize Chroma database
            if self.config.index_type == 'chroma' and CHROMA_AVAILABLE:
                self.indices['chroma'] = chromadb.Client()
                self.chroma_collection = self.indices['chroma'].create_collection(
                    name="vector_store",
                    metadata={"hnsw:space": self.config.similarity_metric}
                )
            
            # Initialize Annoy index
            if self.config.index_type == 'annoy' and ANNOY_AVAILABLE:
                metric = 'angular' if self.config.similarity_metric == 'cosine' else 'euclidean'
                self.indices['annoy'] = AnnoyIndex(self.embedding_dim, metric)
                self.annoy_built = False
                
        except Exception as e:
            logging.warning(f"Failed to initialize some storage backends: {e}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray = None, 
                     metadata: List[Dict[str, Any]] = None, doc_ids: List[str] = None) -> Dict[str, Any]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of document texts
            embeddings: Precomputed embeddings (optional)
            metadata: Document metadata (optional)
            doc_ids: Document IDs (optional)
            
        Returns:
            Addition result
        """
        try:
            start_time = time.time()
            
            # Generate doc IDs if not provided
            if doc_ids is None:
                doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
            
            # Generate embeddings if not provided
            if embeddings is None:
                embedding_generator = EmbeddingGenerator()
                result = embedding_generator.generate_embeddings(documents)
                if result['status'] != 'success':
                    return result
                embeddings = result['embeddings']
            
            # Ensure embeddings is 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Normalize embeddings if using cosine similarity
            if self.config.similarity_metric == 'cosine':
                embeddings = normalize(embeddings, norm='l2')
            
            # Store documents and metadata
            for i, (doc_id, document) in enumerate(zip(doc_ids, documents)):
                doc_metadata = DocumentMetadata(
                    doc_id=doc_id,
                    content=document,
                    embedding_hash=hashlib.md5(embeddings[i].tobytes()).hexdigest(),
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    metadata=metadata[i] if metadata and i < len(metadata) else {},
                    content_length=len(document),
                    embedding_model="current_model"
                )
                
                self.documents[doc_id] = document
                self.metadata_db[doc_id] = doc_metadata
            
            # Add to storage backends
            if self.config.index_type == 'faiss' and 'faiss' in self.indices:
                self._add_to_faiss(embeddings)
            elif self.config.index_type == 'chroma' and 'chroma' in self.indices:
                self._add_to_chroma(documents, embeddings, doc_ids, metadata)
            elif self.config.index_type == 'annoy' and 'annoy' in self.indices:
                self._add_to_annoy(embeddings)
            else:
                # Simple storage
                if self.embeddings is None:
                    self.embeddings = embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, embeddings])
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.storage_stats['total_documents'] += len(documents)
            self.storage_stats['total_embeddings'] += len(embeddings)
            self.storage_stats['last_updated'] = datetime.now().isoformat()
            
            return {
                'status': 'success',
                'documents_added': len(documents),
                'doc_ids': doc_ids,
                'processing_time': processing_time,
                'total_documents': self.storage_stats['total_documents']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to add documents: {str(e)}'}
    
    def _add_to_faiss(self, embeddings: np.ndarray):
        """Add embeddings to FAISS index."""
        self.indices['faiss'].add(embeddings.astype(np.float32))
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def _add_to_chroma(self, documents: List[str], embeddings: np.ndarray, 
                      doc_ids: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to Chroma database."""
        self.chroma_collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=doc_ids,
            metadatas=metadata or [{}] * len(documents)
        )
    
    def _add_to_annoy(self, embeddings: np.ndarray):
        """Add embeddings to Annoy index."""
        start_idx = len(self.embeddings) if self.embeddings is not None else 0
        
        for i, embedding in enumerate(embeddings):
            self.indices['annoy'].add_item(start_idx + i, embedding)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.annoy_built = False  # Need to rebuild
    
    def build_index(self) -> Dict[str, Any]:
        """Build/rebuild search index."""
        try:
            start_time = time.time()
            
            if self.config.index_type == 'annoy' and 'annoy' in self.indices and not self.annoy_built:
                # Build Annoy index
                n_trees = self.config.index_params.get('n_trees', 10) if self.config.index_params else 10
                self.indices['annoy'].build(n_trees)
                self.annoy_built = True
            
            build_time = time.time() - start_time
            self.storage_stats['index_build_time'] = build_time
            
            return {
                'status': 'success',
                'build_time': build_time,
                'index_type': self.config.index_type
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Index building failed: {str(e)}'}
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (optional)
            
        Returns:
            Search results with documents and similarities
        """
        try:
            start_time = time.time()
            
            # Ensure query_embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize if using cosine similarity
            if self.config.similarity_metric == 'cosine':
                query_embedding = normalize(query_embedding, norm='l2')
            
            # Search using appropriate backend
            if self.config.index_type == 'faiss' and 'faiss' in self.indices:
                results = self._search_faiss(query_embedding, top_k)
            elif self.config.index_type == 'chroma' and 'chroma' in self.indices:
                results = self._search_chroma(query_embedding, top_k, filters)
            elif self.config.index_type == 'annoy' and 'annoy' in self.indices:
                results = self._search_annoy(query_embedding, top_k)
            else:
                # Simple similarity search
                results = self._search_simple(query_embedding, top_k)
            
            if results['status'] != 'success':
                return results
            
            # Apply filters if specified and not handled by backend
            if filters and self.config.index_type not in ['chroma']:
                results = self._apply_filters(results, filters)
            
            search_time = time.time() - start_time
            results['search_time'] = search_time
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': f'Search failed: {str(e)}'}
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> Dict[str, Any]:
        """Search using FAISS index."""
        try:
            similarities, indices = self.indices['faiss'].search(query_embedding.astype(np.float32), top_k)
            
            results = []
            doc_ids = list(self.documents.keys())
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    results.append({
                        'doc_id': doc_id,
                        'document': self.documents[doc_id],
                        'similarity': float(similarity),
                        'rank': i + 1,
                        'metadata': asdict(self.metadata_db[doc_id]) if doc_id in self.metadata_db else {}
                    })
            
            return {
                'status': 'success',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'FAISS search failed: {str(e)}'}
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int, 
                      filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search using Chroma database."""
        try:
            chroma_results = self.chroma_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=filters
            )
            
            results = []
            for i in range(len(chroma_results['ids'][0])):
                doc_id = chroma_results['ids'][0][i]
                results.append({
                    'doc_id': doc_id,
                    'document': chroma_results['documents'][0][i],
                    'similarity': 1 - chroma_results['distances'][0][i],  # Convert distance to similarity
                    'rank': i + 1,
                    'metadata': chroma_results['metadatas'][0][i] if chroma_results['metadatas'][0] else {}
                })
            
            return {
                'status': 'success',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Chroma search failed: {str(e)}'}
    
    def _search_annoy(self, query_embedding: np.ndarray, top_k: int) -> Dict[str, Any]:
        """Search using Annoy index."""
        try:
            if not self.annoy_built:
                self.build_index()
            
            indices, distances = self.indices['annoy'].get_nns_by_vector(
                query_embedding[0], top_k, include_distances=True
            )
            
            results = []
            doc_ids = list(self.documents.keys())
            
            for i, (idx, distance) in enumerate(zip(indices, distances)):
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    # Convert distance to similarity
                    similarity = 1 / (1 + distance) if self.config.similarity_metric == 'euclidean' else 1 - distance
                    
                    results.append({
                        'doc_id': doc_id,
                        'document': self.documents[doc_id],
                        'similarity': float(similarity),
                        'rank': i + 1,
                        'metadata': asdict(self.metadata_db[doc_id]) if doc_id in self.metadata_db else {}
                    })
            
            return {
                'status': 'success',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Annoy search failed: {str(e)}'}
    
    def _search_simple(self, query_embedding: np.ndarray, top_k: int) -> Dict[str, Any]:
        """Simple similarity search using scikit-learn."""
        try:
            if self.embeddings is None or len(self.embeddings) == 0:
                return {'status': 'success', 'results': [], 'total_results': 0}
            
            # Calculate similarities
            if self.config.similarity_metric == 'cosine':
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            elif self.config.similarity_metric == 'euclidean':
                distances = euclidean_distances(query_embedding, self.embeddings)[0]
                similarities = 1 / (1 + distances)  # Convert to similarity
            else:
                similarities = np.dot(query_embedding, self.embeddings.T)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            doc_ids = list(self.documents.keys())
            
            results = []
            for i, idx in enumerate(top_indices):
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    results.append({
                        'doc_id': doc_id,
                        'document': self.documents[doc_id],
                        'similarity': float(similarities[idx]),
                        'rank': i + 1,
                        'metadata': asdict(self.metadata_db[doc_id]) if doc_id in self.metadata_db else {}
                    })
            
            return {
                'status': 'success',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Simple search failed: {str(e)}'}
    
    def _apply_filters(self, results: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply metadata filters to search results."""
        try:
            filtered_results = []
            
            for result in results['results']:
                metadata = result.get('metadata', {})
                match = True
                
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    filtered_results.append(result)
            
            # Update ranks
            for i, result in enumerate(filtered_results):
                result['rank'] = i + 1
            
            results['results'] = filtered_results
            results['total_results'] = len(filtered_results)
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': f'Filter application failed: {str(e)}'}
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document by ID."""
        try:
            if doc_id not in self.documents:
                return {'status': 'error', 'message': f'Document {doc_id} not found'}
            
            return {
                'status': 'success',
                'doc_id': doc_id,
                'document': self.documents[doc_id],
                'metadata': asdict(self.metadata_db[doc_id]) if doc_id in self.metadata_db else {}
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get document: {str(e)}'}
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete document from store."""
        try:
            if doc_id not in self.documents:
                return {'status': 'error', 'message': f'Document {doc_id} not found'}
            
            # Remove from documents and metadata
            del self.documents[doc_id]
            if doc_id in self.metadata_db:
                del self.metadata_db[doc_id]
            
            # Note: For FAISS/Annoy, this would require rebuilding the index
            # This is a simplified implementation
            
            self.storage_stats['total_documents'] -= 1
            self.storage_stats['last_updated'] = datetime.now().isoformat()
            
            return {
                'status': 'success',
                'message': f'Document {doc_id} deleted',
                'remaining_documents': self.storage_stats['total_documents']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete document: {str(e)}'}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'status': 'success',
            'stats': self.storage_stats,
            'config': asdict(self.config),
            'available_backends': {
                'faiss': FAISS_AVAILABLE,
                'chroma': CHROMA_AVAILABLE,
                'annoy': ANNOY_AVAILABLE
            }
        }

class EmbeddingsVectorSearchTool:
    """
    Comprehensive embeddings and vector search tool combining generation and storage.
    """
    
    def __init__(self, embedding_config: EmbeddingConfig = None, 
                 search_config: VectorSearchConfig = None):
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.search_config = search_config or VectorSearchConfig()
        
        self.embedding_generator = EmbeddingGenerator(self.embedding_config)
        self.vector_store = VectorStore(self.search_config, self.embedding_config.dimensions)
        
        # Text preprocessing options
        self.preprocessing_enabled = True
        self.preprocessing_options = {
            'lowercase': True,
            'remove_punctuation': False,
            'remove_extra_whitespace': True,
            'min_length': 3
        }
        
        # Operation history
        self.operation_history = []
    
    def add_texts(self, texts: List[str], metadata: List[Dict[str, Any]] = None, 
                 doc_ids: List[str] = None, preprocess: bool = None) -> Dict[str, Any]:
        """
        Add texts to vector store with automatic embedding generation.
        
        Args:
            texts: List of texts to add
            metadata: Document metadata (optional)
            doc_ids: Document IDs (optional)
            preprocess: Whether to preprocess texts (optional)
            
        Returns:
            Addition result
        """
        try:
            start_time = time.time()
            
            # Preprocess texts if enabled
            if preprocess is None:
                preprocess = self.preprocessing_enabled
            
            if preprocess:
                processed_texts = [self._preprocess_text(text) for text in texts]
                # Filter out texts that are too short
                valid_indices = [i for i, text in enumerate(processed_texts) 
                               if len(text) >= self.preprocessing_options['min_length']]
                
                if len(valid_indices) != len(texts):
                    processed_texts = [processed_texts[i] for i in valid_indices]
                    if metadata:
                        metadata = [metadata[i] for i in valid_indices]
                    if doc_ids:
                        doc_ids = [doc_ids[i] for i in valid_indices]
            else:
                processed_texts = texts
            
            if not processed_texts:
                return {'status': 'error', 'message': 'No valid texts after preprocessing'}
            
            # Generate embeddings
            embedding_result = self.embedding_generator.generate_embeddings(processed_texts)
            if embedding_result['status'] != 'success':
                return embedding_result
            
            # Add to vector store
            store_result = self.vector_store.add_documents(
                documents=processed_texts,
                embeddings=embedding_result['embeddings'],
                metadata=metadata,
                doc_ids=doc_ids
            )
            
            if store_result['status'] != 'success':
                return store_result
            
            processing_time = time.time() - start_time
            
            # Record operation
            operation = {
                'type': 'add_texts',
                'timestamp': datetime.now().isoformat(),
                'texts_count': len(texts),
                'valid_texts_count': len(processed_texts),
                'processing_time': processing_time,
                'success': True
            }
            self.operation_history.append(operation)
            
            return {
                'status': 'success',
                'texts_added': len(processed_texts),
                'texts_filtered': len(texts) - len(processed_texts),
                'doc_ids': store_result['doc_ids'],
                'processing_time': processing_time,
                'embedding_info': embedding_result['model_info']
            }
            
        except Exception as e:
            # Record failed operation
            operation = {
                'type': 'add_texts',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
            self.operation_history.append(operation)
            
            return {'status': 'error', 'message': f'Failed to add texts: {str(e)}'}
    
    def search_similar(self, query: str, top_k: int = 10, 
                      filters: Dict[str, Any] = None, preprocess: bool = None) -> Dict[str, Any]:
        """
        Search for texts similar to query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (optional)
            preprocess: Whether to preprocess query (optional)
            
        Returns:
            Search results
        """
        try:
            start_time = time.time()
            
            # Preprocess query if enabled
            if preprocess is None:
                preprocess = self.preprocessing_enabled
            
            if preprocess:
                processed_query = self._preprocess_text(query)
            else:
                processed_query = query
            
            # Generate query embedding
            embedding_result = self.embedding_generator.generate_embeddings(processed_query)
            if embedding_result['status'] != 'success':
                return embedding_result
            
            # Search in vector store
            search_result = self.vector_store.search(
                query_embedding=embedding_result['embeddings'],
                top_k=top_k,
                filters=filters
            )
            
            if search_result['status'] != 'success':
                return search_result
            
            processing_time = time.time() - start_time
            
            # Record operation
            operation = {
                'type': 'search_similar',
                'timestamp': datetime.now().isoformat(),
                'query': query[:100] + '...' if len(query) > 100 else query,
                'results_count': len(search_result['results']),
                'processing_time': processing_time,
                'success': True
            }
            self.operation_history.append(operation)
            
            search_result['total_processing_time'] = processing_time
            search_result['query_processed'] = processed_query
            
            return search_result
            
        except Exception as e:
            # Record failed operation
            operation = {
                'type': 'search_similar',
                'timestamp': datetime.now().isoformat(),
                'query': query[:100] + '...' if len(query) > 100 else query,
                'error': str(e),
                'success': False
            }
            self.operation_history.append(operation)
            
            return {'status': 'error', 'message': f'Search failed: {str(e)}'}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text according to configuration."""
        if not self.preprocessing_enabled:
            return text
        
        processed = text
        
        if self.preprocessing_options['lowercase']:
            processed = processed.lower()
        
        if self.preprocessing_options['remove_punctuation']:
            processed = re.sub(r'[^\w\s]', '', processed)
        
        if self.preprocessing_options['remove_extra_whitespace']:
            processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def cluster_embeddings(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Cluster stored embeddings.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans')
            
        Returns:
            Clustering results
        """
        try:
            if self.vector_store.embeddings is None or len(self.vector_store.embeddings) == 0:
                return {'status': 'error', 'message': 'No embeddings available for clustering'}
            
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(self.vector_store.embeddings)
                cluster_centers = clusterer.cluster_centers_
            else:
                return {'status': 'error', 'message': f'Unsupported clustering method: {method}'}
            
            # Organize results by cluster
            clusters = {}
            doc_ids = list(self.vector_store.documents.keys())
            
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                
                if i < len(doc_ids):
                    clusters[label].append({
                        'doc_id': doc_ids[i],
                        'document': self.vector_store.documents[doc_ids[i]][:100] + '...'
                        if len(self.vector_store.documents[doc_ids[i]]) > 100
                        else self.vector_store.documents[doc_ids[i]]
                    })
            
            return {
                'status': 'success',
                'n_clusters': n_clusters,
                'method': method,
                'clusters': clusters,
                'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
                'cluster_centers': cluster_centers.tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Clustering failed: {str(e)}'}
    
    def reduce_dimensions(self, method: str = 'pca', n_components: int = 2) -> Dict[str, Any]:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            method: Dimensionality reduction method ('pca', 'umap')
            n_components: Number of output dimensions
            
        Returns:
            Reduced embeddings
        """
        try:
            if self.vector_store.embeddings is None or len(self.vector_store.embeddings) == 0:
                return {'status': 'error', 'message': 'No embeddings available for dimension reduction'}
            
            if method == 'pca':
                reducer = PCA(n_components=n_components)
                reduced_embeddings = reducer.fit_transform(self.vector_store.embeddings)
                explained_variance = reducer.explained_variance_ratio_.tolist()
            elif method == 'umap' and UMAP_AVAILABLE:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                reduced_embeddings = reducer.fit_transform(self.vector_store.embeddings)
                explained_variance = None
            else:
                return {'status': 'error', 'message': f'Unsupported or unavailable method: {method}'}
            
            return {
                'status': 'success',
                'method': method,
                'original_dimensions': self.vector_store.embeddings.shape[1],
                'reduced_dimensions': n_components,
                'reduced_embeddings': reduced_embeddings.tolist(),
                'explained_variance': explained_variance
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Dimension reduction failed: {str(e)}'}
    
    def export_data(self, file_path: str, format: str = 'json', 
                   include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Export vector store data to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json', 'csv', 'pickle')
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            Export result
        """
        try:
            export_data = {
                'documents': self.vector_store.documents,
                'metadata': {k: asdict(v) for k, v in self.vector_store.metadata_db.items()},
                'stats': self.vector_store.storage_stats,
                'config': {
                    'embedding_config': asdict(self.embedding_config),
                    'search_config': asdict(self.search_config)
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            if include_embeddings and self.vector_store.embeddings is not None:
                export_data['embeddings'] = self.vector_store.embeddings.tolist()
            
            if format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(export_data, f)
            elif format == 'csv':
                # Export as CSV (limited data)
                df_data = []
                for doc_id, document in self.vector_store.documents.items():
                    row = {
                        'doc_id': doc_id,
                        'document': document,
                        'content_length': len(document)
                    }
                    if doc_id in self.vector_store.metadata_db:
                        metadata = self.vector_store.metadata_db[doc_id]
                        row.update({
                            'created_at': metadata.created_at,
                            'updated_at': metadata.updated_at
                        })
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(file_path, index=False)
            else:
                return {'status': 'error', 'message': f'Unsupported format: {format}'}
            
            file_size = os.path.getsize(file_path)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': format,
                'file_size': file_size,
                'documents_exported': len(self.vector_store.documents),
                'embeddings_included': include_embeddings
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Export failed: {str(e)}'}
    
    def import_data(self, file_path: str, format: str = 'json') -> Dict[str, Any]:
        """
        Import vector store data from file.
        
        Args:
            file_path: Input file path
            format: Import format ('json', 'pickle')
            
        Returns:
            Import result
        """
        try:
            if not os.path.exists(file_path):
                return {'status': 'error', 'message': f'File not found: {file_path}'}
            
            if format == 'json':
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    import_data = pickle.load(f)
            else:
                return {'status': 'error', 'message': f'Unsupported format: {format}'}
            
            # Import documents
            documents = import_data.get('documents', {})
            metadata = import_data.get('metadata', {})
            
            if documents:
                doc_ids = list(documents.keys())
                texts = list(documents.values())
                doc_metadata = [metadata.get(doc_id, {}) for doc_id in doc_ids]
                
                # Import embeddings if available
                embeddings = None
                if 'embeddings' in import_data:
                    embeddings = np.array(import_data['embeddings'])
                
                result = self.add_texts(texts, doc_metadata, doc_ids, preprocess=False)
                if result['status'] != 'success':
                    return result
            
            return {
                'status': 'success',
                'documents_imported': len(documents),
                'file_path': file_path,
                'format': format
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Import failed: {str(e)}'}
    
    def get_operation_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent operation history."""
        return {
            'status': 'success',
            'history': self.operation_history[-limit:],
            'total_operations': len(self.operation_history)
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        embedding_stats = self.embedding_generator.get_model_info()
        storage_stats = self.vector_store.get_stats()
        
        return {
            'status': 'success',
            'embedding_stats': embedding_stats,
            'storage_stats': storage_stats['stats'],
            'total_operations': len(self.operation_history),
            'successful_operations': len([op for op in self.operation_history if op.get('success', False)]),
            'available_features': {
                'clustering': True,
                'dimension_reduction': True,
                'pca': True,
                'umap': UMAP_AVAILABLE,
                'export_import': True
            }
        }


# Agent framework integration
class EmbeddingsVectorSearchAgent:
    """
    Agent wrapper for the embeddings and vector search tool.
    """
    
    def __init__(self, embedding_config: EmbeddingConfig = None, 
                 search_config: VectorSearchConfig = None):
        self.tool = EmbeddingsVectorSearchTool(embedding_config, search_config)
        self.capabilities = [
            'add_texts',
            'search_similar',
            'generate_embeddings',
            'cluster_embeddings',
            'reduce_dimensions',
            'export_data',
            'import_data',
            'get_document',
            'delete_document',
            'get_stats',
            'configure_preprocessing'
        ]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific embeddings/search operation."""
        try:
            if action == 'add_texts':
                return self.tool.add_texts(**kwargs)
            elif action == 'search_similar':
                return self.tool.search_similar(**kwargs)
            elif action == 'generate_embeddings':
                return self.tool.embedding_generator.generate_embeddings(**kwargs)
            elif action == 'cluster_embeddings':
                return self.tool.cluster_embeddings(**kwargs)
            elif action == 'reduce_dimensions':
                return self.tool.reduce_dimensions(**kwargs)
            elif action == 'export_data':
                return self.tool.export_data(**kwargs)
            elif action == 'import_data':
                return self.tool.import_data(**kwargs)
            elif action == 'get_document':
                return self.tool.vector_store.get_document(**kwargs)
            elif action == 'delete_document':
                return self.tool.vector_store.delete_document(**kwargs)
            elif action == 'get_stats':
                return self.tool.get_comprehensive_stats()
            elif action == 'get_history':
                return self.tool.get_operation_history(**kwargs)
            elif action == 'configure_preprocessing':
                for key, value in kwargs.items():
                    if key in self.tool.preprocessing_options:
                        self.tool.preprocessing_options[key] = value
                    elif key == 'enabled':
                        self.tool.preprocessing_enabled = value
                return {'status': 'success', 'config': self.tool.preprocessing_options}
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error executing {action}: {str(e)}'}
    
    def get_capabilities(self) -> List[str]:
        """Return list of available capabilities."""
        return self.capabilities.copy()


# Quick utility functions
@register_tool(tags=["embeddings", "text", "ai", "vectorization"])
def quick_embed(texts: Union[str, List[str]], model: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Quick embedding generation."""
    config = EmbeddingConfig(model_name=model)
    generator = EmbeddingGenerator(config)
    result = generator.generate_embeddings(texts)
    
    if result['status'] == 'success':
        return result['embeddings']
    else:
        raise Exception(result['message'])

@register_tool(tags=["embeddings", "search", "vector", "setup"])
def quick_search_setup(texts: List[str], index_type: str = 'simple') -> EmbeddingsVectorSearchTool:
    """Quick setup of vector search with texts."""
    search_config = VectorSearchConfig(index_type=index_type)
    tool = EmbeddingsVectorSearchTool(search_config=search_config)
    
    result = tool.add_texts(texts)
    if result['status'] != 'success':
        raise Exception(result['message'])
    
    return tool

@register_tool(tags=["embeddings", "search", "similarity", "text"])
def similarity_search(query: str, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """Quick similarity search without persistent storage."""
    tool = quick_search_setup(texts)
    result = tool.search_similar(query, top_k=top_k)
    
    if result['status'] == 'success':
        return result['results']
    else:
        raise Exception(result['message'])

# Advanced tool functions for complex operations
@register_tool(tags=["embeddings", "text", "add", "vector_store"])
def add_texts_to_vector_store(texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None, 
                             doc_ids: Optional[List[str]] = None, model: str = 'all-MiniLM-L6-v2',
                             index_type: str = 'simple', preprocess: bool = True) -> Dict[str, Any]:
    """
    Add texts to a vector store with embeddings.
    
    Args:
        texts: List of text documents to add
        metadata: Optional metadata for each document
        doc_ids: Optional document IDs
        model: Embedding model to use
        index_type: Vector index type ('simple', 'faiss', 'annoy', 'chroma')
        preprocess: Whether to preprocess text
        
    Returns:
        Dictionary with operation status and results
    """
    try:
        embedding_config = EmbeddingConfig(model_name=model)
        search_config = VectorSearchConfig(index_type=index_type)
        tool = EmbeddingsVectorSearchTool(embedding_config, search_config)
        
        return tool.add_texts(texts, metadata, doc_ids, preprocess)
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to add texts: {str(e)}'}

@register_tool(tags=["embeddings", "search", "query", "vector_store"])
def search_similar_texts(query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None,
                        model: str = 'all-MiniLM-L6-v2', index_type: str = 'simple',
                        preprocess: bool = True) -> Dict[str, Any]:
    """
    Search for similar texts in a vector store.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        filters: Optional filters to apply
        model: Embedding model to use
        index_type: Vector index type
        preprocess: Whether to preprocess query
        
    Returns:
        Dictionary with search results
    """
    try:
        embedding_config = EmbeddingConfig(model_name=model)
        search_config = VectorSearchConfig(index_type=index_type)
        tool = EmbeddingsVectorSearchTool(embedding_config, search_config)
        
        return tool.search_similar(query, top_k, filters, preprocess)
    except Exception as e:
        return {'status': 'error', 'message': f'Search failed: {str(e)}'}

@register_tool(tags=["embeddings", "analysis", "clustering", "ml"])
def cluster_text_embeddings(texts: List[str], n_clusters: int = 5, method: str = 'kmeans',
                           model: str = 'all-MiniLM-L6-v2') -> Dict[str, Any]:
    """
    Cluster texts based on their embeddings.
    
    Args:
        texts: List of texts to cluster
        n_clusters: Number of clusters
        method: Clustering method ('kmeans')
        model: Embedding model to use
        
    Returns:
        Dictionary with clustering results
    """
    try:
        embedding_config = EmbeddingConfig(model_name=model)
        tool = EmbeddingsVectorSearchTool(embedding_config)
        
        # Add texts first
        add_result = tool.add_texts(texts)
        if add_result['status'] != 'success':
            return add_result
            
        return tool.cluster_embeddings(n_clusters, method)
    except Exception as e:
        return {'status': 'error', 'message': f'Clustering failed: {str(e)}'}

@register_tool(tags=["embeddings", "analysis", "dimensionality_reduction", "visualization"])
def reduce_embedding_dimensions(texts: List[str], method: str = 'pca', n_components: int = 2,
                               model: str = 'all-MiniLM-L6-v2') -> Dict[str, Any]:
    """
    Reduce dimensionality of text embeddings for visualization.
    
    Args:
        texts: List of texts to process
        method: Reduction method ('pca', 'umap')
        n_components: Number of dimensions to reduce to
        model: Embedding model to use
        
    Returns:
        Dictionary with reduced embeddings
    """
    try:
        embedding_config = EmbeddingConfig(model_name=model)
        tool = EmbeddingsVectorSearchTool(embedding_config)
        
        # Add texts first
        add_result = tool.add_texts(texts)
        if add_result['status'] != 'success':
            return add_result
            
        return tool.reduce_dimensions(method, n_components)
    except Exception as e:
        return {'status': 'error', 'message': f'Dimension reduction failed: {str(e)}'}

@register_tool(tags=["embeddings", "export", "data", "persistence"])
def export_vector_data(file_path: str, texts: List[str], format: str = 'json',
                      include_embeddings: bool = False, model: str = 'all-MiniLM-L6-v2') -> Dict[str, Any]:
    """
    Export vector store data to file.
    
    Args:
        file_path: Output file path
        texts: Texts to export
        format: Export format ('json', 'pickle', 'csv')
        include_embeddings: Whether to include embedding vectors
        model: Embedding model to use
        
    Returns:
        Dictionary with export results
    """
    try:
        embedding_config = EmbeddingConfig(model_name=model)
        tool = EmbeddingsVectorSearchTool(embedding_config)
        
        # Add texts first
        add_result = tool.add_texts(texts)
        if add_result['status'] != 'success':
            return add_result
            
        return tool.export_data(file_path, format, include_embeddings)
    except Exception as e:
        return {'status': 'error', 'message': f'Export failed: {str(e)}'}

@register_tool(tags=["embeddings", "import", "data", "persistence"])
def import_vector_data(file_path: str, format: str = 'json') -> Dict[str, Any]:
    """
    Import vector store data from file.
    
    Args:
        file_path: Input file path
        format: Import format ('json', 'pickle')
        
    Returns:
        Dictionary with import results
    """
    try:
        tool = EmbeddingsVectorSearchTool()
        return tool.import_data(file_path, format)
    except Exception as e:
        return {'status': 'error', 'message': f'Import failed: {str(e)}'}

@register_tool(tags=["embeddings", "stats", "analysis", "info"])
def get_embedding_stats(texts: Optional[List[str]] = None, model: str = 'all-MiniLM-L6-v2') -> Dict[str, Any]:
    """
    Get comprehensive statistics about embeddings and vector store.
    
    Args:
        texts: Optional texts to analyze
        model: Embedding model to use
        
    Returns:
        Dictionary with comprehensive statistics
    """
    try:
        embedding_config = EmbeddingConfig(model_name=model)
        tool = EmbeddingsVectorSearchTool(embedding_config)
        
        if texts:
            add_result = tool.add_texts(texts)
            if add_result['status'] != 'success':
                return add_result
                
        return tool.get_comprehensive_stats()
    except Exception as e:
        return {'status': 'error', 'message': f'Stats retrieval failed: {str(e)}'}

# Agent class wrapper
@register_tool(tags=["embeddings", "agent", "advanced", "wrapper"])
def create_embeddings_agent(embedding_model: str = 'all-MiniLM-L6-v2', 
                           index_type: str = 'simple') -> Dict[str, Any]:
    """
    Create an embeddings and vector search agent.
    
    Args:
        embedding_model: Model to use for embeddings
        index_type: Vector index type to use
        
    Returns:
        Dictionary with agent information and capabilities
    """
    try:
        embedding_config = EmbeddingConfig(model_name=embedding_model)
        search_config = VectorSearchConfig(index_type=index_type)
        agent = EmbeddingsVectorSearchAgent(embedding_config, search_config)
        
        return {
            'status': 'success',
            'agent_id': id(agent),
            'capabilities': agent.get_capabilities(),
            'config': {
                'embedding_model': embedding_model,
                'index_type': index_type
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Agent creation failed: {str(e)}'}