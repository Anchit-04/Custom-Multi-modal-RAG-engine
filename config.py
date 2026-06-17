from dataclasses import dataclass

@dataclass
class Config:
    # Ingestion
    frame_fps: int = 2          # extract 2 frames/sec — sufficient for chunking
    audio_sr: int = 16000       # AST expects 16kHz
    
    # Chunking
    min_chunk_sec: float = 3.0
    max_chunk_sec: float = 45.0
    overlap_sec: float = 2.0
    min_semantic_threshold: float = 0.15   # cosine distance to confirm visual cut
    whisper_model: str = "small"       # 1.5GB VRAM, good enough for timestamps
    
    # Encoding
    frames_per_chunk: int = 8          # sampled uniformly from chunk duration
    shared_dim: int = 512
    temporal_layers: int = 2
    temporal_heads: int = 8
    
    # Optical flow
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    
    # Indexing
    qdrant_path: str = "./qdrant_store"
    micro_collection: str = "micro_chunks"
    meso_collection: str = "meso_chunks"
    meso_group_size: int = 6           # 6 micro chunks → 1 meso chunk
    
    # Retrieval
    micro_top_k: int = 20
    meso_top_k: int = 5
    rerank_top_k: int = 5

cfg = Config()