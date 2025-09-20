"""
Data loading and preprocessing module for camera feeds, encoder, and decoder data.
"""
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class DataLoader:
    """Handles loading and preprocessing of all data files."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.camera_feeds = None
        self.encoder_params = None
        self.decoder_params = None
        self.encoder_schema = None
        self.decoder_schema = None
        self.table_defs = None
        
    def load_all_data(self) -> Dict[str, Any]:
        """Load all data files and return as dictionary."""
        data = {}
        
        # Load camera feeds
        self.camera_feeds = pd.read_csv(self.data_dir / "Table_feeds_v2.csv")
        data['camera_feeds'] = self.camera_feeds
        
        # Load encoder parameters
        with open(self.data_dir / "encoder_params.json", 'r') as f:
            self.encoder_params = json.load(f)
        data['encoder_params'] = self.encoder_params
        
        # Load decoder parameters
        with open(self.data_dir / "decoder_params.json", 'r') as f:
            self.decoder_params = json.load(f)
        data['decoder_params'] = self.decoder_params
        
        # Load schemas
        with open(self.data_dir / "encoder_schema.json", 'r') as f:
            self.encoder_schema = json.load(f)
        data['encoder_schema'] = self.encoder_schema
        
        with open(self.data_dir / "decoder_schema.json", 'r') as f:
            self.decoder_schema = json.load(f)
        data['decoder_schema'] = self.decoder_schema
        
        # Load table definitions
        self.table_defs = pd.read_csv(self.data_dir / "Table_defs_v2.csv")
        data['table_defs'] = self.table_defs
        
        return data
    
    def get_camera_feeds(self) -> pd.DataFrame:
        """Get camera feeds dataframe."""
        if self.camera_feeds is None:
            self.load_all_data()
        return self.camera_feeds
    
    def get_encoder_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        if self.encoder_params is None:
            self.load_all_data()
        return self.encoder_params
    
    def get_decoder_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        if self.decoder_params is None:
            self.load_all_data()
        return self.decoder_params
    
    def get_feeds_by_theater(self, theater: str) -> pd.DataFrame:
        """Filter camera feeds by theater."""
        feeds = self.get_camera_feeds()
        return feeds[feeds['THEATER'] == theater.upper()]
    
    def get_feeds_by_codec(self, codec: str) -> pd.DataFrame:
        """Filter camera feeds by codec."""
        feeds = self.get_camera_feeds()
        return feeds[feeds['CODEC'] == codec.upper()]
    
    def get_feeds_by_resolution(self, min_width: int = None, min_height: int = None, 
                               max_width: int = None, max_height: int = None) -> pd.DataFrame:
        """Filter camera feeds by resolution range."""
        feeds = self.get_camera_feeds()
        
        if min_width:
            feeds = feeds[feeds['RES_W'] >= min_width]
        if min_height:
            feeds = feeds[feeds['RES_H'] >= min_height]
        if max_width:
            feeds = feeds[feeds['RES_W'] <= max_width]
        if max_height:
            feeds = feeds[feeds['RES_H'] <= max_height]
            
        return feeds
    
    def get_feeds_by_latency(self, max_latency: int = None, min_latency: int = None) -> pd.DataFrame:
        """Filter camera feeds by latency range."""
        feeds = self.get_camera_feeds()
        
        if max_latency:
            feeds = feeds[feeds['LAT_MS'] <= max_latency]
        if min_latency:
            feeds = feeds[feeds['LAT_MS'] >= min_latency]
            
        return feeds
    
    def get_high_quality_feeds(self) -> pd.DataFrame:
        """Get feeds with high quality metrics (4K resolution, modern codec, low latency)."""
        feeds = self.get_camera_feeds()
        
        # Define quality criteria
        high_res = (feeds['RES_W'] >= 1920) & (feeds['RES_H'] >= 1080)
        modern_codec = feeds['CODEC'].isin(['H265', 'AV1', 'VP9'])
        low_latency = feeds['LAT_MS'] <= 500
        
        # Calculate quality score
        feeds = feeds.copy()
        feeds['quality_score'] = (
            (high_res.astype(int) * 3) + 
            (modern_codec.astype(int) * 2) + 
            (low_latency.astype(int) * 1)
        )
        
        return feeds.sort_values('quality_score', ascending=False)
    
    def get_feeds_by_model(self, model_tag: str) -> pd.DataFrame:
        """Filter camera feeds by analytics model."""
        feeds = self.get_camera_feeds()
        return feeds[feeds['MODL_TAG'] == model_tag]
    
    def get_encrypted_feeds(self, encrypted: bool = True) -> pd.DataFrame:
        """Filter camera feeds by encryption status."""
        feeds = self.get_camera_feeds()
        return feeds[feeds['ENCR'] == encrypted]
    
    def get_civilian_safe_feeds(self, safe: bool = True) -> pd.DataFrame:
        """Filter camera feeds by civilian safety compliance."""
        feeds = self.get_camera_feeds()
        return feeds[feeds['CIV_OK'] == safe]
