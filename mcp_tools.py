"""
MCP (Model Context Protocol) tools for data retrieval, filtering, and analysis.
These tools provide structured access to camera feed data for the agentic system.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from .data_loader import DataLoader


class MCPTools:
    """MCP tools for camera feed data operations."""
    
    def __init__(self, data_dir: str = "."):
        self.data_loader = DataLoader(data_dir)
        self.data_loader.load_all_data()
    
    def get_all_camera_feeds(self) -> Dict[str, Any]:
        """Get all camera feeds data.
        
        Returns:
            Dict containing all camera feeds with metadata
        """
        feeds = self.data_loader.get_camera_feeds()
        return {
            "feeds": feeds.to_dict('records'),
            "total_count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def filter_by_theater(self, theater: str) -> Dict[str, Any]:
        """Filter camera feeds by geographic theater.
        
        Args:
            theater: Theater code (CONUS, PAC, EUR, ME, AFR, ARC)
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_feeds_by_theater(theater)
        return {
            "feeds": feeds.to_dict('records'),
            "theater": theater,
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def filter_by_codec(self, codec: str) -> Dict[str, Any]:
        """Filter camera feeds by video codec.
        
        Args:
            codec: Codec type (H264, H265, AV1, VP9, MPEG2)
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_feeds_by_codec(codec)
        return {
            "feeds": feeds.to_dict('records'),
            "codec": codec,
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def filter_by_resolution(self, min_width: Optional[int] = None, 
                           min_height: Optional[int] = None,
                           max_width: Optional[int] = None,
                           max_height: Optional[int] = None) -> Dict[str, Any]:
        """Filter camera feeds by resolution range.
        
        Args:
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_feeds_by_resolution(
            min_width, min_height, max_width, max_height
        )
        return {
            "feeds": feeds.to_dict('records'),
            "resolution_filter": {
                "min_width": min_width,
                "min_height": min_height,
                "max_width": max_width,
                "max_height": max_height
            },
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def filter_by_latency(self, max_latency: Optional[int] = None,
                         min_latency: Optional[int] = None) -> Dict[str, Any]:
        """Filter camera feeds by latency range.
        
        Args:
            max_latency: Maximum latency in milliseconds
            min_latency: Minimum latency in milliseconds
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_feeds_by_latency(max_latency, min_latency)
        return {
            "feeds": feeds.to_dict('records'),
            "latency_filter": {
                "max_latency": max_latency,
                "min_latency": min_latency
            },
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def get_high_quality_feeds(self) -> Dict[str, Any]:
        """Get camera feeds with high quality metrics.
        
        Quality is determined by:
        - Resolution (4K = 3 points, 1080p = 2 points, 720p = 1 point)
        - Codec (H265/AV1/VP9 = 2 points, H264 = 1 point, MPEG2 = 0 points)
        - Latency (≤500ms = 1 point, >500ms = 0 points)
        
        Returns:
            Dict containing high quality feeds sorted by quality score
        """
        feeds = self.data_loader.get_high_quality_feeds()
        return {
            "feeds": feeds.to_dict('records'),
            "count": len(feeds),
            "quality_metrics": {
                "resolution_weight": 3,
                "codec_weight": 2,
                "latency_weight": 1
            },
            "columns": list(feeds.columns)
        }
    
    def filter_by_model(self, model_tag: str) -> Dict[str, Any]:
        """Filter camera feeds by analytics model.
        
        Args:
            model_tag: Model tag (e.g., Viper-VL, Hydra-ISR, Raptor-Det)
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_feeds_by_model(model_tag)
        return {
            "feeds": feeds.to_dict('records'),
            "model_tag": model_tag,
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def filter_by_encryption(self, encrypted: bool = True) -> Dict[str, Any]:
        """Filter camera feeds by encryption status.
        
        Args:
            encrypted: True for encrypted feeds, False for unencrypted
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_encrypted_feeds(encrypted)
        return {
            "feeds": feeds.to_dict('records'),
            "encrypted": encrypted,
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def filter_by_civilian_safety(self, safe: bool = True) -> Dict[str, Any]:
        """Filter camera feeds by civilian safety compliance.
        
        Args:
            safe: True for civilian-safe feeds, False for others
            
        Returns:
            Dict containing filtered feeds and metadata
        """
        feeds = self.data_loader.get_civilian_safe_feeds(safe)
        return {
            "feeds": feeds.to_dict('records'),
            "civilian_safe": safe,
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
    
    def get_encoder_parameters(self) -> Dict[str, Any]:
        """Get video encoder parameters.
        
        Returns:
            Dict containing encoder configuration
        """
        return {
            "encoder_params": self.data_loader.get_encoder_params(),
            "description": "Video encoding configuration for all camera feeds"
        }
    
    def get_decoder_parameters(self) -> Dict[str, Any]:
        """Get video decoder parameters.
        
        Returns:
            Dict containing decoder configuration
        """
        return {
            "decoder_params": self.data_loader.get_decoder_params(),
            "description": "Video decoding configuration for all camera feeds"
        }
    
    def analyze_theater_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of camera feeds across theaters.
        
        Returns:
            Dict containing theater statistics
        """
        feeds = self.data_loader.get_camera_feeds()
        theater_counts = feeds['THEATER'].value_counts().to_dict()
        
        return {
            "theater_distribution": theater_counts,
            "total_feeds": len(feeds),
            "theaters": list(theater_counts.keys())
        }
    
    def analyze_codec_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of video codecs.
        
        Returns:
            Dict containing codec statistics
        """
        feeds = self.data_loader.get_camera_feeds()
        codec_counts = feeds['CODEC'].value_counts().to_dict()
        
        return {
            "codec_distribution": codec_counts,
            "total_feeds": len(feeds),
            "codecs": list(codec_counts.keys())
        }
    
    def analyze_resolution_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of video resolutions.
        
        Returns:
            Dict containing resolution statistics
        """
        feeds = self.data_loader.get_camera_feeds()
        
        # Group by common resolution categories
        resolution_categories = {
            '4K (3840x2160)': (feeds['RES_W'] == 3840) & (feeds['RES_H'] == 2160),
            '1440p (2560x1440)': (feeds['RES_W'] == 2560) & (feeds['RES_H'] == 1440),
            '1080p (1920x1080)': (feeds['RES_W'] == 1920) & (feeds['RES_H'] == 1080),
            '720p (1280x720)': (feeds['RES_W'] == 1280) & (feeds['RES_H'] == 720),
            '480p (640x480)': (feeds['RES_W'] == 640) & (feeds['RES_H'] == 480)
        }
        
        resolution_counts = {}
        for category, condition in resolution_categories.items():
            resolution_counts[category] = condition.sum()
        
        return {
            "resolution_distribution": resolution_counts,
            "total_feeds": len(feeds),
            "unique_resolutions": len(feeds[['RES_W', 'RES_H']].drop_duplicates())
        }
    
    def get_feed_by_id(self, feed_id: str) -> Dict[str, Any]:
        """Get specific camera feed by ID.
        
        Args:
            feed_id: Camera feed ID (e.g., FD-ML64LG)
            
        Returns:
            Dict containing feed data or error message
        """
        feeds = self.data_loader.get_camera_feeds()
        feed = feeds[feeds['FEED_ID'] == feed_id]
        
        if len(feed) == 0:
            return {
                "error": f"Feed ID '{feed_id}' not found",
                "available_feeds": feeds['FEED_ID'].tolist()[:10]  # Show first 10 as examples
            }
        
        return {
            "feed": feed.iloc[0].to_dict(),
            "found": True
        }
    
    def search_feeds(self, **filters) -> Dict[str, Any]:
        """Advanced search with multiple filters.
        
        Args:
            **filters: Any combination of filters (theater, codec, min_width, etc.)
            
        Returns:
            Dict containing filtered feeds and applied filters
        """
        feeds = self.data_loader.get_camera_feeds()
        applied_filters = {}
        
        # Apply each filter
        if 'theater' in filters:
            feeds = feeds[feeds['THEATER'] == filters['theater'].upper()]
            applied_filters['theater'] = filters['theater']
        
        if 'codec' in filters:
            feeds = feeds[feeds['CODEC'] == filters['codec'].upper()]
            applied_filters['codec'] = filters['codec']
        
        if 'min_width' in filters:
            feeds = feeds[feeds['RES_W'] >= filters['min_width']]
            applied_filters['min_width'] = filters['min_width']
        
        if 'min_height' in filters:
            feeds = feeds[feeds['RES_H'] >= filters['min_height']]
            applied_filters['min_height'] = filters['min_height']
        
        if 'max_latency' in filters:
            feeds = feeds[feeds['LAT_MS'] <= filters['max_latency']]
            applied_filters['max_latency'] = filters['max_latency']
        
        if 'encrypted' in filters:
            feeds = feeds[feeds['ENCR'] == filters['encrypted']]
            applied_filters['encrypted'] = filters['encrypted']
        
        if 'civilian_safe' in filters:
            feeds = feeds[feeds['CIV_OK'] == filters['civilian_safe']]
            applied_filters['civilian_safe'] = filters['civilian_safe']
        
        return {
            "feeds": feeds.to_dict('records'),
            "applied_filters": applied_filters,
            "count": len(feeds),
            "columns": list(feeds.columns)
        }
