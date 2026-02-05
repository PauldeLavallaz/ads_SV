"""
Cache Manager for Product to Ads Node
Handles caching of generated prompts to avoid regeneration on re-runs
"""

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any, Tuple, Union, List

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


class CacheManager:
    """Manages prompt caching for efficient re-runs"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(
        self,
        node_id: str,
        product_url: str,
        profile: str,
        image_fingerprints: Dict[str, str],
        scraped_urls: Optional[List[str]] = None,
        force_regenerate: bool = False
    ) -> str:
        """
        Generate a unique cache key based on structural inputs
        
        Args:
            node_id: Unique node identifier
            product_url: Product page URL
            profile: Prompt profile ID
            image_fingerprints: Dictionary of image name -> fingerprint hash (user-provided images)
            scraped_urls: List of scraped image URLs (for auto mode)
            force_regenerate: If True, generates a unique key to bypass cache
            
        Returns:
            Cache key string
        """
        key_data = {
            "node_id": node_id,
            "product_url": product_url,
            "profile": profile,
            "images": image_fingerprints
        }
        
        if scraped_urls:
            key_data["scraped_urls"] = sorted(scraped_urls)
        
        if force_regenerate:
            import time
            key_data["bypass_timestamp"] = time.time()
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        
        return f"{node_id}_{key_hash}"
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get_image_fingerprint(self, image_data: Optional[Union[bytes, str]]) -> str:
        """
        Generate a fingerprint hash for image bytes or URL string
        
        Args:
            image_data: Image data bytes or URL string
            
        Returns:
            Hash string or empty string if no image
        """
        if not image_data:
            return ""
        
        if isinstance(image_data, str):
            return hashlib.md5(image_data.encode()).hexdigest()[:16]
        
        return hashlib.md5(image_data).hexdigest()[:16]
    
    def get_cached_prompt(
        self,
        node_id: str,
        product_url: str,
        profile: str,
        images: Dict[str, Optional[bytes]],
        scraped_urls: Optional[List[str]] = None,
        force_regenerate: bool = False
    ) -> Tuple[Optional[str], str]:
        """
        Retrieve a cached prompt if available
        
        Args:
            node_id: Unique node identifier
            product_url: Product page URL
            profile: Prompt profile ID
            images: Dictionary of image name -> bytes (user-provided images only)
            scraped_urls: List of scraped image URLs (for auto mode cache key)
            force_regenerate: If True, bypasses cache and returns None
            
        Returns:
            Tuple of (cached_prompt, status_message)
        """
        if force_regenerate:
            return None, "Cache bypassed - force regenerate requested"
        
        image_fingerprints = {
            name: self.get_image_fingerprint(data)
            for name, data in images.items()
        }
        
        cache_key = self._generate_cache_key(node_id, product_url, profile, image_fingerprints, scraped_urls)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None, "Cache miss - no cached prompt found"
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            max_age = 24 * 60 * 60
            if time.time() - cache_data.get("timestamp", 0) > max_age:
                os.remove(cache_path)
                return None, "Cache expired"
            
            if cache_data.get("fingerprints") != image_fingerprints:
                return None, "Cache invalidated - images changed"
            
            blueprint = cache_data.get("blueprint")
            if blueprint:
                # Verify it's a valid full blueprint (not legacy nano_banana_prompt only)
                if len(blueprint.keys()) > 1 or "nano_banana_prompt" not in blueprint:
                    return blueprint, "Blueprint served from cache"
                else:
                    # Legacy cache entry with only nano_banana_prompt - invalidate
                    return None, "Cache invalidated - legacy format"
            
            # Legacy cache with nano_banana_prompt at top level - invalidate
            prompt = cache_data.get("nano_banana_prompt")
            if prompt:
                return None, "Cache invalidated - legacy format"
            
            return None, "Cache hit but blueprint empty"
            
        except Exception as e:
            return None, f"Cache read error: {e}"
    
    def save_prompt(
        self,
        node_id: str,
        product_url: str,
        profile: str,
        images: Dict[str, Optional[bytes]],
        blueprint: Dict[str, Any],
        scraped_urls: Optional[List[str]] = None,
        product_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a generated blueprint to cache
        
        Args:
            node_id: Unique node identifier
            product_url: Product page URL
            profile: Prompt profile ID
            images: Dictionary of image name -> bytes (user-provided images only)
            blueprint: The complete blueprint dict to cache
            scraped_urls: List of scraped image URLs (for auto mode cache key)
            product_data: Optional product data to cache
            
        Returns:
            True if saved successfully
        """
        image_fingerprints = {
            name: self.get_image_fingerprint(data)
            for name, data in images.items()
        }
        
        cache_key = self._generate_cache_key(node_id, product_url, profile, image_fingerprints, scraped_urls)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            "timestamp": time.time(),
            "node_id": node_id,
            "product_url": product_url,
            "profile": profile,
            "fingerprints": image_fingerprints,
            "scraped_urls": scraped_urls,
            "blueprint": blueprint,
            "product_data": product_data
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Cache write error: {e}")
            return False
    
    def invalidate_cache(self, node_id: str) -> int:
        """
        Invalidate all cache entries for a node
        
        Args:
            node_id: Unique node identifier
            
        Returns:
            Number of cache entries removed
        """
        count = 0
        prefix = f"{node_id}_"
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(prefix):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
        except Exception as e:
            print(f"Cache invalidation error: {e}")
        
        return count
    
    def clear_all_cache(self) -> int:
        """
        Clear all cached prompts
        
        Returns:
            Number of cache entries removed
        """
        count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
        except Exception as e:
            print(f"Cache clear error: {e}")
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "oldest_entry": None,
            "newest_entry": None
        }
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    stats["total_entries"] += 1
                    stats["total_size_bytes"] += os.path.getsize(filepath)
                    
                    mtime = os.path.getmtime(filepath)
                    if stats["oldest_entry"] is None or mtime < stats["oldest_entry"]:
                        stats["oldest_entry"] = mtime
                    if stats["newest_entry"] is None or mtime > stats["newest_entry"]:
                        stats["newest_entry"] = mtime
                        
        except Exception as e:
            print(f"Cache stats error: {e}")
        
        return stats
    
    def _get_image_cache_dir(self) -> str:
        """Get directory for cached images"""
        img_cache = os.path.join(self.cache_dir, "images")
        os.makedirs(img_cache, exist_ok=True)
        return img_cache
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        elif ".webp" in url.lower():
            ext = ".webp"
        return f"{url_hash}{ext}"
    
    def get_cached_image(self, url: str) -> Optional[bytes]:
        """
        Get cached image bytes by URL
        
        Args:
            url: Image URL
            
        Returns:
            Image bytes if cached, None otherwise
        """
        img_cache_dir = self._get_image_cache_dir()
        filename = self._url_to_filename(url)
        filepath = os.path.join(img_cache_dir, filename)
        
        if os.path.exists(filepath):
            try:
                max_age = 7 * 24 * 60 * 60
                if time.time() - os.path.getmtime(filepath) > max_age:
                    os.remove(filepath)
                    return None
                
                with open(filepath, 'rb') as f:
                    return f.read()
            except Exception:
                return None
        return None
    
    def save_image_to_cache(self, url: str, image_bytes: bytes) -> bool:
        """
        Save image bytes to cache
        
        Args:
            url: Image URL (used as key)
            image_bytes: Image data
            
        Returns:
            True if saved successfully
        """
        if not image_bytes:
            return False
        
        img_cache_dir = self._get_image_cache_dir()
        filename = self._url_to_filename(url)
        filepath = os.path.join(img_cache_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            return True
        except Exception:
            return False
