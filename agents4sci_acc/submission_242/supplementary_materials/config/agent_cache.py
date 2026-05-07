"""
Agent Cache System for SimulEval++

This module handles caching of generated freelancer profiles and job posts
to reduce API calls and improve consistency across simulation runs.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import threading

logger = logging.getLogger(__name__)

class AgentCache:
    """Cache system for freelancer profiles and job posts"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.freelancer_cache_file = self.cache_dir / "freelancer_profiles.json"
        self.client_cache_file = self.cache_dir / "client_profiles.json"
        self.job_templates_file = self.cache_dir / "job_templates.json"
        
        # Thread lock for concurrent access protection
        self._lock = threading.RLock()
    
    def save_freelancer_profiles(self, profiles: List[Dict], append_to_existing: bool = True):
        """Save generated freelancer profiles to cache
        
        Args:
            profiles: New profiles to save
            append_to_existing: If True, merge with existing cache. If False, overwrite.
        """
        try:
            existing_profiles = []
            if append_to_existing and self.freelancer_cache_file.exists():
                existing_data = self.load_freelancer_profiles()
                if existing_data:
                    existing_profiles = existing_data
            
            # Merge existing and new profiles, avoiding duplicates by name
            existing_names = {p.get('name', '') for p in existing_profiles}
            new_profiles = [p for p in profiles if p.get('name', '') not in existing_names]
            
            all_profiles = existing_profiles + new_profiles
            
            cache_data = {
                "generated_at": datetime.now().isoformat(),
                "profiles": all_profiles
            }
            
            with open(self.freelancer_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(new_profiles)} new freelancer profiles to cache (total: {len(all_profiles)})")
            
        except Exception as e:
            logger.error(f"Failed to save freelancer profiles: {e}")
    
    def load_freelancer_profiles(self, requested_count: int = None) -> Optional[List[Dict]]:
        """Load cached freelancer profiles with smart sampling
        
        Args:
            requested_count: Number of profiles needed. If None, returns all cached profiles.
                           If more than cached, returns all cached (caller should generate more).
                           If less than cached, returns random sample.
        """
        try:
            if not self.freelancer_cache_file.exists():
                return None
            
            with open(self.freelancer_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            all_profiles = cache_data.get("profiles", [])
            
            if not requested_count:
                logger.info(f"Loaded {len(all_profiles)} freelancer profiles from cache")
                return all_profiles
            
            if len(all_profiles) <= requested_count:
                logger.info(f"Loaded {len(all_profiles)} freelancer profiles from cache (requested {requested_count})")
                return all_profiles
            else:
                # Sample random subset
                sampled_profiles = random.sample(all_profiles, requested_count)
                logger.info(f"Sampled {len(sampled_profiles)} freelancer profiles from {len(all_profiles)} cached profiles")
                return sampled_profiles
            
        except Exception as e:
            logger.error(f"Failed to load freelancer profiles: {e}")
            return None
    
    def save_client_profiles(self, profiles: List[Dict], append_to_existing: bool = True):
        """Save generated client profiles to cache
        
        Args:
            profiles: New profiles to save
            append_to_existing: If True, merge with existing cache. If False, overwrite.
        """
        try:
            existing_profiles = []
            if append_to_existing and self.client_cache_file.exists():
                existing_data = self.load_client_profiles()
                if existing_data:
                    existing_profiles = existing_data
            
            # Merge existing and new profiles, avoiding duplicates by company name
            existing_names = {p.get('company_name', '') for p in existing_profiles}
            new_profiles = [p for p in profiles if p.get('company_name', '') not in existing_names]
            
            all_profiles = existing_profiles + new_profiles
            
            cache_data = {
                "generated_at": datetime.now().isoformat(),
                "profiles": all_profiles
            }
            
            with open(self.client_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(new_profiles)} new client profiles to cache (total: {len(all_profiles)})")
            
        except Exception as e:
            logger.error(f"Failed to save client profiles: {e}")
    
    def load_client_profiles(self, requested_count: int = None) -> Optional[List[Dict]]:
        """Load cached client profiles with smart sampling
        
        Args:
            requested_count: Number of profiles needed. If None, returns all cached profiles.
                           If more than cached, returns all cached (caller should generate more).
                           If less than cached, returns random sample.
        """
        try:
            if not self.client_cache_file.exists():
                return None
            
            with open(self.client_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            all_profiles = cache_data.get("profiles", [])
            
            if not requested_count:
                logger.info(f"Loaded {len(all_profiles)} client profiles from cache")
                return all_profiles
            
            if len(all_profiles) <= requested_count:
                logger.info(f"Loaded {len(all_profiles)} client profiles from cache (requested {requested_count})")
                return all_profiles
            else:
                # Sample random subset
                sampled_profiles = random.sample(all_profiles, requested_count)
                logger.info(f"Sampled {len(sampled_profiles)} client profiles from {len(all_profiles)} cached profiles")
                return sampled_profiles
            
        except Exception as e:
            logger.error(f"Failed to load client profiles: {e}")
            return None
    
    def save_job_templates(self, job_templates: List[Dict], append_to_existing: bool = True):
        """Save generated job post templates (thread-safe)
        
        Args:
            job_templates: New templates to save
            append_to_existing: If True, merge with existing cache. If False, overwrite.
        """
        with self._lock:  # Thread-safe access
            try:
                existing_templates = []
                if append_to_existing and self.job_templates_file.exists():
                    existing_data = self.load_job_templates()
                    if existing_data:
                        existing_templates = existing_data
                
                # Merge existing and new templates, avoiding duplicates by title and description
                existing_keys = {(t.get('title', ''), t.get('description', '')) for t in existing_templates}
                new_templates = [t for t in job_templates 
                               if (t.get('title', ''), t.get('description', '')) not in existing_keys]
                
                all_templates = existing_templates + new_templates
                
                cache_data = {
                    "generated_at": datetime.now().isoformat(),
                    "templates": all_templates
                }
                
                # Safe atomic write using a unique temp file name
                import tempfile
                import os
                
                # Create temp file in same directory as target
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix='.tmp', 
                    prefix='job_templates_',
                    dir=self.cache_dir
                )
                
                try:
                    # Write to temp file
                    with os.fdopen(temp_fd, 'w') as temp_file:
                        json.dump(cache_data, temp_file, indent=2)
                    
                    # Atomic move (works on most filesystems)
                    if os.name == 'nt':  # Windows
                        if self.job_templates_file.exists():
                            self.job_templates_file.unlink()
                    os.rename(temp_path, self.job_templates_file)
                    
                except Exception as atomic_error:
                    # Clean up temp file and fall back to direct write
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    logger.warning(f"Atomic write failed, using direct write: {atomic_error}")
                    with open(self.job_templates_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                
                logger.info(f"Saved {len(new_templates)} new job templates to cache (total: {len(all_templates)})")
                
            except Exception as e:
                logger.error(f"Failed to save job templates: {e}")
    
    def load_job_templates(self, requested_count: int = None) -> Optional[List[Dict]]:
        """Load cached job templates with smart sampling (thread-safe)
        
        Args:
            requested_count: Number of templates needed. If None, returns all cached templates.
                           If more than cached, returns all cached (caller should generate more).
                           If less than cached, returns random sample.
        """
        with self._lock:  # Thread-safe access
            try:
                if not self.job_templates_file.exists():
                    return None
                
                # Check file size to avoid reading empty/corrupted files
                if self.job_templates_file.stat().st_size == 0:
                    logger.warning("Job templates file is empty, returning None")
                    return None
                
                with open(self.job_templates_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        logger.warning("Job templates file content is empty, returning None")
                        return None
                
                try:
                    cache_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse job templates JSON: {e}")
                    # Move corrupted file and return None
                    corrupted_file = self.job_templates_file.with_suffix('.corrupted')
                    self.job_templates_file.rename(corrupted_file)
                    logger.error(f"Moved corrupted file to {corrupted_file}")
                    return None
                
                all_templates = cache_data.get("templates", [])
                
                if not requested_count:
                    logger.info(f"Loaded {len(all_templates)} job templates from cache")
                    return all_templates
                
                if len(all_templates) <= requested_count:
                    logger.info(f"Loaded {len(all_templates)} job templates from cache (requested {requested_count})")
                    return all_templates
                else:
                    # Sample random subset
                    sampled_templates = random.sample(all_templates, requested_count)
                    logger.info(f"Sampled {len(sampled_templates)} job templates from {len(all_templates)} cached templates")
                    return sampled_templates
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse job templates JSON (likely corrupted during write): {e}")
                return None
            except Exception as e:
                logger.error(f"Failed to load job templates: {e}")
                return None
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            if self.freelancer_cache_file.exists():
                self.freelancer_cache_file.unlink()
            if self.client_cache_file.exists():
                self.client_cache_file.unlink()
            if self.job_templates_file.exists():
                self.job_templates_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def cache_exists(self) -> Dict[str, bool]:
        """Check which cache files exist"""
        return {
            "freelancers": self.freelancer_cache_file.exists(),
            "clients": self.client_cache_file.exists(),
            "job_templates": self.job_templates_file.exists()
        }
