"""
Brief Manager - Handles saving and loading campaign briefs
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class BriefManager:
    """Manages campaign brief storage and retrieval"""
    
    def __init__(self, output_base_path: Optional[str] = None):
        """
        Initialize the BriefManager
        
        Args:
            output_base_path: Base path for saving briefs. 
                            Defaults to ComfyUI output folder.
        """
        if output_base_path:
            self.output_base = output_base_path
        else:
            self.output_base = self._get_comfyui_output_path()
        
        self.briefs_folder = os.path.join(self.output_base, "campaign_briefs")
        os.makedirs(self.briefs_folder, exist_ok=True)
    
    def _get_comfyui_output_path(self) -> str:
        """Get ComfyUI output folder path"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        comfyui_output = os.path.join(parent_dir, "output")
        
        if not os.path.exists(comfyui_output):
            comfyui_output = os.path.join(current_dir, "output")
            os.makedirs(comfyui_output, exist_ok=True)
        
        return comfyui_output
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()[:50]
    
    def save_brief(
        self,
        brand_name: str,
        brief_data: Dict[str, Any],
        product_data: Optional[Dict[str, Any]] = None,
        product_url: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Save a campaign brief to the output folder
        
        Args:
            brand_name: Name of the brand
            brief_data: The generated brief JSON
            product_data: Optional product data to include
            product_url: Optional product URL
            
        Returns:
            Tuple of (file_path, log_message)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_brand = self._sanitize_filename(brand_name)
        
        filename = f"{safe_brand}_{timestamp}_brief.json"
        filepath = os.path.join(self.briefs_folder, filename)
        
        output_data = {
            "metadata": {
                "brand": brand_name,
                "generated_at": datetime.now().isoformat(),
                "product_url": product_url
            },
            "brief": brief_data
        }
        
        if product_data:
            output_data["product_data"] = product_data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log_message = f"Brief saved: {filepath}"
        return filepath, log_message
    
    def load_brief(self, filepath: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Load a previously saved brief
        
        Args:
            filepath: Path to the brief JSON file
            
        Returns:
            Tuple of (brief_data, log_message)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            brief = data.get("brief", data)
            return brief, f"Brief loaded from: {filepath}"
            
        except FileNotFoundError:
            return None, f"Brief not found: {filepath}"
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON in brief: {e}"
        except Exception as e:
            return None, f"Error loading brief: {e}"
    
    def list_briefs(self, brand_filter: Optional[str] = None) -> list:
        """
        List all saved briefs, optionally filtered by brand
        
        Args:
            brand_filter: Optional brand name to filter by
            
        Returns:
            List of brief file paths
        """
        briefs = []
        
        if not os.path.exists(self.briefs_folder):
            return briefs
        
        for filename in os.listdir(self.briefs_folder):
            if filename.endswith("_brief.json"):
                if brand_filter:
                    safe_filter = self._sanitize_filename(brand_filter)
                    if not filename.lower().startswith(safe_filter.lower()):
                        continue
                briefs.append(os.path.join(self.briefs_folder, filename))
        
        briefs.sort(reverse=True)
        return briefs
    
    def get_latest_brief(self, brand_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Get the most recent brief for a brand
        
        Args:
            brand_name: Name of the brand
            
        Returns:
            Tuple of (brief_data, log_message)
        """
        briefs = self.list_briefs(brand_filter=brand_name)
        
        if not briefs:
            return None, f"No briefs found for brand: {brand_name}"
        
        return self.load_brief(briefs[0])
    
    def generate_execution_log(
        self,
        brand_name: str,
        brief_data: Dict[str, Any],
        phase_results: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Generate and save an execution log alongside the brief
        
        Args:
            brand_name: Name of the brand
            brief_data: The generated brief
            phase_results: Results from each phase execution
            
        Returns:
            Tuple of (log_filepath, message)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_brand = self._sanitize_filename(brand_name)
        
        log_filename = f"{safe_brand}_{timestamp}_execution_log.txt"
        log_filepath = os.path.join(self.briefs_folder, log_filename)
        
        log_lines = [
            "=" * 60,
            f"CAMPAIGN BRIEF EXECUTION LOG",
            f"Brand: {brand_name}",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            "BRIEF SUMMARY",
            "-" * 40
        ]
        
        if isinstance(brief_data, dict):
            campaign_brief = brief_data.get("campaign_brief", brief_data)
            
            if "1_strategic_objective" in campaign_brief:
                obj = campaign_brief["1_strategic_objective"]
                log_lines.append(f"Strategic Objective: {obj.get('campaign_purpose', 'N/A')}")
            
            if "2_central_message" in campaign_brief:
                msg = campaign_brief["2_central_message"]
                log_lines.append(f"Central Message: {msg.get('core_idea', 'N/A')}")
            
            if "3_visual_tone_of_voice" in campaign_brief:
                tone = campaign_brief["3_visual_tone_of_voice"]
                log_lines.append(f"Visual Tone: {tone.get('primary_tone', 'N/A')}")
        
        log_lines.extend([
            "",
            "PHASE EXECUTION RESULTS",
            "-" * 40
        ])
        
        for phase_name, result in phase_results.items():
            log_lines.append(f"\n{phase_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    log_lines.append(f"  {key}: {value}")
            else:
                log_lines.append(f"  {result}")
        
        log_lines.extend([
            "",
            "=" * 60,
            "END OF LOG",
            "=" * 60
        ])
        
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))
        
        return log_filepath, f"Execution log saved: {log_filepath}"
