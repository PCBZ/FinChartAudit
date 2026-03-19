# data/download_misviz_data.py

import json
from pathlib import Path
import requests
from tqdm import tqdm


def download_misviz_annotations():
    """Download misviz.json from official repository."""
    
    json_dir = Path('data/misviz')
    json_path = json_dir / 'misviz.json'
    
    # Already exists, skip
    if json_path.exists():
        print(f"✓ {json_path} already exists")
        return json_path
    
    # Create directory
    json_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading misviz.json...")
    
    # Download from GitHub
    url = "https://raw.githubusercontent.com/UKPLab/arxiv2025-misviz/main/data/misviz/misviz.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(json_path, 'w') as f:
            f.write(response.text)
        
        print(f"✓ Downloaded to {json_path}")
        return json_path
        
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        print(f"Please manually download from: {url}")
        return None


def verify_misviz_structure():
    """Verify Misviz data structure."""
    
    json_path = Path('data/misviz/misviz.json')
    
    if not json_path.exists():
        print(f"✗ {json_path} not found")
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} samples from {json_path}")
        
        # Check structure
        if len(data) > 0:
            first_sample = data[0]
            required_fields = ['image_path', 'chart_type', 'misleader']
            
            for field in required_fields:
                if field not in first_sample:
                    print(f"✗ Missing field: {field}")
                    return False
            
            print(f"✓ Data structure valid")
            print(f"  - Sample fields: {list(first_sample.keys())}")
            print(f"  - Splits: {set(item.get('split') for item in data)}")
            
            return True
    
    except Exception as e:
        print(f"✗ Error reading {json_path}: {e}")
        return False


def setup_misviz():
    """Setup Misviz dataset (download if needed)."""
    
    print("\n" + "="*60)
    print("MISVIZ DATASET SETUP")
    print("="*60 + "\n")
    
    # Create directory structure
    image_dir = Path('data/misviz/images')
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory structure created: {image_dir}")
    
    # Download annotations if needed
    json_path = download_misviz_annotations()
    
    if json_path is None:
        return False
    
    # Verify structure
    if not verify_misviz_structure():
        return False
    
    return True


if __name__ == "__main__":
    success = setup_misviz()
    exit(0 if success else 1)