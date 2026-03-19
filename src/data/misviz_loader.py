# src/data/misviz_loader.py

import json
from pathlib import Path


class MisvizDataset:
    """Dataset loader for Misviz benchmark (2,604 visualizations with 12 misleader types)."""
    
    def __init__(
        self,
        json_path: str = 'data/misviz/misviz.json',
        image_dir: str = 'data/misviz/images',
        split: str | None = None,
    ) -> None:
        """
        Args:
            json_path: Path to misviz.json annotations file
            image_dir: Path to directory containing visualization images
            split: Dataset split ('train', 'val', 'test') or None for all data
        """
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.split = split

        # Create directories if they don't exist
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.json_path}")
        
        with open(self.json_path, 'r') as f:
            all_data = json.load(f)
        
        if split is not None:
            self.data = [item for item in all_data if item.get('split') == split]
        else:
            self.data = all_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict[str, object]:
        item = self.data[idx]
        return {
            'id': idx,
            'image_path': str(self.image_dir / item['image_path']),
            'chart_type': item.get('chart_type', []),
            'misleader': item.get('misleader', []),
            'bbox': item.get('bbox', []),
            'split': item.get('split'),
        }
    
    def get_all_misleaders(self) -> list[str]:
        """Extract all unique misleader types from the dataset."""
        misleaders = set()
        for item in self.data:
            misleaders.update(item.get('misleader', []))
        return sorted(list(misleaders))
    
    def get_misleader_distribution(self) -> dict[str, int]:
        """Get frequency of each misleader type."""
        dist = {}
        for item in self.data:
            for m in item.get('misleader', []):
                dist[m] = dist.get(m, 0) + 1
        return dist
    
    def get_by_misleader(self, misleader_type: str) -> list[dict[str, object]]:
        """Get all samples containing a specific misleader type."""
        return [item for item in self.data if misleader_type in item.get('misleader', [])]
    
    def print_statistics(self):
        """Print dataset statistics."""
        print(f"\nTotal samples: {len(self)}")
        
        print("\nMisleader distribution:")
        for m, count in sorted(self.get_misleader_distribution().items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"  {m}: {count}")


if __name__ == "__main__":
    dataset = MisvizDataset()
    dataset.print_statistics()
    
    sample = dataset[0]
    print(f"\nExample sample:")
    print(f"  Image: {sample['image_path']}")
    print(f"  Misleaders: {sample['misleader']}")