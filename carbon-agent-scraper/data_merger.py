#!/usr/bin/env python
"""
EirGrid Data Merger Utility
Intelligently merges data files, replacing forecast values with actual values
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging


class DataMerger:
    """Handles intelligent merging of EirGrid data files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge_files(self, file_paths: List[str], output_path: Optional[str] = None) -> str:
        """
        Merge multiple data files intelligently
        
        Args:
            file_paths: List of file paths to merge
            output_path: Optional output path (defaults to updating first file)
            
        Returns:
            Path to merged file
        """
        
        if not file_paths:
            raise ValueError("No files provided to merge")
        
        # Load all files
        all_data = []
        for path in file_paths:
            self.logger.info(f"Loading {path}")
            with open(path, 'r') as f:
                all_data.append(json.load(f))
        
        # Start with first file as base
        merged = all_data[0].copy()
        
        # Merge remaining files
        for data in all_data[1:]:
            merged = self._merge_two_datasets(merged, data)
        
        # Update metadata
        merged['metadata']['last_updated'] = datetime.now().isoformat()
        merged['metadata']['merged_from'] = file_paths
        
        # Save result
        output = output_path or file_paths[0]
        with open(output, 'w') as f:
            json.dump(merged, f, indent=2)
        
        self.logger.info(f"Merged data saved to {output}")
        return output
    
    def _merge_two_datasets(self, base: Dict, new: Dict) -> Dict:
        """Merge two datasets intelligently"""
        
        # Get time series data
        base_ts = base.get('data', {}).get('time_series', [])
        new_ts = new.get('data', {}).get('time_series', [])
        
        # Create lookup by time
        merged_by_time = {point['time']: point for point in base_ts}
        
        # Process new data
        for new_point in new_ts:
            time_key = new_point['time']
            
            if time_key in merged_by_time:
                # Merge logic: prefer actual over forecast
                existing = merged_by_time[time_key]
                
                # Check if we're replacing forecast with actual
                if existing.get('is_forecast', False) and not new_point.get('is_forecast', False):
                    self.logger.debug(f"Replacing forecast with actual for {time_key}")
                    merged_by_time[time_key] = new_point
                
                # Update if values differ
                elif existing.get('value') != new_point.get('value'):
                    # Prefer non-forecast values
                    if not new_point.get('is_forecast', False):
                        merged_by_time[time_key] = new_point
                    elif existing.get('is_forecast', False):
                        # Both are forecasts, take the newer one
                        merged_by_time[time_key] = new_point
                
                # Merge forecast values if present
                if 'forecast_value' in new_point and 'forecast_value' not in existing:
                    merged_by_time[time_key]['forecast_value'] = new_point['forecast_value']
                    
            else:
                # Add new time point
                merged_by_time[time_key] = new_point
        
        # Convert back to sorted list
        merged_ts = list(merged_by_time.values())
        merged_ts.sort(key=lambda x: self._parse_time(x['time']))
        
        base['data']['time_series'] = merged_ts
        
        # Update metadata
        if 'metadata' in new:
            # Merge regions if different
            base_region = base.get('metadata', {}).get('region', 'unknown')
            new_region = new.get('metadata', {}).get('region', 'unknown')
            
            if base_region != new_region and new_region != 'unknown':
                if base_region == 'unknown':
                    base['metadata']['region'] = new_region
                else:
                    base['metadata']['region'] = 'all'  # Mixed regions
        
        return base
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse various time formats"""
        formats = [
            '%d-%b-%Y %H:%M:%S',
            '%d %B %Y %H:%M',
            '%d %b %Y %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str.split('.')[0], fmt)  # Remove milliseconds
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse time: {time_str}")
        return datetime.now()
    
    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a data file and return statistics"""
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        time_series = data.get('data', {}).get('time_series', [])
        
        stats = {
            'file': file_path,
            'total_points': len(time_series),
            'actual_values': 0,
            'forecast_values': 0,
            'null_values': 0,
            'time_range': None
        }
        
        if time_series:
            # Count types
            for point in time_series:
                if point.get('value') is None:
                    stats['null_values'] += 1
                elif point.get('is_forecast', False):
                    stats['forecast_values'] += 1
                else:
                    stats['actual_values'] += 1
            
            # Get time range
            times = [self._parse_time(p['time']) for p in time_series]
            stats['time_range'] = {
                'start': min(times).isoformat(),
                'end': max(times).isoformat()
            }
        
        return stats
    
    def clean_duplicates(self, directory: str, area: Optional[str] = None):
        """
        Find and merge duplicate files for the same area/date
        
        Args:
            directory: Directory to scan
            area: Optional specific area to process
        """
        
        dir_path = Path(directory)
        
        # Group files by area and date
        file_groups = {}
        
        for file_path in dir_path.glob("*.json"):
            # Parse filename (expected format: area_region_YYYYMMDD.json)
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                file_area = parts[0]
                date_part = parts[-1]
                
                if area and file_area != area:
                    continue
                
                key = f"{file_area}_{date_part}"
                if key not in file_groups:
                    file_groups[key] = []
                file_groups[key].append(str(file_path))
        
        # Process groups with multiple files
        for key, files in file_groups.items():
            if len(files) > 1:
                self.logger.info(f"Found {len(files)} files for {key}")
                
                # Analyze files to determine best merge strategy
                file_stats = [self.analyze_file(f) for f in files]
                
                # Sort by number of actual values (descending)
                files_sorted = sorted(
                    zip(files, file_stats),
                    key=lambda x: x[1]['actual_values'],
                    reverse=True
                )
                
                # Use file with most actual values as base
                base_file = files_sorted[0][0]
                self.logger.info(f"Using {Path(base_file).name} as base (most actual values)")
                
                # Merge all files
                self.merge_files([f[0] for f in files_sorted], base_file)
                
                # Optionally remove other files
                response = input(f"Remove {len(files)-1} duplicate files? (y/n): ")
                if response.lower() == 'y':
                    for f, _ in files_sorted[1:]:
                        Path(f).unlink()
                        self.logger.info(f"Removed {f}")


def main():
    """CLI for data merger utility"""
    
    parser = argparse.ArgumentParser(
        description="Merge EirGrid data files intelligently",
        epilog="""
Examples:
  # Merge specific files
  python data_merger.py file1.json file2.json -o merged.json
  
  # Clean duplicates in a directory
  python data_merger.py --clean-dir ./data
  
  # Analyze a file
  python data_merger.py --analyze file.json
        """
    )
    
    parser.add_argument('files', nargs='*',
                       help='Files to merge')
    
    parser.add_argument('-o', '--output',
                       help='Output file path')
    
    parser.add_argument('--clean-dir',
                       help='Clean duplicate files in directory')
    
    parser.add_argument('--area',
                       help='Specific area to process when cleaning')
    
    parser.add_argument('--analyze',
                       help='Analyze a data file')
    
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    merger = DataMerger()
    
    try:
        if args.analyze:
            # Analyze mode
            stats = merger.analyze_file(args.analyze)
            print("\nFile Analysis:")
            print("-" * 40)
            for key, value in stats.items():
                print(f"{key}: {value}")
                
        elif args.clean_dir:
            # Clean directory mode
            merger.clean_duplicates(args.clean_dir, args.area)
            
        elif args.files:
            # Merge mode
            output = merger.merge_files(args.files, args.output)
            print(f"Merged data saved to: {output}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())