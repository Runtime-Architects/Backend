"""
EirGrid Data Merger Utility
Intelligently merges data files, replacing forecast values with actual values
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import re


class OrganizedDataMerger:
    """Handles intelligent merging of EirGrid data files in organized structure"""
    
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
                data = json.load(f)
                all_data.append((path, data))
        
        # Start with first file as base
        base_path, merged = all_data[0]
        merged = merged.copy()
        
        # Merge remaining files
        for path, data in all_data[1:]:
            merged = self._merge_two_datasets(merged, data)
        
        # Update metadata
        if 'metadata' not in merged:
            merged['metadata'] = {}
        
        merged['metadata']['last_updated'] = datetime.now().isoformat()
        merged['metadata']['merged_from'] = file_paths
        
        # Determine output path
        if output_path:
            output = output_path
        else:
            # Use the first file path but update with merged date range
            merged_date_range = self._calculate_merged_date_range(merged)
            if merged_date_range:
                output = self._update_filename_with_date_range(base_path, merged_date_range)
            else:
                output = base_path
        
        # Ensure output directory exists
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        with open(output, 'w') as f:
            json.dump(merged, f, indent=2)
        
        self.logger.info(f"Merged data saved to {output}")
        return output
    
    def _merge_two_datasets(self, base: Dict, new: Dict) -> Dict:
        """Merge two datasets intelligently"""
        
        # Get time series data from both datasets
        base_data = base.get('data', {}) if 'data' in base else base
        new_data = new.get('data', {}) if 'data' in new else new
        
        base_ts = base_data.get('time_series', [])
        new_ts = new_data.get('time_series', [])
        
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
                
                # Update if values differ and not downgrading from actual to forecast
                elif (existing.get('value') != new_point.get('value') and 
                      not (not existing.get('is_forecast', False) and new_point.get('is_forecast', False))):
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
        
        # Update the base structure
        if 'data' in base:
            base['data']['time_series'] = merged_ts
        else:
            base['time_series'] = merged_ts
        
        # Update metadata
        if 'metadata' in new:
            if 'metadata' not in base:
                base['metadata'] = {}
            
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
            '%Y-%m-%dT%H:%M:%S',
            '%d %B %Y, %H:%M',
            '%d %b %Y, %H:%M'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str.split('.')[0], fmt)  # Remove milliseconds
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse time: {time_str}")
        return datetime.now()
    
    def _calculate_merged_date_range(self, merged_data: Dict) -> Optional[Tuple[str, str]]:
        """Calculate the date range of merged data"""
        
        data_section = merged_data.get('data', {}) if 'data' in merged_data else merged_data
        time_series = data_section.get('time_series', [])
        
        if not time_series:
            return None
        
        # Parse all times to find min and max dates
        dates = []
        for point in time_series:
            try:
                parsed_time = self._parse_time(point['time'])
                dates.append(parsed_time.date())
            except:
                continue
        
        if dates:
            min_date = min(dates).strftime('%Y-%m-%d')
            max_date = max(dates).strftime('%Y-%m-%d')
            return min_date, max_date
        
        return None
    
    def _update_filename_with_date_range(self, original_path: str, date_range: Tuple[str, str]) -> str:
        """Update filename with new date range"""
        
        path_obj = Path(original_path)
        filename = path_obj.stem
        
        # Parse existing filename to extract area and region
        parts = filename.split('_')
        
        # New consistent format: area_region_start_end (4 parts)
        if len(parts) >= 4:
            area = parts[0]
            region = parts[1]
            new_filename = f"{area}_{region}_{date_range[0]}_{date_range[1]}.json"
        else:
            # Fallback for old format files
            area = parts[0]
            # Default to 'all' region if not specified
            new_filename = f"{area}_all_{date_range[0]}_{date_range[1]}.json"
        
        return str(path_obj.parent / new_filename)
    
    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a data file and return statistics"""
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old and new data structures
        data_section = data.get('data', {}) if 'data' in data else data
        time_series = data_section.get('time_series', [])
        
        stats = {
            'file': file_path,
            'total_points': len(time_series),
            'actual_values': 0,
            'forecast_values': 0,
            'null_values': 0,
            'time_range': None,
            'area': data.get('metadata', {}).get('area', 'unknown'),
            'region': data.get('metadata', {}).get('region', 'unknown'),
            'date_from': data.get('metadata', {}).get('date_from', 'unknown'),
            'date_to': data.get('metadata', {}).get('date_to', 'unknown')
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
    
    def find_metric_files(self, data_dir: str, area: str = None) -> Dict[str, List[str]]:
        """
        Find all files for metrics in the organized structure
        
        Args:
            data_dir: Data directory path
            area: Optional specific area to search
            
        Returns:
            Dictionary mapping areas to list of files
        """
        
        data_path = Path(data_dir)
        metric_files = {}
        
        if area:
            # Search specific metric directory
            metric_dir = data_path / area
            if metric_dir.exists() and metric_dir.is_dir():
                files = list(metric_dir.glob(f"{area}_*.json"))
                if files:
                    metric_files[area] = [str(f) for f in files]
        else:
            # Search all metric directories
            for metric_dir in data_path.iterdir():
                if metric_dir.is_dir():
                    metric_name = metric_dir.name
                    files = list(metric_dir.glob(f"{metric_name}_*.json"))
                    if files:
                        metric_files[metric_name] = [str(f) for f in files]
        
        return metric_files
    
    def find_overlapping_files(self, data_dir: str, area: str, target_date_from: str, target_date_to: str, region: str = None) -> List[str]:
        """
        Find files that overlap with the given date range
        
        Args:
            data_dir: Data directory path
            area: Metric area
            target_date_from: Target start date (YYYY-MM-DD)
            target_date_to: Target end date (YYYY-MM-DD)
            region: Optional region filter
            
        Returns:
            List of overlapping file paths
        """
        
        metric_dir = Path(data_dir) / area
        if not metric_dir.exists():
            return []
        
        overlapping_files = []
        target_start = datetime.strptime(target_date_from, '%Y-%m-%d')
        target_end = datetime.strptime(target_date_to, '%Y-%m-%d')
        
        # Look for files in the metric directory
        pattern = f"{area}_*.json"
        
        for file_path in metric_dir.glob(pattern):
            try:
                # Extract dates and region from filename
                filename = file_path.stem
                parts = filename.split('_')
                
                # New consistent format: area_region_start_end (4 parts)
                if len(parts) >= 4:
                    file_region = parts[1]
                    file_start_str = parts[2]
                    file_end_str = parts[3]
                    
                    # Apply region filter if specified
                    if region and file_region != region:
                        continue
                        
                    file_start = datetime.strptime(file_start_str, '%Y-%m-%d')
                    file_end = datetime.strptime(file_end_str, '%Y-%m-%d')
                    
                    # Check for overlap
                    if target_start <= file_end and target_end >= file_start:
                        overlapping_files.append(str(file_path))
                        
            except (ValueError, IndexError) as e:
                self.logger.debug(f"Could not parse filename {file_path.name}: {e}")
                continue
        
        return overlapping_files
    
    def clean_overlapping_files(self, data_dir: str, area: str = None, dry_run: bool = True):
        """
        Find and optionally merge overlapping files for the same area
        
        Args:
            data_dir: Directory to scan
            area: Optional specific area to process
            dry_run: If True, only show what would be done without making changes
        """
        
        metric_files = self.find_metric_files(data_dir, area)
        
        for metric_area, files in metric_files.items():
            if len(files) <= 1:
                continue
            
            self.logger.info(f"\n📊 Processing {metric_area} ({len(files)} files)")
            
            # Group files by potential overlaps
            overlaps_found = []
            
            for i, file1 in enumerate(files):
                for j, file2 in enumerate(files[i+1:], i+1):
                    # Extract date ranges from both files
                    try:
                        range1 = self._extract_date_range_from_filename(file1)
                        range2 = self._extract_date_range_from_filename(file2)
                        
                        if range1 and range2:
                            # Check for overlap
                            start1, end1 = range1
                            start2, end2 = range2
                            
                            if start1 <= end2 and start2 <= end1:
                                overlaps_found.append((file1, file2, range1, range2))
                                
                    except Exception as e:
                        self.logger.debug(f"Error checking overlap between {file1} and {file2}: {e}")
            
            if overlaps_found:
                self.logger.info(f"  🔍 Found {len(overlaps_found)} overlapping file pairs:")
                
                for file1, file2, range1, range2 in overlaps_found:
                    file1_name = Path(file1).name
                    file2_name = Path(file2).name
                    self.logger.info(f"    📄 {file1_name} ({range1[0]} to {range1[1]})")
                    self.logger.info(f"    📄 {file2_name} ({range2[0]} to {range2[1]})")
                    
                    if not dry_run:
                        # Analyze files to determine merge strategy
                        stats1 = self.analyze_file(file1)
                        stats2 = self.analyze_file(file2)
                        
                        # Prefer file with more actual values
                        if stats1['actual_values'] >= stats2['actual_values']:
                            base_file, merge_file = file1, file2
                        else:
                            base_file, merge_file = file2, file1
                        
                        self.logger.info(f"    🔄 Merging into {Path(base_file).name}")
                        
                        try:
                            merged_path = self.merge_files([base_file, merge_file], base_file)
                            
                            # Remove the merged file if different from base
                            if merge_file != merged_path:
                                Path(merge_file).unlink()
                                self.logger.info(f"    🗑️  Removed {Path(merge_file).name}")
                                
                        except Exception as e:
                            self.logger.error(f"    ❌ Merge failed: {e}")
                    
                    self.logger.info("")
            else:
                self.logger.info(f"  ✅ No overlapping files found")
        
        if dry_run:
            self.logger.info("\n💡 This was a dry run. Use --execute to actually merge files.")
    
    def _extract_date_range_from_filename(self, file_path: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract date range from filename"""
        
        filename = Path(file_path).stem
        parts = filename.split('_')
        
        try:
            # New consistent format: area_region_start_end (4 parts)
            if len(parts) >= 4:
                start_str = parts[2]
                end_str = parts[3]
            else:
                # Fallback for old format files (should not occur with new system)
                start_str = parts[1]
                end_str = parts[2]
            
            start_date = datetime.strptime(start_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_str, '%Y-%m-%d')
            
            return start_date, end_date
        except (ValueError, IndexError):
            pass
        
        return None


def main():
    """CLI for organized data merger utility"""
    
    parser = argparse.ArgumentParser(
        description="Merge EirGrid data files in organized structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge specific files
  python data_merger.py file1.json file2.json -o merged.json
  
  # Clean overlapping files in organized structure (dry run)
  python data_merger.py --clean-dir ./data
  
  # Actually merge overlapping files
  python data_merger.py --clean-dir ./data --execute
  
  # Clean specific metric only
  python data_merger.py --clean-dir ./data --area co2_intensity
  
  # Analyze a file
  python data_merger.py --analyze data/co2_intensity/co2_intensity_2025-06-23_2025-06-23.json
  
  # Find overlapping files for specific date range
  python data_merger.py --find-overlaps ./data co2_intensity 2025-06-20 2025-06-25
        """
    )
    
    parser.add_argument('files', nargs='*',
                       help='Files to merge')
    
    parser.add_argument('-o', '--output',
                       help='Output file path')
    
    parser.add_argument('--clean-dir',
                       help='Clean overlapping files in organized directory structure')
    
    parser.add_argument('--area',
                       help='Specific area to process when cleaning')
    
    parser.add_argument('--analyze',
                       help='Analyze a data file')
    
    parser.add_argument('--find-overlaps',
                       nargs=4,
                       metavar=('DATA_DIR', 'AREA', 'DATE_FROM', 'DATE_TO'),
                       help='Find overlapping files for area and date range')
    
    parser.add_argument('--execute',
                       action='store_true',
                       help='Actually execute merge operations (default is dry run)')
    
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    merger = OrganizedDataMerger()
    
    try:
        if args.analyze:
            # Analyze mode
            stats = merger.analyze_file(args.analyze)
            print("\n📊 FILE ANALYSIS:")
            print("-" * 50)
            print(f"File: {stats['file']}")
            print(f"Area: {stats['area']}")
            print(f"Region: {stats['region']}")
            print(f"Date Range: {stats['date_from']} to {stats['date_to']}")
            print(f"Total Points: {stats['total_points']}")
            print(f"Actual Values: {stats['actual_values']}")
            print(f"Forecast Values: {stats['forecast_values']}")
            print(f"Null Values: {stats['null_values']}")
            if stats['time_range']:
                print(f"Time Range: {stats['time_range']['start']} to {stats['time_range']['end']}")
                
        elif args.find_overlaps:
            # Find overlaps mode
            data_dir, area, date_from, date_to = args.find_overlaps
            overlapping = merger.find_overlapping_files(data_dir, area, date_from, date_to)
            
            print(f"\n🔍 OVERLAPPING FILES for {area} ({date_from} to {date_to}):")
            print("-" * 60)
            if overlapping:
                for file_path in overlapping:
                    print(f"📄 {file_path}")
            else:
                print("No overlapping files found.")
                
        elif args.clean_dir:
            # Clean directory mode
            print(f"\n🧹 CLEANING OVERLAPPING FILES {'(DRY RUN)' if not args.execute else '(EXECUTING)'}")
            print("=" * 60)
            merger.clean_overlapping_files(args.clean_dir, args.area, dry_run=not args.execute)
            
        elif args.files:
            # Merge mode
            output = merger.merge_files(args.files, args.output)
            print(f"✅ Merged data saved to: {output}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"💥 Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())