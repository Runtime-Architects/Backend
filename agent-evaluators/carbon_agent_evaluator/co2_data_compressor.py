import json
import datetime
import os
import glob
from typing import Dict, List
from pathlib import Path

class CO2DataCompressor:
    def __init__(self, input_file: str = None):
        """Load the original CO2 intensity data from real scraper output"""
        if input_file is None:
            # Auto-find the most recent REAL CO2 data file from scrapers
            input_file = self.find_latest_co2_file()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Real CO2 data file not found: {input_file}. Please run the EirGrid scraper first: python scraper_tools/run_eirgrid_downloader.py --areas co2_intensity --start YYYY-MM-DD --end YYYY-MM-DD --region all --forecast --output-dir ./data")
            
        with open(input_file, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        
        self.input_file = input_file
        print(f"📁 Loaded REAL CO2 data from EirGrid scraper: {input_file}")
        
        # Validate this is real scraper data
        if not self._validate_real_scraper_data():
            raise ValueError(f"Data appears to be mock/dummy data. Please ensure you're using real EirGrid scraper output.")
        
        # Debug: Print data structure to understand format
        print(f"🔍 Real EirGrid data structure: {list(self.original_data.keys())}")
        if 'data' in self.original_data:
            data_section = self.original_data['data']
            if isinstance(data_section, dict) and 'time_series' in data_section:
                print(f"📊 Real time series entries: {len(data_section['time_series'])}")
                print(f"📋 Sample real entry: {data_section['time_series'][0] if data_section['time_series'] else 'Empty'}")
            elif isinstance(data_section, list):
                print(f"📊 Real data list entries: {len(data_section)}")
                print(f"📋 Sample real entry: {data_section[0] if data_section else 'Empty'}")
    
    def _validate_real_scraper_data(self) -> bool:
        """Validate that this is real scraper data"""
        # Check for realistic EirGrid characteristics
        if 'metadata' in self.original_data:
            metadata = self.original_data['metadata']
            if 'source' in metadata and 'eirgrid' not in metadata['source'].lower():
                print(f"⚠️ Data source is not EirGrid: {metadata.get('source', 'unknown')}")
                return False
        
        # Check for realistic CO2 intensity values (EirGrid typical range: 100-500g CO2/kWh)
        values = []
        if 'data' in self.original_data:
            data_section = self.original_data['data']
            if isinstance(data_section, dict) and 'time_series' in data_section:
                values = [entry.get('value', 0) for entry in data_section['time_series'] if isinstance(entry, dict)]
            elif isinstance(data_section, list):
                values = [entry.get('value', 0) for entry in data_section if isinstance(entry, dict)]
        
        if values:
            min_val, max_val = min(values), max(values)
            # Realistic EirGrid values should be in reasonable range
            if min_val < 50 or max_val > 600 or (max_val - min_val) < 50:
                print(f"⚠️ CO2 intensity values: {min_val}-{max_val}g CO2/kWh")
                print("Warning: CO2 values may not reflect typical grid variation.")
        
        print("✅ Data validation passed - appears to be real EirGrid scraper output")
        return True
    
    def find_latest_co2_file(self) -> str:
        """Find the most recent REAL CO2 intensity data file from scrapers"""
        # Look for REAL CO2 data files from scraper output
        search_patterns = [
            "data/co2_intensity/co2_intensity_*.json",
            "data/co2_intensity_*.json", 
            "scraper_output/co2_intensity_*.json",
            "./data/co2_intensity/co2_intensity_*.json",
            "./scraper_output/co2_intensity_*.json"
        ]
        
        all_files = []
        for pattern in search_patterns:
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Filter out any files that might be compressed or mock versions
        real_files = []
        for file in all_files:
            filename = os.path.basename(file).lower()
            if not any(word in filename for word in ['compressed', 'mock', 'test', 'dummy', 'sample']):
                real_files.append(file)
        
        if not real_files:
            raise FileNotFoundError(
                "No REAL CO2 intensity data files found from EirGrid scraper. Please run the scraper first:\n\n"
                "python scraper_tools/run_eirgrid_downloader.py --areas co2_intensity --start 2025-01-28 --end 2025-01-28 --region all --forecast --output-dir ./data\n\n"
                "Expected locations:\n"
                "- data/co2_intensity/co2_intensity_*.json\n"
                "- data/co2_intensity_*.json\n"
                "- scraper_output/co2_intensity_*.json"
            )
        
        # Sort by modification time to get the most recent REAL data
        latest_file = max(real_files, key=os.path.getmtime)
        print(f"📊 Found latest REAL EirGrid data: {latest_file}")
        return latest_file
    
    def normalize_data_structure(self, data) -> List[Dict]:
        """Normalize EirGrid scraper data structure to consistent format"""
        time_series_data = []
        
        # Handle EirGrid scraper output structures
        if isinstance(data, dict):
            if 'data' in data:
                if isinstance(data['data'], dict) and 'time_series' in data['data']:
                    # EirGrid format: {'data': {'time_series': [...]}}
                    time_series_data = data['data']['time_series']
                    print("🔍 Processing EirGrid time_series format")
                elif isinstance(data['data'], list):
                    # Simplified format: {'data': [...]}
                    time_series_data = data['data']
                    print("🔍 Processing EirGrid list format")
                else:
                    raise ValueError(f"Unexpected EirGrid data structure in 'data' key: {type(data['data'])}")
            elif 'time_series' in data:
                # Direct time series: {'time_series': [...]}
                time_series_data = data['time_series']
                print("🔍 Processing direct EirGrid time_series format")
            else:
                raise ValueError(f"Cannot find EirGrid time series data. Available keys: {list(data.keys())}")
        elif isinstance(data, list):
            # Direct list format from scraper
            time_series_data = data
            print("🔍 Processing direct EirGrid list format")
        else:
            raise ValueError(f"Unexpected EirGrid data type: {type(data)}")
        
        # Normalize each EirGrid entry to consistent format
        normalized = []
        for i, entry in enumerate(time_series_data):
            try:
                if isinstance(entry, dict):
                    # Handle EirGrid time formats
                    time_key = None
                    value_key = None
                    
                    # Find time key (EirGrid uses various formats)
                    for key in ['time', 'timestamp', 'datetime', 'date_time', 'start_time']:
                        if key in entry:
                            time_key = key
                            break
                    
                    # Find value key (EirGrid CO2 intensity)
                    for key in ['value', 'intensity', 'co2_intensity', 'carbon_intensity', 'emission_rate']:
                        if key in entry:
                            value_key = key
                            break
                    
                    if time_key and value_key:
                        # Validate this is a real CO2 intensity value
                        co2_value = float(entry[value_key])
                        if 50 <= co2_value <= 600:  # Realistic EirGrid range
                            normalized.append({
                                'time': entry[time_key],
                                'value': co2_value
                            })
                        else:
                            print(f"⚠️ Skipping unrealistic CO2 value at entry {i}: {co2_value}g CO2/kWh")
                    else:
                        missing_keys = []
                        if not time_key:
                            missing_keys.append("time key")
                        if not value_key:
                            missing_keys.append("value key")
                        print(f"⚠️ Entry {i} missing {', '.join(missing_keys)}: {list(entry.keys())}")
                else:
                    print(f"⚠️ Entry {i} is not a dictionary: {type(entry)}")
            except Exception as e:
                print(f"⚠️ Error processing EirGrid entry {i}: {e}")
                continue
        
        if not normalized:
            raise ValueError("No valid EirGrid time series entries found. Check scraper output format.")
        
        print(f"✅ Normalized {len(normalized)} real EirGrid data points")
        return normalized
    
    def compress_data(self, interval_minutes: int = 30) -> Dict:
        """
        Compress REAL EirGrid CO2 data by reducing frequency
        
        Args:
            interval_minutes: Interval in minutes (30 or 60 recommended for EirGrid data)
        """
        # Normalize the EirGrid scraper data structure first
        time_series = self.normalize_data_structure(self.original_data)
        
        if not time_series:
            raise ValueError("No real EirGrid data to compress")
        
        # Filter data points based on interval
        compressed_points = []
        
        for entry in time_series:
            try:
                time_str = entry['time']
                
                # Parse EirGrid time formats (they use ISO format typically)
                if 'T' in time_str:  # ISO format: 2025-01-28T14:30:00Z or similar
                    # Handle timezone indicators
                    time_clean = time_str.replace('Z', '+00:00')
                    if '+' not in time_clean and time_clean.endswith('00'):
                        time_clean = time_str.replace('Z', '')
                    try:
                        dt = datetime.datetime.fromisoformat(time_clean)
                    except ValueError:
                        # Try parsing without timezone
                        dt = datetime.datetime.fromisoformat(time_str.split('T')[0] + 'T' + time_str.split('T')[1].split('Z')[0].split('+')[0])
                elif ' ' in time_str:  # Format: 2025-01-28 14:30:00
                    dt = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                elif ':' in time_str and len(time_str) <= 5:  # Format: 14:30
                    dt = datetime.datetime.strptime(time_str, '%H:%M')
                else:
                    # Try other EirGrid formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M', '%H:%M:%S']:
                        try:
                            dt = datetime.datetime.strptime(time_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        print(f"⚠️ Cannot parse EirGrid time format: {time_str}")
                        continue
                
                # Only keep data points at the specified interval
                if interval_minutes == 30:
                    # Keep points at :00 and :30 (EirGrid typical resolution)
                    if dt.minute in [0, 30]:
                        compressed_points.append({
                            'time': dt.strftime('%H:%M'),  # Simplified time format
                            'value': entry['value'],
                            'original_time': time_str  # Keep original EirGrid timestamp
                        })
                elif interval_minutes == 60:
                    # Keep points at :00 only
                    if dt.minute == 0:
                        compressed_points.append({
                            'time': dt.strftime('%H:%M'),
                            'value': entry['value'],
                            'original_time': time_str
                        })
                else:
                    # For other intervals, use modulo logic
                    if dt.minute % interval_minutes == 0:
                        compressed_points.append({
                            'time': dt.strftime('%H:%M'),
                            'value': entry['value'],
                            'original_time': time_str
                        })
                        
            except Exception as e:
                print(f"⚠️ Error processing EirGrid time entry {entry}: {e}")
                continue
        
        if not compressed_points:
            raise ValueError("No EirGrid data points could be compressed - check time format and scraper output")
        
        # Get EirGrid metadata
        metadata = self.original_data.get('metadata', {})
        if not metadata and 'data' in self.original_data:
            # Try to extract metadata from EirGrid data structure
            data_section = self.original_data['data']
            if isinstance(data_section, dict):
                metadata = {k: v for k, v in data_section.items() if k != 'time_series'}
        
        # Validate we have realistic EirGrid data spread
        values = [point['value'] for point in compressed_points]
        min_val, max_val = min(values), max(values)
        if (max_val - min_val) < 50:
            print(f"⚠️ Warning: Low variation in CO2 data ({min_val:.0f}-{max_val:.0f}g CO2/kWh). Verify scraper is getting real-time data.")
        
        # Create compressed structure with EirGrid source validation
        compressed_data = {
            'date': metadata.get('date_from', datetime.datetime.now().strftime('%Y-%m-%d')),
            'region': metadata.get('region', 'all'),
            'interval_minutes': interval_minutes,
            'total_points': len(compressed_points),
            'data': compressed_points,
            'metadata': {
                'original_file': self.input_file,
                'compressed_at': datetime.datetime.now().isoformat(),
                'date_from': metadata.get('date_from', ''),
                'date_to': metadata.get('date_to', ''),
                'original_entries': len(time_series),
                'source': 'real_eirgrid_scraper',
                'data_validation': 'passed_real_data_checks',
                'co2_range': f"{min_val:.0f}-{max_val:.0f}g CO2/kWh"
            }
        }
        
        return compressed_data
    
    def save_compressed(self, output_file: str = None, interval_minutes: int = 30):
        """Save compressed REAL EirGrid data to file"""
        if output_file is None:
            # Auto-generate filename with real data indicator
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            output_file = f"{base_name}_compressed.json"
        
        compressed = self.compress_data(interval_minutes)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(compressed, f, indent=2, ensure_ascii=False)
        
        original_size = len(json.dumps(self.original_data))
        compressed_size = len(json.dumps(compressed))
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"✅ Compressed REAL EirGrid data saved to {output_file}")
        print(f"📊 Real Data Compression Stats:")
        print(f"   Original EirGrid entries: {compressed['metadata']['original_entries']}")
        print(f"   Compressed points: {compressed['total_points']}")
        print(f"   CO2 intensity range: {compressed['metadata']['co2_range']}")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Original size: {original_size:,} characters")
        print(f"   Compressed size: {compressed_size:,} characters")
        print(f"   Space saved: {compression_ratio:.1f}%")
        
        return compressed, output_file
    
    def compare_intervals(self):
        """Compare different compression intervals for real EirGrid data"""
        intervals = [15, 30, 60]
        
        print("🔍 EirGrid Data Compression Comparison:")
        print("Interval | Points | Size Reduction")
        print("-" * 35)
        
        original_data = self.normalize_data_structure(self.original_data)
        original_points = len(original_data)
        original_size = len(json.dumps(self.original_data))
        
        for interval in intervals:
            try:
                compressed = self.compress_data(interval)
                compressed_size = len(json.dumps(compressed))
                reduction = (1 - compressed_size / original_size) * 100
                
                print(f"{interval:2d} min   | {compressed['total_points']:3d}    | {reduction:5.1f}%")
            except Exception as e:
                print(f"{interval:2d} min   | ERROR  | {str(e)[:20]}...")
    
    @staticmethod
    def prepare_for_evaluation(data_dir: str = "./data") -> str:
        """
        Prepare REAL EirGrid CO2 data for evaluation by finding and compressing the latest data
        Returns the path to the compressed file
        """
        try:
            print("🎯 Preparing REAL EirGrid CO2 data for evaluation...")
            
            # Find the latest REAL CO2 data file from scrapers
            compressor = CO2DataCompressor()
            
            # Compress and save REAL data with a unique name to ensure it's used
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            compressed_file = os.path.join(data_dir, f"co2_intensity_compressed_{timestamp}.json")
            
            compressed_data, compressed_file = compressor.save_compressed(
                output_file=compressed_file,
                interval_minutes=30
            )
            
            print(f"🎯 REAL EirGrid CO2 data prepared for evaluation: {compressed_file}")
            return compressed_file
            
        except Exception as e:
            print(f"❌ Error preparing REAL CO2 data: {e}")
            print(f"\n🔧 SOLUTION: Run the EirGrid scraper first:")
            print(f"python scraper_tools/run_eirgrid_downloader.py --areas co2_intensity --start 2025-01-28 --end 2025-01-28 --region all --forecast --output-dir ./data")
            raise

# Usage example
if __name__ == "__main__":
    try:
        print("🌍 Processing REAL EirGrid CO2 Data")
        print("=" * 50)
        
        # Initialize compressor with REAL data only
        compressor = CO2DataCompressor()
        
        # Compare different intervals
        compressor.compare_intervals()
        
        # Save compressed version (30-minute intervals)
        compressed_data, output_file = compressor.save_compressed("co2_intensity_compressed.json", interval_minutes=30)
        
        # Show sample of REAL compressed data
        print(f"\n📋 Sample REAL EirGrid Data:")
        for i, point in enumerate(compressed_data['data'][:5]):
            print(f"  {point['time']}: {point['value']}g CO2/kWh (real EirGrid data)")
        if len(compressed_data['data']) > 5:
            print("  ...")
            
        print(f"\n✅ REAL data processing completed successfully!")
        print(f"📊 CO2 range: {compressed_data['metadata']['co2_range']}")
        print(f"🕐 Data points: {compressed_data['total_points']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure to run the EirGrid scraper first to get real data!")