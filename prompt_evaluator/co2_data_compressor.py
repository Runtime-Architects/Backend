import json
import datetime
from typing import Dict, List

class CO2DataCompressor:
    def __init__(self, input_file: str):
        """Load the original CO2 intensity data"""
        with open(input_file, 'r') as f:
            self.original_data = json.load(f)
    
    def compress_data(self, interval_minutes: int = 30) -> Dict:
        """
        Compress CO2 data by reducing frequency and simplifying structure
        
        Args:
            interval_minutes: Interval in minutes (30 or 60 recommended)
        """
        time_series = self.original_data['data']['time_series']
        
        # Filter data points based on interval
        compressed_points = []
        
        for entry in time_series:
            dt = datetime.datetime.strptime(entry['time'], '%Y-%m-%d %H:%M:%S')
            
            # Only keep data points at the specified interval
            if interval_minutes == 30:
                # Keep points at :00 and :30
                if dt.minute in [0, 30]:
                    compressed_points.append({
                        'time': dt.strftime('%H:%M'),  # Simplified time format
                        'value': entry['value']
                    })
            elif interval_minutes == 60:
                # Keep points at :00 only
                if dt.minute == 0:
                    compressed_points.append({
                        'time': dt.strftime('%H:%M'),
                        'value': entry['value']
                    })
            else:
                # For other intervals, use modulo logic
                if dt.minute % interval_minutes == 0:
                    compressed_points.append({
                        'time': dt.strftime('%H:%M'),
                        'value': entry['value']
                    })
        
        # Create compressed structure
        compressed_data = {
            'date': self.original_data['metadata']['date_from'],
            'region': self.original_data['metadata']['region'],
            'interval_minutes': interval_minutes,
            'total_points': len(compressed_points),
            'data': compressed_points
        }
        
        return compressed_data
    
    def save_compressed(self, output_file: str, interval_minutes: int = 30):
        """Save compressed data to file"""
        compressed = self.compress_data(interval_minutes)
        
        with open(output_file, 'w') as f:
            json.dump(compressed, f, indent=2)
        
        original_size = len(json.dumps(self.original_data))
        compressed_size = len(json.dumps(compressed))
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"✅ Compressed data saved to {output_file}")
        print(f"📊 Compression Stats:")
        print(f"   Original points: {len(self.original_data['data']['time_series'])}")
        print(f"   Compressed points: {compressed['total_points']}")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Original size: {original_size:,} characters")
        print(f"   Compressed size: {compressed_size:,} characters")
        print(f"   Space saved: {compression_ratio:.1f}%")
        
        return compressed
    
    def compare_intervals(self):
        """Compare different compression intervals"""
        intervals = [15, 30, 60]
        
        print("🔍 Compression Comparison:")
        print("Interval | Points | Size Reduction")
        print("-" * 35)
        
        original_points = len(self.original_data['data']['time_series'])
        original_size = len(json.dumps(self.original_data))
        
        for interval in intervals:
            compressed = self.compress_data(interval)
            compressed_size = len(json.dumps(compressed))
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"{interval:2d} min   | {compressed['total_points']:3d}    | {reduction:5.1f}%")

# Usage example
if __name__ == "__main__":
    # Initialize compressor
    compressor = CO2DataCompressor("co2_intensity_all_2025-06-29_2025-06-29.json")
    
    # Compare different intervals
    compressor.compare_intervals()
    
    # Save compressed version (30-minute intervals)
    compressed_data = compressor.save_compressed("co2_intensity_compressed.json", interval_minutes=30)
    
    # Show sample of compressed data
    print("\n📋 Sample Compressed Data:")
    for i, point in enumerate(compressed_data['data'][:5]):
        print(f"  {point['time']}: {point['value']}g CO2/kWh")
    print("  ...")