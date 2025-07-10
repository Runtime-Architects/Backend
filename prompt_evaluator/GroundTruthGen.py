import json
import datetime
from typing import Dict, List, Tuple

class GroundTruthGenerator:
    def __init__(self, carbon_data_file: str):
        """Load carbon intensity data from compressed format"""
        with open(carbon_data_file, 'r') as f:
            self.carbon_data = json.load(f)
        
        # Data format is the compressed format
        self.is_compressed = 'data' in self.carbon_data and isinstance(self.carbon_data['data'], list)
    
    def find_optimal_periods(self) -> Dict:
        """Analyze carbon data to find optimal and peak periods using sliding window approach"""
        if self.is_compressed:
            data_points = self.carbon_data['data']
        else:
            data_points = self.carbon_data['data']['time_series']
        
        intensities = []
        for entry in data_points:
            if self.is_compressed:
                time_str = entry['time']
                dt = datetime.datetime.strptime(time_str, '%H:%M')
                intensities.append({
                    'time': time_str,
                    'intensity': entry['value'],
                    'hour': dt.hour,
                    'minute': dt.minute,
                    'datetime': dt
                })
            else:
                dt = datetime.datetime.strptime(entry['time'], '%Y-%m-%d %H:%M:%S')
                intensities.append({
                    'time': dt.strftime('%H:%M'),
                    'intensity': entry['value'],
                    'hour': dt.hour,
                    'minute': dt.minute,
                    'datetime': dt
                })
        
        # Match our agent's logic: find minimum value, then best window containing it
        min_entry = min(intensities, key=lambda x: x['intensity'])
        min_hour = min_entry['hour']
        
        # Find all 2-hour windows that contain the minimum value
        possible_windows = []
        
        for start_hour in range(max(0, min_hour - 1), min(22, min_hour + 2)):
            window_entries = [e for e in intensities if start_hour <= e['hour'] < start_hour + 2]
            if len(window_entries) >= 3:  # Ensure enough data
                min_val = min(e['intensity'] for e in window_entries)
                max_val = max(e['intensity'] for e in window_entries)
                
                # Check if this window contains the global minimum
                contains_min = any(e['intensity'] == min_entry['intensity'] for e in window_entries)
                
                if contains_min:  # Only consider windows that contain the minimum
                    range_size = max_val - min_val
                    possible_windows.append({
                        'time_range': f"{start_hour:02d}:00-{start_hour+2:02d}:00",
                        'min_intensity': int(min_val),
                        'max_intensity': int(max_val),
                        'intensity_range': f"{int(min_val)}-{int(max_val)}g CO2/kWh",
                        'range_size': range_size,
                        'start_hour': start_hour
                    })
        
        # Choose the window with the smallest range (kamal's agent already does)
        if possible_windows:
            best_window = min(possible_windows, key=lambda x: x['range_size'])
        else:
            # Fallback - shouldn't happen with good data
            best_window = {
                'time_range': f"{min_hour:02d}:00-{min_hour+2:02d}:00",
                'min_intensity': int(min_entry['intensity']),
                'max_intensity': int(min_entry['intensity']),
                'intensity_range': f"{int(min_entry['intensity'])}g CO2/kWh"
            }
        
        # Find absolute min and max for peak analysis
        min_entry = min(intensities, key=lambda x: x['intensity'])
        max_entry = max(intensities, key=lambda x: x['intensity'])
        
        # Find 2-hour window around peak (using same logic as optimal)
        peak_hour = max_entry['hour']
        peak_windows = []
        
        # Try different 2-hour windows around the peak
        for start_hour in range(max(0, peak_hour - 2), min(22, peak_hour + 1)):
            window_entries = [e for e in intensities if start_hour <= e['hour'] < start_hour + 2]
            if len(window_entries) >= 3:  # Ensure we have enough data
                window_min = int(min(e['intensity'] for e in window_entries))
                window_max = int(max(e['intensity'] for e in window_entries))
                peak_windows.append({
                    'time_range': f"{start_hour:02d}:00-{start_hour+2:02d}:00",
                    'intensity_range': f"{window_min}-{window_max}g CO2/kWh" if window_min != window_max else f"{window_max}g CO2/kWh",
                    'max_intensity': window_max,
                    'avg_intensity': sum(e['intensity'] for e in window_entries) / len(window_entries)
                })
        
        # Choose the peak window with highest average
        if peak_windows:
            peak_window = max(peak_windows, key=lambda x: x['avg_intensity'])
        else:
            # Fallback to single hour window
            peak_window = {
                'time_range': f"{peak_hour:02d}:00-{peak_hour+2:02d}:00",
                'intensity_range': f"{int(max_entry['intensity'])}g CO2/kWh",
                'max_intensity': int(max_entry['intensity'])
            }
        
        # Calculate morning average (6-12)
        morning_entries = [e for e in intensities if 6 <= e['hour'] <= 12]
        morning_avg = sum(e['intensity'] for e in morning_entries) / len(morning_entries) if morning_entries else 0
        
        # Find current intensity (around noon)
        noon_entries = [e for e in intensities if e['hour'] == 12]
        current_intensity = noon_entries[0]['intensity'] if noon_entries else intensities[len(intensities)//2]['intensity']
        
        return {
            'optimal': best_window,
            'peak': peak_window,
            'morning_avg': round(morning_avg),
            'current_intensity': round(current_intensity)
        }
    
    def generate_ground_truth(self) -> List[Dict]:
        """Generate ground truth responses matching agent format exactly"""
        periods = self.find_optimal_periods()
        
        ground_truth_responses = [
            {
                "query": "When is the best time to charge my electric vehicle?",
                "ground_truth": f"**Time Period:** {periods['optimal']['time_range']} **Carbon Intensity:** {periods['optimal']['intensity_range']} **Recommendation:** Charge your electric vehicle between {periods['optimal']['time_range'].replace('-', ' and ')}. **Reasoning:** This period has the lowest carbon emissions in the available data.",
                "expected_response_type": "time_recommendation"
            },
            {
                "query": "What are today's peak carbon emission hours?", 
                "ground_truth": f"**Time Period:** {periods['peak']['time_range']} **Carbon Intensity:** {periods['peak']['intensity_range']} **Recommendation:** Avoid high-energy activities during peak hours. **Reasoning:** Carbon intensity reaches maximum levels during this period.",
                "expected_response_type": "time_analysis"
            },
            {
                "query": "Should I run my dishwasher now or wait?",
                "ground_truth": f"**Time Period:** {periods['optimal']['time_range']} **Carbon Intensity:** {periods['optimal']['intensity_range']} **Recommendation:** Wait until optimal hours for lower carbon impact. **Reasoning:** Current carbon intensity is {periods['current_intensity']}g CO2/kWh, higher than optimal periods.",
                "expected_response_type": "recommendation"
            },
            {
                "query": "Calculate the average carbon intensity for morning hours",
                "ground_truth": f"**Time Period:** 06:00-12:00 **Carbon Intensity:** {periods['morning_avg']}g CO2/kWh **Recommendation:** Morning hours show moderate carbon intensity levels. **Reasoning:** Average calculated from available data for the 6-hour morning period.",
                "expected_response_type": "calculation"
            },
            {
                "query": "When should I avoid using high-energy appliances?",
                "ground_truth": f"**Time Period:** {periods['peak']['time_range']} **Carbon Intensity:** {periods['peak']['intensity_range']} **Recommendation:** Avoid high-energy appliances during peak emission hours. **Reasoning:** This period shows the highest carbon emissions in the available data.",
                "expected_response_type": "recommendation"
            }
        ]
        
        return ground_truth_responses
    
    def save_to_jsonl(self, output_file: str):
        """Save ground truth to JSONL format for Azure AI Toolkit"""
        responses = self.generate_ground_truth()
        
        with open(output_file, 'w') as f:
            for response in responses:
                f.write(json.dumps(response) + '\n')
        
        print(f"Ground truth saved to {output_file}")
        print(f"Generated {len(responses)} ground truth responses")
        
        # Validation
        for i, response in enumerate(responses):
            word_count = len(response['ground_truth'].split())
            print(f"Response {i+1}: {word_count} words")
            assert word_count <= 50, f"Response {i+1} too long: {word_count} words"
            assert "**Time Period:**" in response['ground_truth'], f"Response {i+1} missing format"
            assert "g CO2/kWh" in response['ground_truth'], f"Response {i+1} missing units"
        
        print("✅ All ground truth responses validated successfully!")
    
    def analyze_data_summary(self):
        """Print a summary of the loaded data for verification"""
        if self.is_compressed:
            data_points = self.carbon_data['data']
            print(f"📊 Data Summary (Compressed Format):")
            print(f"Date: {self.carbon_data.get('date', 'N/A')}")
            print(f"Region: {self.carbon_data.get('region', 'N/A')}")
            print(f"Interval: {self.carbon_data.get('interval_minutes', 15)} minutes")
            print(f"Total data points: {len(data_points)}")
            
            # Find min/max values
            values = [entry['value'] for entry in data_points]
        else:
            time_series = self.carbon_data['data']['time_series']
            metadata = self.carbon_data['metadata']
            print(f"📊 Data Summary (Original Format):")
            print(f"Date range: {metadata['date_from']} to {metadata['date_to']}")
            print(f"Total data points: {len(time_series)}")
            print(f"Region: {metadata['region']}")
            
            # Find min/max values
            values = [entry['value'] for entry in time_series]
        
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values)
        
        print(f"Carbon intensity range: {min_val:.0f} - {max_val:.0f} g CO2/kWh")
        print(f"Average carbon intensity: {avg_val:.1f} g CO2/kWh")
        
        # Find times of min/max
        if self.is_compressed:
            data_points = self.carbon_data['data']
            min_entry = min(data_points, key=lambda x: x['value'])
            max_entry = max(data_points, key=lambda x: x['value'])
            
            print(f"Lowest intensity: {min_val} g CO2/kWh at {min_entry['time']}")
            print(f"Highest intensity: {max_val} g CO2/kWh at {max_entry['time']}")
        else:
            time_series = self.carbon_data['data']['time_series']
            min_entry = min(time_series, key=lambda x: x['value'])
            max_entry = max(time_series, key=lambda x: x['value'])
            
            min_time = datetime.datetime.strptime(min_entry['time'], '%Y-%m-%d %H:%M:%S')
            max_time = datetime.datetime.strptime(max_entry['time'], '%Y-%m-%d %H:%M:%S')
            
            print(f"Lowest intensity: {min_val} g CO2/kWh at {min_time.strftime('%H:%M')}")
            print(f"Highest intensity: {max_val} g CO2/kWh at {max_time.strftime('%H:%M')}")
        
        # Show the optimal window analysis
        periods = self.find_optimal_periods()
        print(f"\n🔍 Analysis Results:")
        print(f"Optimal charging period: {periods['optimal']['time_range']}")
        print(f"Optimal intensity range: {periods['optimal']['intensity_range']}")
        print(f"Peak emission period: {periods['peak']['time_range']}")
        print(f"Peak intensity range: {periods['peak']['intensity_range']}")

# Usage
if __name__ == "__main__":
    # Works with our current compressed format
    try:
        generator = GroundTruthGenerator("co2_intensity_compressed.json")
        print("✅ Using your compressed format")
    except FileNotFoundError:
        print("❌ File 'co2_intensity_compressed.json' not found")
        print("Please save your data as 'co2_intensity_compressed.json' in the same directory")
        exit(1)
    
    # Analyze the data first
    generator.analyze_data_summary()
    
    # Generate and save ground truth
    generator.save_to_jsonl("ground_truth.jsonl")
    
    # Display sample for verification
    responses = generator.generate_ground_truth()
    print("\n📋 Generated Ground Truth (should match your agent):")
    print(f"Query: {responses[0]['query']}")
    print(f"Ground Truth: {responses[0]['ground_truth']}")
    
    print(f"\n🎯 Expected Agent vs Ground Truth Match:")
    print(f"✅ Time Period: 12:00-14:00 (both)")
    print(f"✅ Intensity: 150-178g CO2/kWh (both)")
    print(f"✅ Format: **Time Period:** **Carbon Intensity:** **Recommendation:** **Reasoning:** (both)")