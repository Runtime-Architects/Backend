import sys
from run_eirgrid_downloader import main as eirgrid_main

def call_as_cli():
    # Simulate command line arguments
    sys.argv = [
        'run_eirgrid_downloader.py',
        '--areas', 'co2_intensity,wind_generation',
        '--start', '2025-06-22',
        '--end', '2025-06-22',
        '--region', 'all',
        '--forecast',
        '--output-dir', './data'
    ]
    
    # Run the main function
    return eirgrid_main()

if __name__ == "__main__":
    exit_code = call_as_cli()
    print(f"CLI exited with code {exit_code}")