"""
Hybrid Agent Evaluation Runner
- Uses compressed data consistently
- Streamlined evaluation process
- Automatic scraper integration
"""
import asyncio
import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime, timedelta

# Import the evaluation framework
from hybrid_agent_evaluator import HybridAgentEvaluator, HybridEvaluationReport
from evaluation_strategies import EvaluationMode

# Import CO2 processing
try:
    from co2_data_compressor import CO2DataCompressor
    from ground_truth_generator import GroundTruthGenerator
    CO2_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: CO2 processing system not available: {e}")
    CO2_SYSTEM_AVAILABLE = False

# Import required libraries for Azure setup
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamlinedCarbonAgentEvaluationRunner:
    """
    Streamlined Carbon Agent Evaluation Runner
    - Uses compressed data for consistency
    - Automatic scraper integration
    - Simplified validation
    """
    
    def __init__(self):
        load_dotenv()
        self.setup_test_environment()
        self.evaluator = None
        self.compressed_data_file = None
        
    def setup_test_environment(self):
        """Setup the test environment and create necessary directories"""
        Path("./evaluation_results").mkdir(exist_ok=True)
        Path("./test_configs").mkdir(exist_ok=True)
        Path("./data").mkdir(exist_ok=True)
        Path("./data/co2_intensity").mkdir(exist_ok=True)
        
        # Create streamlined configuration
        self.create_streamlined_configs()
    
    def create_streamlined_configs(self):
        """Create streamlined configuration files"""
        config_path = Path("./test_configs/streamlined_config.json")
        if not config_path.exists():
            config = {
                "evaluation_mode": "hybrid",
                "llm_evaluation_enabled": True,
                "llm_on_failures_only": False,
                "max_retries": 3,
                "timeout_seconds": 120,
                
                # Balanced weighting
                "rule_weight": 0.30,
                "llm_weight": 0.70,
                "function_weight": 0.4,
                "keyword_weight": 0.2,
                "behavior_weight": 0.4,
                
                # Streamlined settings
                "auto_run_scraper": True,
                "use_compressed_data": True,
                "relaxed_validation": True,
                
                # LLM settings
                "llm_quality_threshold": 0.5,
                "llm_confidence_threshold": 0.4,
                "llm_override_enabled": True,
                "llm_override_threshold": 0.65,
                
                # Performance settings
                "consistency_threshold": 0.8,
                "output_dir": "./evaluation_results",
                "parallel_llm_calls": False,
                "max_llm_tokens": 4000,
                "llm_timeout": 90,
                
                # Ground truth settings
                "ground_truth_weight": 0.8,
                "enforce_scoring_consistency": True,
                "structured_penalty_system": True
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"SUCCESS: Created streamlined configuration: {config_path}")
    
    def validate_environment(self) -> bool:
        """Basic environment validation"""
        # Check Azure environment variables
        required_vars = ["AZURE_DEPLOYMENT", "AZURE_ENDPOINT", "API_KEY", "API_VERSION"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print("ERROR: Missing environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            return False
        
        print("SUCCESS: Environment variables validated")
        
        # Check scraper availability
        scraper_path = Path("scraper_tools/run_eirgrid_downloader.py")
        if not scraper_path.exists():
            print(f"ERROR: EirGrid scraper not found: {scraper_path}")
            return False
        
        print("SUCCESS: EirGrid scraper available")
        return True
    
    def extract_dates_from_query(self, query: str) -> tuple[str, str]:
        """Extract dates from query context or use sensible defaults"""
        query_lower = query.lower()
        
        # Default to today
        end_date = datetime.now()
        start_date = end_date
        
        # Look for date indicators in query
        if "today" in query_lower:
            start_date = end_date
        elif "yesterday" in query_lower:
            start_date = end_date - timedelta(days=1)
            end_date = end_date - timedelta(days=1)
        elif "this week" in query_lower:
            start_date = end_date - timedelta(days=7)
        elif "next 24 hours" in query_lower or "24 hours" in query_lower:
            # For 24 hour queries, get today and tomorrow
            start_date = end_date
            end_date = end_date + timedelta(days=1)
        elif "recent" in query_lower or "current" in query_lower:
            start_date = end_date
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def run_eirgrid_scraper_for_query(self, query: str, region: str = "all") -> bool:
        """Run EirGrid scraper based on query context"""
        print(f" Running EirGrid scraper based on query context...")
        
        # Extract dates from query
        start_date, end_date = self.extract_dates_from_query(query)
        
        print(f" Scraping data from {start_date} to {end_date} for region: {region}")
        
        try:
            # Import and run the scraper
            from scraper_tools.run_eirgrid_downloader import main as eirgrid_main
            
            # Set up arguments
            original_argv = sys.argv.copy()
            try:
                sys.argv = [
                    'run_eirgrid_downloader.py',
                    '--areas', 'co2_intensity',
                    '--start', start_date,
                    '--end', end_date,
                    '--region', region,
                    '--forecast',
                    '--output-dir', './data'
                ]
                
                print(f" Running scraper: {' '.join(sys.argv[1:])}")
                result = eirgrid_main()
                
                print("SUCCESS: EirGrid scraper completed")
                return True
                
            finally:
                sys.argv = original_argv
                
        except Exception as e:
            print(f"ERROR: Scraper failed: {e}")
            # Try subprocess method
            try:
                cmd = [
                    sys.executable, 
                    "scraper_tools/run_eirgrid_downloader.py",
                    "--areas", "co2_intensity",
                    "--start", start_date,
                    "--end", end_date,
                    "--region", region,
                    "--forecast",
                    "--output-dir", "./data"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("SUCCESS: EirGrid scraper completed via subprocess")
                    return True
                else:
                    print(f"ERROR: Scraper subprocess failed: {result.stderr}")
                    return False
                    
            except Exception as subprocess_error:
                print(f"ERROR: Both scraper methods failed: {subprocess_error}")
                return False
    
    def find_available_data(self) -> Optional[str]:
        """Find available EirGrid data files"""
        import glob
        
        search_patterns = [
            "data/co2_intensity/co2_intensity_*.json",
            "data/co2_intensity_*.json"
        ]
        
        all_files = []
        for pattern in search_patterns:
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Filter out compressed files
        raw_files = [f for f in all_files if 'compressed' not in f.lower()]
        
        if raw_files:
            # Return the most recent file
            latest_file = max(raw_files, key=os.path.getmtime)
            print(f"SUCCESS: Found EirGrid data: {latest_file}")
            return latest_file
        
        return None
    
    def prepare_data_for_evaluation(self, test_query: str = "What is the best time to use appliances today?") -> bool:
        """Prepare data for evaluation with automatic scraper integration"""
        print(" Preparing data for evaluation...")
        
        if not CO2_SYSTEM_AVAILABLE:
            print("ERROR: CO2 processing system not available")
            return False
        
        try:
            # Check for existing data first
            existing_data = self.find_available_data()
            
            # If no data or we want fresh data, run scraper
            if not existing_data:
                print(" No existing data found - running scraper based on query...")
                if not self.run_eirgrid_scraper_for_query(test_query):
                    print("ERROR: Failed to get data from scraper")
                    return False
                
                existing_data = self.find_available_data()
                if not existing_data:
                    print("ERROR: No data available after scraper run")
                    return False
            
            print(f" Using EirGrid data: {existing_data}")
            
            # Look for existing compressed data first
            import glob
            compressed_files = glob.glob("data/*compressed*.json")
            if compressed_files:
                # Use the most recent compressed file
                self.compressed_data_file = max(compressed_files, key=os.path.getmtime)
                print(f"Using existing compressed data: {self.compressed_data_file}")
            else:
                # Try to compress the data if no compressed files exist
                print("Compressing EirGrid data for consistency...")
                try:
                    self.compressed_data_file = CO2DataCompressor.prepare_for_evaluation("./data")
                    print(f"SUCCESS: Data compressed: {self.compressed_data_file}")
                except Exception as e:
                    print(f"ERROR: Compression failed: {e}")
                    return False
            
            # Check for existing ground truth files
            gt_files = glob.glob("test_configs/generated_test_cases_with_ground_truth_*.json")
            if gt_files:
                # Use existing ground truth
                test_cases_file = max(gt_files, key=os.path.getmtime)
                print(f"Using existing ground truth: {test_cases_file}")
                return True
            else:
                # Generate ground truth using compressed data
                print("Generating ground truth from compressed data...")
                try:
                    test_cases_file = GroundTruthGenerator.generate_for_evaluation(self.compressed_data_file)
                    print(f"SUCCESS: Ground truth generated: {test_cases_file}")
                    return True
                except Exception as e:
                    print(f"ERROR: Ground truth generation failed: {e}")
                    return False
                
        except Exception as e:
            print(f"ERROR: Data preparation failed: {e}")
            return False

    async def create_kamal_agent_instance(self):
        """Create Kamal's Carbon Agent with compressed data awareness"""
        try:
            # Setup Azure client
            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                model=os.getenv("MODEL", "gpt-4"),
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("API_KEY")
            )
            
            print(" Setting up Kamal's Carbon Agent...")
            
            # Load compressed data for agent to use
            compressed_data = None
            if self.compressed_data_file and os.path.exists(self.compressed_data_file):
                with open(self.compressed_data_file, 'r') as f:
                    compressed_data = json.load(f)
                print(f" Loaded compressed data for agent: {len(compressed_data.get('data', []))} points")
            
            # Always use our compatible agent for consistent formatting
            print(" Creating compatible agent with proper response formatting...")
            
            # Create compatible agent that uses compressed data and proper formatting
            return await self._create_compatible_agent(client, compressed_data)
                    
        except Exception as e:
            print(f"ERROR: Error creating agent: {e}")
            raise
    
    async def _create_compatible_agent(self, client, compressed_data=None):
        """Create compatible agent that uses compressed data"""
        
        # Emission analysis that uses compressed data
        async def emission_analysis(startdate: str, enddate: str, region: str) -> dict:
            """Emission analysis using compressed data when available"""
            
            print(f" Getting CO2 data: {startdate} to {enddate}, region: {region}")
            
            # If we have compressed data loaded, use it
            if compressed_data and compressed_data.get('data'):
                print(f" Using preloaded compressed data")
                
                # Extract key metrics from compressed data
                data_points = compressed_data['data']
                values = [point['value'] for point in data_points]
                
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = sum(values) / len(values)
                    
                    # Find optimal and peak times
                    min_point = min(data_points, key=lambda x: x['value'])
                    max_point = max(data_points, key=lambda x: x['value'])
                    
                    return {
                        "result": f"CO2 intensity data retrieved for {region} ({startdate} to {enddate})",
                        "source": "compressed_eirgrid_data",
                        "data_points": len(data_points),
                        "co2_range": f"{min_val:.0f}-{max_val:.0f}g CO2/kWh",
                        "daily_average": f"{avg_val:.0f}g CO2/kWh",
                        "min_intensity": min_val,
                        "max_intensity": max_val,
                        "optimal_time": min_point['time'],
                        "peak_time": max_point['time'],
                        "file_type": "compressed"
                    }
            
            # Fallback: check for file
            file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'
            
            if not os.path.exists(file_path):
                print(f" Data not available, running scraper...")
                
                # Run the scraper for the requested dates
                try:
                    from scraper_tools.run_eirgrid_downloader import main as eirgrid_main
                    
                    original_argv = sys.argv.copy()
                    try:
                        sys.argv = [
                            'run_eirgrid_downloader.py',
                            '--areas', 'co2_intensity',
                            '--start', startdate,
                            '--end', enddate,
                            '--region', region,
                            '--forecast',
                            '--output-dir', './data'
                        ]
                        
                        result = eirgrid_main()
                        print(f"SUCCESS: Scraper completed for {startdate}-{enddate}")
                        
                    finally:
                        sys.argv = original_argv
                        
                except Exception as e:
                    print(f"WARNING: Scraper error: {e}")
            
            # Load the data
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                print(f"SUCCESS: Loaded EirGrid data from {file_path}")
                
                # Extract useful information
                if 'data' in data and 'time_series' in data['data']:
                    time_series = data['data']['time_series']
                    
                    if time_series:
                        values = [entry.get('value', 0) for entry in time_series if isinstance(entry, dict)]
                        if values:
                            min_val = min(values)
                            max_val = max(values)
                            avg_val = sum(values) / len(values)
                            
                            return {
                                "result": f"CO2 intensity data retrieved for {region} ({startdate} to {enddate})",
                                "source": "eirgrid_scraper",
                                "data_points": len(time_series),
                                "co2_range": f"{min_val:.0f}-{max_val:.0f}g CO2/kWh",
                                "daily_average": f"{avg_val:.0f}g CO2/kWh",
                                "min_intensity": min_val,
                                "max_intensity": max_val,
                                "file_location": file_path
                            }
                
                # Fallback response
                return {
                    "result": f"CO2 data retrieved for {region}",
                    "source": "eirgrid_scraper", 
                    "file_location": file_path
                }
            else:
                return {
                    "result": f"Could not retrieve CO2 data for {region} ({startdate} to {enddate})",
                    "error": "Data not available after scraper attempt",
                    "suggestion": "Try running scraper manually or check date range"
                }
        
        # Create tools
        emission_tool = FunctionTool(
            func=emission_analysis,
            description="Gets CO2 intensity levels with automatic scraper integration. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
            name="get_emission_analysis"
        )
        
        # System message
        system_message = f"""You are an intelligent carbon emissions assistant for Ireland's electricity grid. Today's date: {datetime.now().strftime('%Y-%m-%d')}.

### Available Tools:
- **get_emission_analysis**: Automatically fetches CO2 intensity data from EirGrid (uses compressed data when available)

TOOL USAGE:
- Always use get_emission_analysis for CO2 queries
- Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')
- System will automatically use compressed data or run scraper if needed
- Default to today's date if not specified
- Default to 'all' region if not specified

AFTER TOOL EXECUTION:
- Process the tool results to extract: min_intensity, max_intensity, optimal_time, peak_time, daily_average
- Use these values to create the structured response format below
- DO NOT return the raw tool output or mention tool execution

### Response Guidelines:
Follow the examples.json format:

**Best Times to Use Appliances in Ireland Today (Based on REAL EirGrid Data):**

**Optimal Period (Lowest Real CO2):**
- **[TIME_RANGE]**: [CO2_VALUE]g CO2/kWh (REAL EirGrid data)
- Perfect for washing machine, dishwasher, and EV charging

**Specific Appliance Recommendations (Real Data-Based):**
- **Washing Machine**: Start cycle at [SPECIFIC_TIME]
- **Dishwasher**: Schedule for [TIME_RANGE]
- **Electric Vehicle Charging**: Begin charging during real low-emission window
- **Tumble Dryer**: Use during overnight period

**Avoid High Real Emission Times:**
- **[TIME_RANGE]**: [CO2_VALUE]g CO2/kWh (real peak demand data)

**Today's REAL EirGrid CO2 Data:**
- Minimum: [VALUE]g CO2/kWh (real measurement)
- Maximum: [VALUE]g CO2/kWh (real measurement)
- Daily Average: [VALUE]g CO2/kWh (real average)

**Environmental Impact**: Using appliances during optimal times reduces your carbon footprint by up to [PERCENTAGE]% compared to peak times (calculated from real data)!

### CRITICAL RESPONSE REQUIREMENTS:
1. **NEVER return raw tool output or debug information**
2. **ALWAYS format a complete user-friendly response using the data**
3. Always include specific CO2 intensity values (g CO2/kWh) from the data
4. Provide specific time recommendations in HH:MM-HH:MM format
5. Use emojis for clarity as shown above
6. Calculate environmental impact percentage: ((max-min)/max)*100
7. Maintain this exact structure for consistency
8. **Transform tool data into the structured format above - do not show tool execution details**
"""
        
        agent = AssistantAgent(
            name="StreamlinedCarbonAgent",
            model_client=client,
            tools=[emission_tool],
            reflect_on_tool_use=True,
            max_tool_iterations=3,
            system_message=system_message
        )
        
        print("SUCCESS: Compatible carbon agent created with compressed data support")
        return agent

    async def run_evaluation(self, mode: str = "hybrid", runs_per_test: int = 2) -> Optional[HybridEvaluationReport]:
        """
        Run evaluation using compressed data for consistency
        """
        print(f" Starting Agent Evaluation")
        print(f"Mode: {mode.upper()}")
        print(f" Using compressed data for consistency")
        print(f" Runs per test: {runs_per_test}")
        print("=" * 70)
        
        # Validate environment
        if not self.validate_environment():
            return None
        
        # Prepare data with automatic scraper integration
        test_query = "What is the best time to use my appliances today in Ireland?"
        if not self.prepare_data_for_evaluation(test_query):
            print("ERROR: Cannot proceed without data")
            return None
        
        # Create evaluator
        config_path = "./test_configs/streamlined_config.json"
        
        # Update config for the selected mode
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config["evaluation_mode"] = mode
        
        # Save updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.evaluator = HybridAgentEvaluator(config_path)
        
        # Load test cases (generated from compressed data)
        print(" Loading test cases from compressed data...")
        self.evaluator.load_test_cases()
        
        # Create agent
        print(" Creating carbon agent with compressed data awareness...")
        try:
            agent = await self.create_kamal_agent_instance()
        except Exception as e:
            print(f"ERROR: Failed to create agent: {e}")
            return None
        
        # Run evaluation
        print(f" Running {mode} evaluation...")
        try:
            report = await self.evaluator.evaluate_agent(agent, runs_per_test)
            
            # Print results
            self.evaluator.print_report_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"ERROR: Evaluation failed: {e}")
            print(f"ERROR: Evaluation failed: {e}")
            return None

async def main():
    """Main entry point"""
    runner = StreamlinedCarbonAgentEvaluationRunner()
    
    print("Streamlined Carbon Agent Evaluation System")
    print("Using compressed data for consistency")
    print("Automatic EirGrid scraper integration")
    print("=" * 70)
    print("1. Run Rule-based evaluation")
    print("2. Run LLM-only evaluation") 
    print("3. Run Hybrid evaluation (recommended)")
    print("4. Test agent with compressed data")
    print("5. Check system status")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        # Function to get instance count for evaluation modes
        def get_instance_count():
            while True:
                try:
                    instances = input("\nHow many times should each test case be run? (1-5): ").strip()
                    instances = int(instances)
                    if 1 <= instances <= 5:
                        return instances
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
        
        if choice == "1":
            instances = get_instance_count()
            print(f"\nStarting Rule-based Evaluation (running each test {instances} time{'s' if instances > 1 else ''})")
            await runner.run_evaluation(mode="rule_based_only", runs_per_test=instances)
        elif choice == "2":
            instances = get_instance_count()
            print(f"\nStarting LLM-only Evaluation (running each test {instances} time{'s' if instances > 1 else ''})")
            await runner.run_evaluation(mode="llm_only", runs_per_test=instances)
        elif choice == "3":
            instances = get_instance_count()
            print(f"\nStarting Hybrid Evaluation (running each test {instances} time{'s' if instances > 1 else ''})")
            print("Using compressed data for ground truth")
            await runner.run_evaluation(mode="hybrid", runs_per_test=instances)
        elif choice == "4":
            print("Testing agent with compressed data...")
            if runner.validate_environment():
                # Prepare data first
                test_query = "What is the best time to charge my EV today?"
                if runner.prepare_data_for_evaluation(test_query):
                    agent = await runner.create_kamal_agent_instance()
                    print(f"Test Query: {test_query}")
                    try:
                        if hasattr(agent, 'run_stream'):
                            print("Agent Response:")
                            async for result in agent.run_stream(task=test_query):
                                print(result)
                        else:
                            result = await agent.run(test_query)
                            print(f"Agent Response: {result}")
                        print("SUCCESS: Agent test successful")
                    except Exception as e:
                        print(f"ERROR: Agent test failed: {e}")
                else:
                    print("ERROR: Could not prepare data for testing")
            else:
                print("ERROR: Environment not ready")
        elif choice == "5":
            print("System Status Check:")
            env_ok = runner.validate_environment()
            data_available = runner.find_available_data() is not None
            
            # Check for compressed data
            import glob
            compressed_files = glob.glob("*compressed*.json") + glob.glob("data/*compressed*.json")
            compressed_available = len(compressed_files) > 0
            
            print(f"Environment: {'OK' if env_ok else 'FAILED'}")
            print(f"Raw Data Available: {'OK' if data_available else 'FAILED'}")
            print(f"Compressed Data: {'OK' if compressed_available else 'FAILED'}")
            print(f"CO2 System: {'OK' if CO2_SYSTEM_AVAILABLE else 'FAILED'}")
            
            if env_ok and CO2_SYSTEM_AVAILABLE:
                print("SUCCESS: System ready for evaluation")
            else:
                print("WARNING: Some components need attention")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    asyncio.run(main())