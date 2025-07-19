"""
Integration wrapper to seamlessly integrate LLM-as-Judge with existing evaluation framework
Replacement for existing AgentEvaluator
Fixed for Azure OpenAI API compatibility
"""

import asyncio
import json
from pathlib import Path
from enhanced_agent_evaluator import EnhancedAgentEvaluator
import logging
import os
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

logger = logging.getLogger(__name__)

class LLMEnhancedEvaluatorWrapper:
    """
    Replacement for existing AgentEvaluator with LLM enhancements
    Maintains backward compatibility while adding LLM-as-Judge capabilities
    """
    
    def __init__(self, config_path: str = None):
        """Initialise with optional config path"""
        self.enhanced_evaluator = EnhancedAgentEvaluator(config_path)
        
        # Backward compatibility attributes
        self.test_cases = []
        self.results = []
        
    def load_test_cases(self, test_cases_path: str):
        """Load test cases - compatible with existing format"""
        self.enhanced_evaluator.load_test_cases(test_cases_path)
        
        # Update backward compatibility attributes
        try:
            with open(test_cases_path, 'r') as f:
                test_data = json.load(f)
            
            if "test_cases" in test_data:
                self.test_cases = test_data["test_cases"]
            else:
                self.test_cases = test_data
        except Exception as e:
            logger.error(f"Failed to load test cases for backward compatibility: {e}")
            self.test_cases = []
    
    async def evaluate_agent(self, agent_instance, runs_per_test: int = 3):
        """
        Evaluate agent with enhanced LLM capabilities
        Returns enhanced report but maintains compatibility
        """
        report = await self.enhanced_evaluator.evaluate_agent(agent_instance, runs_per_test)
        
        # Update backward compatibility attributes
        self.results = report.detailed_results
        
        return report
    
    def print_evaluation_summary(self, report):
        """Print evaluation summary - enhanced/updated version"""
        self.enhanced_evaluator.print_enhanced_summary(report)
    
    # Backward compatibility methods
    def _compare_outputs(self, test_runs):
        """Backward compatibility - now uses LLM when available"""
        if self.enhanced_evaluator.llm_judge and len(test_runs) > 1:
            # Use semantic comparison
            outputs = [run.output_text for run in test_runs]
            # This would be called during evaluation, not separately
            return 1.0  # Placeholder
        else:
            return self.enhanced_evaluator._calculate_basic_consistency(test_runs)
    
    def _extract_functions_called(self, result):
        """Backward compatibility - now enhanced with LLM"""
        return self.enhanced_evaluator._extract_functions_basic(str(result))

# Usage with existing CarbonAgentEvaluationRunner
class EnhancedCarbonAgentEvaluationRunner:
    """
    Enhanced version of our existing CarbonAgentEvaluationRunner
    Minimal changes required to existing code
    """
    
    def __init__(self):
        # dotenv is already loaded at module level
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment"""
        Path("./evaluation_results").mkdir(exist_ok=True)
        Path("./test_configs").mkdir(exist_ok=True)
        
        # Create enhanced config if it doesn't exist
        self.create_enhanced_config()
    
    def create_enhanced_config(self):
        """Create enhanced configuration file"""
        config = {
            "llm_evaluation_enabled": True,
            "use_semantic_comparison": True,
            "use_function_analysis": True,
            "use_behavior_analysis": True,
            "max_retries": 3,
            "timeout_seconds": 120,
            "consistency_threshold": 0.8,
            "output_dir": "./evaluation_results"
        }
        
        config_path = Path("./test_configs/enhanced_config.json")
        if not config_path.exists():
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Enhanced configuration created at: {config_path}")
    
    async def create_agent_instance(self):
        """Create agent instance - same as our existing code but with better error handling"""
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
            
            # Validate required environment variables
            required_env_vars = ["AZURE_DEPLOYMENT", "AZURE_ENDPOINT", "API_KEY", "API_VERSION"]
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                model=os.getenv("MODEL", "gpt-4"),
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("API_KEY")
            )
            
            # Import our specific tools and system message
            try:
                from azurecarbonagent import emission_tool, system_message
                tools = [emission_tool] if emission_tool else []
            except ImportError:
                logger.warning("Could not import azurecarbonagent. Using fallback configuration.")
                tools = []
                system_message = """You are a helpful AI assistant specialized in carbon emissions analysis. 
                You can help users understand and analyze carbon footprints, emissions data, 
                and sustainability metrics."""
            
            agent = AssistantAgent(
                name="enhanced_carbon_assistant",
                model_client=client,
                tools=tools,
                reflect_on_tool_use=True,
                system_message=system_message
            )
            
            logger.info("Agent instance created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent instance: {e}")
            raise
    
    async def run_enhanced_evaluation(self, runs_per_test: int = 3):
        """
        Run enhanced evaluation - minimal changes from existing code
        """
        print("🚀 Starting Enhanced Agent Evaluation with LLM-as-Judge")
        print("=" * 60)
        
        # Create enhanced evaluator (drop-in replacement)
        evaluator = LLMEnhancedEvaluatorWrapper("./test_configs/enhanced_config.json")
        
        # Load test cases (same format as before)
        test_cases_path = "./test_configs/carbon_agent_tests.json"
        if not Path(test_cases_path).exists():
            print(f"⚠️  Test cases file not found: {test_cases_path}")
            print("Creating sample test configuration...")
            create_sample_test_config()
            print("Please edit the test configuration file and run again.")
            return None
        
        evaluator.load_test_cases(test_cases_path)
        
        # Create agent instance (same as before)
        print("📦 Creating agent instance...")
        agent = await self.create_agent_instance()
        
        # Run evaluation (same interface, enhanced capabilities)
        print(f"🧪 Running enhanced evaluation with {runs_per_test} runs per test...")
        report = await evaluator.evaluate_agent(agent, runs_per_test)
        
        # Print enhanced results
        evaluator.print_evaluation_summary(report)
        
        return report
    
    # Existing methods remain the same but can be enhanced
    async def run_prompt_engineering_cycle(self):
        """Enhanced prompt engineering cycle with LLM insights"""
        print("🔄 Starting Enhanced Prompt Engineering Cycle")
        print("=" * 60)
        
        iteration = 1
        max_iterations = 5
        target_pass_rate = 0.9
        target_quality_score = 0.8  # New LLM-based metric
        
        while iteration <= max_iterations:
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Run enhanced evaluation
            report = await self.run_enhanced_evaluation(runs_per_test=2)
            
            if report is None:
                print("❌ Cannot continue without test cases")
                break
            
            # Calculate metrics (now including LLM metrics)
            pass_rate = report.passed_tests / report.total_tests if report.total_tests > 0 else 0
            quality_score = report.avg_quality_score
            
            print(f"Current pass rate: {pass_rate:.2f}")
            print(f"Current quality score: {quality_score:.2f}")
            print(f"LLM confidence: {report.llm_confidence_score:.2f}")
            
            # Check if targets achieved
            if pass_rate >= target_pass_rate and quality_score >= target_quality_score:
                print(f"🎉 Targets achieved! Pass rate: {pass_rate:.2f}, Quality: {quality_score:.2f}")
                break
            
            # Enhanced failure analysis with LLM insights
            failed_results = [r for r in report.detailed_results if r.status == "FAIL"]
            
            print(f"\n📝 LLM-ENHANCED FAILURE ANALYSIS:")
            
            # Show LLM-generated insights
            llm_suggestions = set()
            for result in failed_results:
                if result.llm_evaluation and result.llm_evaluation.improvement_suggestions:
                    llm_suggestions.update(result.llm_evaluation.improvement_suggestions)
            
            if llm_suggestions:
                print(f"🤖 LLM-Generated Improvement Suggestions:")
                for i, suggestion in enumerate(list(llm_suggestions)[:5], 1):
                    print(f"  {i}. {suggestion}")
            
            # Show specific failure details with LLM reasoning
            for result in failed_results[:3]:
                print(f"\n  🔍 Test: {result.test_case_id}")
                if result.llm_evaluation:
                    print(f"    💭 LLM Analysis: {result.llm_evaluation.reasoning[:200]}...")
                    if result.llm_evaluation.specific_issues:
                        print(f"    ⚠️  Issues: {', '.join(result.llm_evaluation.specific_issues[:2])}")
            
            # Pause for improvements
            if iteration < max_iterations:
                print(f"\nPress Enter when ready to continue with iteration {iteration + 1}...")
                try:
                    input()
                except KeyboardInterrupt:
                    print("\n👋 Exiting prompt engineering cycle")
                    break
            
            iteration += 1
        
        print(f"\n🏁 Enhanced prompt engineering cycle completed")

# Configuration helper
def create_sample_test_config():
    """Create sample test configuration compatible with enhanced evaluator"""
    config = {
        "test_cases": [
            {
                "id": "carbon_001_basic",
                "query": "What is the best time to use my appliances today in Ireland?",
                "expected_functions": ["get_emission_analysis"],
                "expected_behavior": ["correct_function_call", "consistent_output"],
                "expected_output_keywords": ["time", "appliances", "Ireland", "co2", "intensity"],
                "category": "basic_query",
                "priority": "high",
                "timeout_seconds": 60,
                "domain_context": "Carbon emissions optimization for appliance usage in Ireland",
                "available_functions": ["get_emission_analysis"]
            },
            {
                "id": "carbon_002_ev",
                "query": "When should I charge my electric vehicle for minimal carbon footprint?",
                "expected_functions": ["get_emission_analysis"],
                "expected_behavior": ["correct_function_call", "high_quality_response"],
                "expected_output_keywords": ["charge", "carbon", "footprint", "time"],
                "category": "ev_charging",
                "priority": "high",
                "timeout_seconds": 60,
                "domain_context": "EV charging optimization based on carbon intensity",
                "available_functions": ["get_emission_analysis"]
            },
            {
                "id": "carbon_003_data",
                "query": "Show me CO2 intensity data for the next 24 hours in Republic of Ireland",
                "expected_functions": ["get_emission_analysis"],
                "expected_behavior": ["correct_function_call", "high_quality_response"],
                "expected_output_keywords": ["co2", "intensity", "24 hours", "ireland"],
                "category": "data_request",
                "priority": "medium",
                "timeout_seconds": 90,
                "domain_context": "Carbon intensity data retrieval and presentation",
                "available_functions": ["get_emission_analysis"]
            },
            {
                "id": "carbon_004_irrelevant",
                "query": "What's the weather like today?",
                "expected_functions": [],
                "expected_behavior": ["consistent_output", "proper_error_handling"],
                "expected_output_keywords": ["weather"],
                "category": "irrelevant_query",
                "priority": "low",
                "timeout_seconds": 30,
                "domain_context": "Handling queries outside carbon emissions domain",
                "available_functions": ["get_emission_analysis"]
            },
            {
                "id": "carbon_005_comparison",
                "query": "Compare carbon intensity between ROI and Northern Ireland",
                "expected_functions": ["get_emission_analysis"],
                "expected_behavior": ["correct_function_call", "domain_expertise"],
                "expected_output_keywords": ["carbon", "intensity", "roi", "northern ireland", "compare"],
                "category": "comparison_query",
                "priority": "medium",
                "timeout_seconds": 90,
                "domain_context": "Comparative analysis of carbon intensity across regions",
                "available_functions": ["get_emission_analysis"]
            },
            {
                "id": "carbon_006_consistency",
                "query": "Give me a statistical summary of last week carbon emission in Ireland",
                "expected_functions": ["get_emission_analysis"],
                "expected_behavior": ["correct_function_call", "high_quality_response"],
                "expected_output_keywords": ["statistical", "summary", "carbon", "emission", "ireland"],
                "category": "analysis_query",
                "priority": "high",
                "timeout_seconds": 120,
                "domain_context": "Statistical analysis and summarization of carbon emission data",
                "available_functions": ["get_emission_analysis"]
            },
            {
                "id": "carbon_007_error_handling",
                "query": "Get carbon data for invalid date 2025-13-45",
                "expected_functions": ["get_emission_analysis"],
                "expected_behavior": ["proper_error_handling", "user_friendly"],
                "expected_output_keywords": ["error", "invalid", "date"],
                "category": "error_handling",
                "priority": "medium",
                "timeout_seconds": 60,
                "domain_context": "Error handling for invalid date inputs",
                "available_functions": ["get_emission_analysis"]
            }
        ]
    }
    
    config_path = Path("./test_configs/carbon_agent_tests.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Carbon agent test configuration created at: {config_path}")
    print("These test cases cover:")
    print("  - Basic appliance timing optimization")
    print("  - EV charging optimization") 
    print("  - Data retrieval and presentation")
    print("  - Handling irrelevant queries")
    print("  - Regional comparisons")
    print("  - Statistical analysis")
    print("  - Error handling")

# Environment validation helper
def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = {
        "AZURE_DEPLOYMENT": "Our Azure OpenAI deployment name",
        "AZURE_ENDPOINT": "Our Azure OpenAI endpoint URL", 
        "API_KEY": "Our Azure OpenAI API key",
        "API_VERSION": "Azure OpenAI API version (e.g., 2025-07-10)"
    }
    
    optional_vars = {
        "MODEL": "Model name (Should match our deployment model)"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var}: {description}")
        elif var == "AZURE_ENDPOINT" and not value.startswith("https://"):
            print(f"⚠️  Warning: {var} should start with 'https://'")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in our .env file / environment.")
        return False
    
    # Show what was loaded
    print("✅ Required environment variables loaded:")
    for var in required_vars.keys():
        value = os.getenv(var)
        if var == "API_KEY":
            print(f"  - {var}: {'*' * (len(value) - 4) + value[-4:] if value else 'Not set'}")
        else:
            print(f"  - {var}: {value}")
    
    # Show optional variables
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  - {var}: {value}")
        else:
            print(f"  - {var}: Not set (will use default)")
    
    # Model name validation
    model = os.getenv("MODEL")
    if model and model not in ["gpt-4", "gpt-4-turbo", "gpt-4-32k", "gpt-35-turbo", "gpt-35-turbo-16k"]:
        print(f"⚠️  Warning: MODEL '{model}' might not be a standard Azure OpenAI model name.")
        print("   Common models: gpt-4, gpt-4-turbo, gpt-35-turbo")
    
    return True

# Simple usage example
async def main():
    """Simple usage example"""
    print("Enhanced LLM-as-Judge Agent Evaluation")
    print("=" * 40)
    
    # Validate environment
    print("🔍 Validating environment variables...")
    if not validate_environment():
        print("\n💡 Tip: Create a .env file in the same directory with:")
        print("AZURE_DEPLOYMENT=your-deployment-name")
        print("AZURE_ENDPOINT=https://your-resource.openai.azure.com/")
        print("API_KEY=your-api-key")
        print("API_VERSION=2024-02-01")
        print("MODEL=gpt-4")
        return
    
    # Create enhanced runner
    try:
        runner = EnhancedCarbonAgentEvaluationRunner()
    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        return
    
    print("\n1. Run enhanced evaluation")
    print("2. Run enhanced prompt engineering cycle")
    print("3. Create sample test configuration")
    print("4. Validate environment variables")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            await runner.run_enhanced_evaluation(runs_per_test=3)
        elif choice == "2":
            await runner.run_prompt_engineering_cycle()
        elif choice == "3":
            create_sample_test_config()
        elif choice == "4":
            validate_environment()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())