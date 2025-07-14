import asyncio
import json
import sys
from pathlib import Path

# Import the evaluation framework and agent code
from agent_evaluation_framework import AgentEvaluator
from azurecarbonagent import main as carbon_agent_main
from azuremultiagent import main as multi_agent_main

# Import required libraries for Azure setup
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv

class CarbonAgentEvaluationRunner:
    """
    Comprehensive evaluation runner for Carbon Agent - Should meet Saeed's requirements
    """
    
    def __init__(self):
        load_dotenv()
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup the test environment and create necessary directories"""
        Path("./evaluation_results").mkdir(exist_ok=True)
        Path("./test_configs").mkdir(exist_ok=True)
        
        # Create test configuration if it doesn't exist
        self.create_test_config()
    
    def create_test_config(self):
        """Create test configuration file"""
        config = {
            "evaluation_config": {
                "max_retries": 3,
                "timeout_seconds": 120,
                "consistency_threshold": 0.8,
                "telemetry_endpoint": None,
                "output_dir": "./evaluation_results"
            },
            "test_cases": [
                {
                    "id": "carbon_001_basic",
                    "query": "What is the best time to use my appliances today in Ireland?",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "consistent_output"],
                    "expected_output_keywords": ["time", "appliances", "Ireland", "co2", "intensity"],
                    "category": "basic_query",
                    "priority": "high",
                    "timeout_seconds": 60
                },
                {
                    "id": "carbon_002_ev",
                    "query": "When should I charge my electric vehicle for minimal carbon footprint?",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call"],
                    "expected_output_keywords": ["charge", "carbon", "footprint", "time"],
                    "category": "ev_charging",
                    "priority": "high",
                    "timeout_seconds": 60
                },
                {
                    "id": "carbon_003_data",
                    "query": "Show me CO2 intensity data for the next 24 hours in Republic of Ireland",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call"],
                    "expected_output_keywords": ["co2", "intensity", "24 hours", "ireland"],
                    "category": "data_request",
                    "priority": "medium",
                    "timeout_seconds": 90
                },
                {
                    "id": "carbon_004_irrelevant",
                    "query": "What's the weather like today?",
                    "expected_functions": [],
                    "expected_behavior": ["consistent_output"],
                    "expected_output_keywords": ["weather"],
                    "category": "irrelevant_query",
                    "priority": "low",
                    "timeout_seconds": 30
                },
                {
                    "id": "carbon_005_comparison",
                    "query": "Compare carbon intensity between ROI and Northern Ireland",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call"],
                    "expected_output_keywords": ["carbon", "intensity", "roi", "northern ireland", "compare"],
                    "category": "comparison_query",
                    "priority": "medium",
                    "timeout_seconds": 90
                },
                {
                    "id": "carbon_006_consistency",
                    "query": "Give me a statistical summary of last week carbon emission in Ireland",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call"],
                    "expected_output_keywords": ["statistical", "summary", "carbon", "emission", "ireland"],
                    "category": "analysis_query",
                    "priority": "high",
                    "timeout_seconds": 120
                },
                {
                    "id": "carbon_007_error_handling",
                    "query": "Get carbon data for invalid date 2025-13-45",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["proper_error_handling"],
                    "expected_output_keywords": ["error", "invalid", "date"],
                    "category": "error_handling",
                    "priority": "medium",
                    "timeout_seconds": 60
                }
            ]
        }
        
        config_path = Path("./test_configs/carbon_agent_tests.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Test configuration created at: {config_path}")
    
    async def create_agent_instance(self):
        """Create an instance of the Carbon Agent for testing"""
        try:
            # Setup Azure client
            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                model=os.getenv("MODEL"),
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("API_KEY")
            )
            
            # Import the emission tool from Kamal's carbon agent
            from azurecarbonagent import emission_tool, system_message
            
            # Create agent instance
            agent = AssistantAgent(
                name="test_carbon_assistant",
                model_client=client,
                tools=[emission_tool],
                reflect_on_tool_use=True,
                system_message=system_message
            )
            
            return agent
            
        except Exception as e:
            print(f"Error creating agent instance: {e}")
            raise
    
    async def run_evaluation(self, runs_per_test: int = 3):
        """Run the complete evaluation suite"""
        print("🚀 Starting Carbon Agent Evaluation")
        print("=" * 60)
        
        # Create evaluator
        evaluator = AgentEvaluator("./test_configs/evaluation_config.json")
        
        # Load test cases
        evaluator.load_test_cases("./test_configs/carbon_agent_tests.json")
        
        # Create agent instance
        print("📦 Creating agent instance...")
        agent = await self.create_agent_instance()
        
        # Run evaluation
        print(f"🧪 Running evaluation with {runs_per_test} runs per test...")
        report = await evaluator.evaluate_agent(agent, runs_per_test)
        
        # Print results
        self.print_evaluation_summary(report)
        
        return report
    
    def print_evaluation_summary(self, report):
        """Print a detailed evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Agent: {report.agent_name}")
        print(f"Date: {report.evaluation_date}")
        print(f"Total Tests: {report.total_tests}")
        print(f"✅ Passed: {report.passed_tests} ({report.passed_tests/report.total_tests*100:.1f}%)")
        print(f"❌ Failed: {report.failed_tests} ({report.failed_tests/report.total_tests*100:.1f}%)")
        print(f"⚠️  Errors: {report.error_tests}")
        print(f"⏰ Timeouts: {report.timeout_tests}")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"Average Execution Time: {report.avg_execution_time:.2f}s")
        print(f"Consistency Score: {report.consistency_score:.2f}")
        print(f"Function Call Accuracy: {report.function_call_accuracy:.2f}")
        
        print(f"\n🎯 BEHAVIOR ANALYSIS:")
        for behavior, score in report.behavior_scores.items():
            print(f"  {behavior}: {score:.2f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Detailed test results
        print(f"\n📋 DETAILED RESULTS:")
        for result in report.detailed_results:
            status_emoji = "✅" if result.status.value == "PASS" else "❌"
            print(f"  {status_emoji} {result.test_case_id}: {result.status.value} "
                  f"({result.execution_time:.2f}s, consistency: {result.consistency_score:.2f})")
    
    async def run_prompt_engineering_cycle(self):
        """
        Implement the prompt engineering cycle that Saeed described:
        1. Test current prompt
        2. Analyze results
        3. Improve prompt
        4. Re-test
        """
        print("🔄 Starting Prompt Engineering Cycle")
        print("=" * 60)
        
        iteration = 1
        max_iterations = 5
        target_pass_rate = 0.9
        
        while iteration <= max_iterations:
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Run evaluation
            report = await self.run_evaluation(runs_per_test=2)
            
            # Calculate current pass rate
            pass_rate = report.passed_tests / report.total_tests
            print(f"Current pass rate: {pass_rate:.2f}")
            
            # Check if we've reached target
            if pass_rate >= target_pass_rate:
                print(f"🎉 Target pass rate achieved! ({pass_rate:.2f} >= {target_pass_rate})")
                break
            
            # Analyze failures and suggest improvements
            failed_results = [r for r in report.detailed_results if r.status.value == "FAIL"]
            
            print(f"\n📝 ANALYSIS OF {len(failed_results)} FAILURES:")
            failure_patterns = self.analyze_failure_patterns(failed_results)
            
            print("\n🛠️  SUGGESTED PROMPT IMPROVEMENTS:")
            improvements = self.suggest_prompt_improvements(failure_patterns)
            for imp in improvements:
                print(f"  • {imp}")
            
            # Ask user to update prompts
            if iteration < max_iterations:
                print(f"\n⏸️  Please update the system prompt based on the suggestions above.")
                print(f"Press Enter when ready to continue with iteration {iteration + 1}...")
                input()
            
            iteration += 1
        
        print(f"\n🏁 Prompt engineering cycle completed after {iteration-1} iterations")
    
    def analyze_failure_patterns(self, failed_results):
        """Analyze patterns in failed test results"""
        patterns = {
            "missing_function_calls": 0,
            "incorrect_output": 0,
            "timeout_issues": 0,
            "consistency_issues": 0
        }
        
        for result in failed_results:
            if not result.functions_called:
                patterns["missing_function_calls"] += 1
            
            if result.consistency_score < 0.7:
                patterns["consistency_issues"] += 1
            
            if "timeout" in result.error_message or "":
                patterns["timeout_issues"] += 1
            
            if len(result.output_text) < 50:  # Very short outputs might indicate issues
                patterns["incorrect_output"] += 1
        
        return patterns
    
    def suggest_prompt_improvements(self, patterns):
        """Suggest specific prompt improvements based on failure patterns - Could use an LLM"""
        improvements = []
        
        if patterns["missing_function_calls"] > 0:
            improvements.append(
                "Add more explicit instructions about when to use the emission_tool"
            )
            improvements.append(
                "Include examples of queries that should trigger function calls"
            )
        
        if patterns["consistency_issues"] > 0:
            improvements.append(
                "Add more structured response templates to improve consistency"
            )
            improvements.append(
                "Include specific format requirements for outputs"
            )
        
        if patterns["timeout_issues"] > 0:
            improvements.append(
                "Optimize the agent logic to reduce execution time"
            )
            improvements.append(
                "Add time-aware decision making in the prompt"
            )
        
        if patterns["incorrect_output"] > 0:
            improvements.append(
                "Add more detailed examples of expected output format"
            )
            improvements.append(
                "Include validation steps in the prompt"
            )
        
        return improvements


# Configuration file creator
def create_evaluation_config():
    """Create the evaluation configuration file"""
    config = {
        "max_retries": 3,
        "timeout_seconds": 120,
        "consistency_threshold": 0.8,
        "telemetry_endpoint": None,
        "output_dir": "./evaluation_results"
    }
    
    config_path = Path("./test_configs/evaluation_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Evaluation configuration created at: {config_path}")


async def main():
    """Main entry point for the evaluation"""
    runner = CarbonAgentEvaluationRunner()
    
    # Create configuration if needed
    create_evaluation_config()
    
    # Choose evaluation mode
    print("Carbon Agent Evaluation Framework")
    print("=" * 40)
    print("1. Run single evaluation")
    print("2. Run prompt engineering cycle")
    print("3. Create test configuration only")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        await runner.run_evaluation(runs_per_test=3)
    elif choice == "2":
        await runner.run_prompt_engineering_cycle()
    elif choice == "3":
        runner.create_test_config()
        print("Test configuration created successfully!")
    else:
        print("Invalid choice. Exiting.")
        return


if __name__ == "__main__":
    asyncio.run(main())