import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Import the hybrid evaluation framework
from hybrid_agent_evaluator import HybridAgentEvaluator, HybridEvaluationReport
from evaluation_strategies import EvaluationMode

# Import required libraries for Azure setup
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific logger levels for debugging
evaluation_logger = logging.getLogger('hybrid_agent_evaluator')
combiner_logger = logging.getLogger('result_combiner')

class HybridCarbonAgentEvaluationRunner:
    """
    Enhanced Carbon Agent Evaluation Runner with Hybrid Capabilities
    """
    
    def __init__(self):
        load_dotenv()
        self.setup_test_environment()
        self.evaluator = None
        
    def setup_test_environment(self):
        """Setup the test environment and create necessary directories"""
        Path("./evaluation_results").mkdir(exist_ok=True)
        Path("./test_configs").mkdir(exist_ok=True)
        
        # Create configuration files if they don't exist
        self.create_default_configs()
    
    def create_default_configs(self):
        """Create default configuration files"""
        
        # Hybrid evaluation configuration
        hybrid_config_path = Path("./test_configs/hybrid_config.json")
        if not hybrid_config_path.exists():
            hybrid_config = {
                "evaluation_mode": "hybrid",
                "llm_evaluation_enabled": True,
                "llm_on_failures_only": False,
                "max_retries": 3,
                "timeout_seconds": 120,
                
                # Weighting
                "rule_weight": 0.4,
                "llm_weight": 0.6,
                "function_weight": 0.4,
                "keyword_weight": 0.3,
                "behavior_weight": 0.3,
                
                # LLM settings
                "llm_quality_threshold": 0.7,
                "llm_confidence_threshold": 0.6,
                "llm_override_enabled": True,
                "llm_override_threshold": 0.8,
                
                # Performance
                "consistency_threshold": 0.8,
                "output_dir": "./evaluation_results",
                "parallel_llm_calls": False,
                "max_llm_tokens": 2000,
                "llm_timeout": 30
            }
            
            with open(hybrid_config_path, 'w') as f:
                json.dump(hybrid_config, f, indent=2)
            
            print(f"✅ Created hybrid configuration: {hybrid_config_path}")
        
        # Test cases configuration
        test_cases_path = Path("./test_configs/carbon_agent_tests.json")
        if not test_cases_path.exists():
            self.create_enhanced_test_cases()
    
    def create_enhanced_test_cases(self):
        """Create test cases with LLM-specific fields"""
        config = {
            "test_cases": [
                {
                    "id": "carbon_001_basic",
                    "query": "What is the best time to use my appliances today in Ireland?",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "high_quality_response", "user_friendly"],
                    "expected_output_keywords": ["time", "appliances", "Ireland", "co2", "intensity"],
                    "category": "basic_query",
                    "priority": "high",
                    "timeout_seconds": 60,
                    "domain_context": "Carbon emissions optimization for appliance usage in Ireland",
                    "available_functions": ["get_emission_analysis"]
                },
                {
                    "id": "carbon_002_ev_charging",
                    "query": "When should I charge my electric vehicle for minimal carbon footprint?",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "high_quality_response", "domain_expertise"],
                    "expected_output_keywords": ["charge", "carbon", "footprint", "time", "EV"],
                    "category": "ev_charging",
                    "priority": "high",
                    "timeout_seconds": 60,
                    "domain_context": "EV charging optimization based on carbon intensity",
                    "available_functions": ["get_emission_analysis"]
                },
                {
                    "id": "carbon_003_data_retrieval",
                    "query": "Show me CO2 intensity data for the next 24 hours in Republic of Ireland",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "high_quality_response"],
                    "expected_output_keywords": ["co2", "intensity", "24 hours", "ireland", "data"],
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
                    "expected_behavior": ["proper_error_handling", "user_friendly"],
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
                    "expected_behavior": ["correct_function_call", "domain_expertise", "high_quality_response"],
                    "expected_output_keywords": ["carbon", "intensity", "roi", "northern ireland", "compare"],
                    "category": "comparison_query",
                    "priority": "medium",
                    "timeout_seconds": 90,
                    "domain_context": "Comparative analysis of carbon intensity across regions",
                    "available_functions": ["get_emission_analysis"]
                },
                {
                    "id": "carbon_006_analysis",
                    "query": "Give me a statistical summary of last week's carbon emissions in Ireland",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "high_quality_response", "domain_expertise"],
                    "expected_output_keywords": ["statistical", "summary", "carbon", "emission", "ireland", "week"],
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
                },
                {
                    "id": "carbon_008_optimization",
                    "query": "How can I reduce my household's carbon footprint using smart appliance scheduling?",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "high_quality_response", "user_friendly", "domain_expertise"],
                    "expected_output_keywords": ["reduce", "carbon", "footprint", "household", "smart", "scheduling"],
                    "category": "optimization_query",
                    "priority": "high",
                    "timeout_seconds": 90,
                    "domain_context": "Carbon footprint optimization through smart scheduling",
                    "available_functions": ["get_emission_analysis"]
                },
                {
                    "id": "carbon_009_consistency",
                    "query": "What are the carbon emission patterns in Ireland during peak hours?",
                    "expected_functions": ["get_emission_analysis"],
                    "expected_behavior": ["correct_function_call", "high_quality_response", "domain_expertise"],
                    "expected_output_keywords": ["carbon", "emission", "patterns", "ireland", "peak", "hours"],
                    "category": "consistency_test",
                    "priority": "medium",
                    "timeout_seconds": 90,
                    "domain_context": "Analysis of carbon emission patterns during peak usage periods",
                    "available_functions": ["get_emission_analysis"]
                }
            ]
        }
        
        config_path = Path("./test_configs/carbon_agent_tests.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created enhanced test cases: {config_path}")
    
    def validate_environment(self) -> bool:
        """Validate environment variables and setup"""
        required_vars = {
            "AZURE_DEPLOYMENT": "Azure OpenAI deployment name",
            "AZURE_ENDPOINT": "Azure OpenAI endpoint URL", 
            "API_KEY": "Azure OpenAI API key",
            "API_VERSION": "Azure OpenAI API version"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                missing_vars.append(f"{var}: {description}")
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print("\n💡 Create a .env file with these variables:")
            print("AZURE_DEPLOYMENT=your-deployment-name")
            print("AZURE_ENDPOINT=https://your-resource.openai.azure.com/")
            print("API_KEY=your-api-key") 
            print("API_VERSION=2024-02-01")
            print("MODEL=gpt-4")
            return False
        
        print("✅ Environment variables validated")
        return True
    
    async def create_agent_instance(self):
        """Create an instance of the Carbon Agent for testing"""
        try:
            # Setup Azure client
            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                model=os.getenv("MODEL", "gpt-4"),
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("API_KEY")
            )
            
            # Import Kamal's carbon agent configuration
            try:
                from azurecarbonagent import emission_tool, system_message
                tools = [emission_tool] if emission_tool else []
                system_msg = system_message
                logger.info("✅ Using azurecarbonagent configuration")
            except ImportError as e:
                logger.warning(f"Could not import azurecarbonagent: {e}")
                logger.info("Using fallback configuration")
                tools = []
                system_msg = """You are a helpful AI assistant specialized in carbon emissions analysis and sustainability. 
                You can help users understand carbon footprints, emissions data, and sustainability metrics.
                When users ask about carbon-related topics, provide helpful, accurate information."""
            
            # Create agent instance
            agent = AssistantAgent(
                name="hybrid_carbon_assistant",
                model_client=client,
                tools=tools,
                reflect_on_tool_use=True,
                system_message=system_msg
            )
            
            logger.info("✅ Agent instance created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"❌ Error creating agent instance: {e}")
            raise
    
    async def run_evaluation(self, mode: str = "hybrid", runs_per_test: int = 3) -> Optional[HybridEvaluationReport]:
        """
        Run evaluation with specified mode
        
        Args:
            mode: "rule_based", "llm_only", or "hybrid"
            runs_per_test: Number of runs per test case
        """
        print(f"🚀 Starting {mode.upper()} Agent Evaluation")
        print("=" * 70)
        
        # Validate environment
        if not self.validate_environment():
            return None
        
        # Create evaluator with the specified mode
        config_path = "./test_configs/hybrid_config.json"
        
        # Update config for the selected mode
        with open(config_path, 'r') as f:
            config = json.load(f)
        config["evaluation_mode"] = mode
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.evaluator = HybridAgentEvaluator(config_path)
        
        # Load test cases
        test_cases_path = "./test_configs/carbon_agent_tests.json"
        if not Path(test_cases_path).exists():
            print(f"❌ Test cases file not found: {test_cases_path}")
            return None
        
        self.evaluator.load_test_cases(test_cases_path)
        
        # Create agent instance
        print("📦 Creating agent instance...")
        try:
            agent = await self.create_agent_instance()
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            return None
        
        # Run evaluation
        print(f"🧪 Running {mode} evaluation with {runs_per_test} runs per test...")
        try:
            # Enable debug logging for better diagnostics
            if logger.level <= logging.INFO:
                logging.getLogger('hybrid_agent_evaluator').setLevel(logging.DEBUG)
                logging.getLogger('result_combiner').setLevel(logging.DEBUG)
            
            report = await self.evaluator.evaluate_agent(agent, runs_per_test)
            
            # Print results
            self.print_evaluation_results(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            print(f"❌ Evaluation failed: {e}")
            return None
    
    def print_evaluation_results(self, report: HybridEvaluationReport):
        """Print comprehensive evaluation results"""
        self.evaluator.print_report_summary(report)
        
        # Additional analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        
        # Agreement analysis (if hybrid mode)
        if report.evaluation_mode == "hybrid":
            from result_combiner import ResultCombiner
            combiner = ResultCombiner(self.evaluator.config)
            agreement_metrics = combiner.calculate_agreement_metrics(report.detailed_results)
            
            if agreement_metrics.get("samples", 0) > 0:
                print(f"🤝 Rule-LLM Agreement: {agreement_metrics['agreement_rate']:.2f}")
                print(f"📊 Score Correlation: {agreement_metrics['score_agreement']:.2f}")
                print(f"🔄 LLM Override Rate: {agreement_metrics['override_rate']:.2f}")
        
        # Performance insights
        slow_tests = [r for r in report.detailed_results if r.execution_time > 30]
        if slow_tests:
            print(f"⏰ Slow tests (>30s): {len(slow_tests)}")
            for test in slow_tests[:3]:
                print(f"  • {test.test_case_id}: {test.execution_time:.1f}s")
        
        # Tool interaction analysis
        tool_stats = {}
        error_count = 0
        for result in report.detailed_results:
            tools_used = result.structured_output.get("summary", {}).get("tools_used", [])
            has_errors = result.structured_output.get("summary", {}).get("has_errors", False)
            
            for tool in tools_used:
                tool_stats[tool] = tool_stats.get(tool, 0) + 1
            
            if has_errors:
                error_count += 1
        
        if tool_stats:
            print(f"🔧 Tool Usage:")
            for tool, count in tool_stats.items():
                print(f"  • {tool}: {count} times")
        
        if error_count > 0:
            print(f"⚠️  Tests with errors: {error_count}")
            # Show sample errors
            error_tests = [r for r in report.detailed_results if r.structured_output.get("summary", {}).get("has_errors", False)]
            for test in error_tests[:2]:
                errors = test.structured_output.get("errors", [])
                if errors:
                    print(f"  • {test.test_case_id}: {errors[0].get('message', 'Unknown error')}")
        
        # Response quality analysis
        response_lengths = []
        for result in report.detailed_results:
            length = result.structured_output.get("summary", {}).get("response_length", 0)
            if length > 0:
                response_lengths.append(length)
        
        if response_lengths:
            import statistics
            print(f"📝 Response Analysis:")
            print(f"  • Average response length: {statistics.mean(response_lengths):.0f} chars")
            print(f"  • Response length range: {min(response_lengths)}-{max(response_lengths)} chars")
        
        # Quality distribution
        quality_scores = [r.final_score for r in report.detailed_results if r.final_score > 0]
        if quality_scores:
            import statistics
            print(f"📈 Quality Distribution:")
            print(f"  • Mean: {statistics.mean(quality_scores):.2f}")
            print(f"  • Median: {statistics.median(quality_scores):.2f}")
            if len(quality_scores) > 1:
                print(f"  • Std Dev: {statistics.stdev(quality_scores):.2f}")
    
    async def run_prompt_engineering_cycle(self, mode: str = "hybrid"):
        """
        Enhanced prompt engineering cycle with hybrid evaluation
        """
        print(f"🔄 Starting Enhanced Prompt Engineering Cycle ({mode.upper()})")
        print("=" * 70)
        
        iteration = 1
        max_iterations = 5
        target_pass_rate = 0.90
        target_quality_score = 0.80
        
        best_report = None
        best_score = 0.0
        
        while iteration <= max_iterations:
            print(f"\n🔄 ITERATION {iteration}/{max_iterations}")
            print("-" * 40)
            
            # Run evaluation
            report = await self.run_evaluation(mode=mode, runs_per_test=2)
            
            if report is None:
                print("❌ Cannot continue without evaluation results")
                break
            
            # Calculate metrics
            pass_rate = report.pass_rate()
            combined_score = report.avg_combined_score
            
            print(f"\n📊 ITERATION RESULTS:")
            print(f"Pass Rate: {pass_rate:.2%} (target: {target_pass_rate:.0%})")
            print(f"Quality Score: {combined_score:.2f} (target: {target_quality_score:.2f})")
            
            # Track best performance
            overall_score = (pass_rate + combined_score) / 2
            if overall_score > best_score:
                best_score = overall_score
                best_report = report
                print(f"🏆 New best overall score: {overall_score:.2f}")
            
            # Check if targets achieved
            if pass_rate >= target_pass_rate and combined_score >= target_quality_score:
                print(f"\n🎉 TARGETS ACHIEVED!")
                print(f"✅ Pass rate: {pass_rate:.2%}")
                print(f"✅ Quality score: {combined_score:.2f}")
                break
            
            # Analyze failures and provide insights
            failed_results = [r for r in report.detailed_results if r.status.value == "FAIL"]
            
            print(f"\n📝 FAILURE ANALYSIS ({len(failed_results)} failures):")
            
            # Categorize failures
            failure_categories = {}
            for result in failed_results:
                category = self._categorize_failure(result)
                if category not in failure_categories:
                    failure_categories[category] = []
                failure_categories[category].append(result)
            
            for category, failures in failure_categories.items():
                print(f"  • {category}: {len(failures)} cases")
                # Show example
                if failures:
                    example = failures[0]
                    print(f"    Example: {example.test_case_id}")
                    if example.improvement_suggestions:
                        print(f"    Suggestion: {example.improvement_suggestions[0]}")
            
            # Mode-specific insights
            if mode == "hybrid":
                self._print_hybrid_insights(report)
            elif mode == "llm_only":
                self._print_llm_insights(report)
            else:
                self._print_rule_insights(report)
            
            # Pause for improvements
            if iteration < max_iterations:
                print(f"\n⏸️  Review the analysis above and update Kamal's agent configuration.")
                print(f"Press Enter to continue with iteration {iteration + 1}, or Ctrl+C to exit...")
                try:
                    input()
                except KeyboardInterrupt:
                    print("\n👋 Exiting prompt engineering cycle")
                    break
            
            iteration += 1
        
        # Final summary
        print(f"\n🏁 PROMPT ENGINEERING CYCLE COMPLETED")
        print("=" * 50)
        if best_report:
            print(f"🏆 Best Performance Achieved:")
            print(f"  • Pass Rate: {best_report.pass_rate():.2%}")
            print(f"  • Quality Score: {best_report.avg_combined_score:.2f}")
            print(f"  • Overall Score: {best_score:.2f}")
        
        return best_report
    
    def _categorize_failure(self, result) -> str:
        """Categorize failure reasons"""
        if not result.functions_called:
            return "Missing Function Calls"
        elif result.llm_quality_score < 0.5:
            return "Low Quality Response"
        elif result.consistency_score < 0.6:
            return "Inconsistent Behavior"
        elif "error" in result.output_text.lower():
            return "Error Handling Issues"
        else:
            return "Other Issues"
    
    def _print_hybrid_insights(self, report: HybridEvaluationReport):
        """Print insights specific to hybrid evaluation"""
        print(f"\n🔍 HYBRID INSIGHTS:")
        
        # Agreement analysis
        agreements = len([r for r in report.detailed_results 
                         if r.llm_status is not None and r.rule_based_status == r.llm_status])
        total_with_llm = len([r for r in report.detailed_results if r.llm_status is not None])
        
        if total_with_llm > 0:
            agreement_rate = agreements / total_with_llm
            print(f"  • Rule-LLM Agreement: {agreement_rate:.2%}")
            
            if agreement_rate < 0.7:
                print(f"  ⚠️  Low agreement suggests evaluation criteria misalignment")
        
        # LLM override analysis
        overrides = len([r for r in report.detailed_results 
                        if r.llm_status != r.rule_based_status and r.status == r.llm_status])
        if overrides > 0:
            print(f"  • LLM overrode rule-based decisions: {overrides} times")
    
    def _print_llm_insights(self, report: HybridEvaluationReport):
        """Print insights specific to LLM-only evaluation"""
        print(f"\n🧠 LLM-ONLY INSIGHTS:")
        
        # Confidence analysis
        confidences = [r.llm_confidence for r in report.detailed_results if r.llm_confidence > 0]
        if confidences:
            import statistics
            avg_confidence = statistics.mean(confidences)
            print(f"  • Average LLM Confidence: {avg_confidence:.2f}")
            
            if avg_confidence < 0.6:
                print(f"  ⚠️  Low confidence suggests unclear or ambiguous responses")
        
        # Quality distribution
        qualities = [r.llm_quality_score for r in report.detailed_results if r.llm_quality_score > 0]
        if qualities:
            import statistics
            print(f"  • Quality Score Range: {min(qualities):.2f} - {max(qualities):.2f}")
    
    def _print_rule_insights(self, report: HybridEvaluationReport):
        """Print insights specific to rule-based evaluation"""
        print(f"\n📏 RULE-BASED INSIGHTS:")
        
        # Function call analysis
        total_tests = len(report.detailed_results)
        with_functions = len([r for r in report.detailed_results if r.functions_called])
        
        print(f"  • Tests with function calls: {with_functions}/{total_tests}")
        
        # Keyword matching analysis
        keyword_scores = []
        for result in report.detailed_results:
            if hasattr(result, 'keyword_matches') and hasattr(result, 'total_keywords'):
                if result.total_keywords > 0:
                    keyword_scores.append(result.keyword_matches / result.total_keywords)
        
        if keyword_scores:
            import statistics
            avg_keyword_score = statistics.mean(keyword_scores)
            print(f"  • Average keyword match rate: {avg_keyword_score:.2%}")
    
    def quick_test(self, query: str = None):
        """Run a quick single test for debugging"""
        if query is None:
            query = "What is the best time to use appliances in Ireland today?"
        
        print(f"🔍 Quick Test: '{query}'")
        print("-" * 50)
        
        # This would be implemented to run a single query through the agent
        # For now, just show what would happen
        print("This would run the query through our agents and show:")
        print("• Raw agent output")
        print("• Rule-based analysis") 
        print("• LLM analysis (if enabled)")
        print("• Combined result")
        print("\nImplement this by running a single test case...")
    
    def view_conversation_details(self, report, test_case_id: str = None):
        """View detailed conversation flow for debugging"""
        if not report:
            print("No report available")
            return
        
        if test_case_id:
            # Show specific test case
            results = [r for r in report.detailed_results if r.test_case_id == test_case_id]
            if not results:
                print(f"Test case '{test_case_id}' not found")
                return
            result = results[0]
        else:
            # Show first test case
            if not report.detailed_results:
                print("No detailed results available")
                return
            result = report.detailed_results[0]
        
        print(f"\n🔍 CONVERSATION DETAILS: {result.test_case_id}")
        print("=" * 60)
        
        # Conversation flow
        print(f"📋 Conversation Flow:")
        for step in result.conversation_flow:
            step_num = step.get("step", "?")
            step_type = step.get("type", "unknown")
            if step_type == "user_input":
                print(f"  {step_num}. 👤 User: {step.get('content', '')}")
            elif step_type == "tool_call":
                status = step.get("status", "unknown")
                function = step.get("function", "unknown")
                status_emoji = "✅" if status == "success" else "❌" if status == "error" else "⚠️"
                print(f"  {step_num}. 🔧 Tool: {function} {status_emoji}")
            elif step_type == "agent_response":
                length = step.get("length", 0)
                preview = step.get("preview", "")
                print(f"  {step_num}. 🤖 Agent: ({length} chars) {preview}")
        
        # Tool interactions
        if result.tool_interactions:
            print(f"\n🔧 Tool Interactions:")
            for i, tool in enumerate(result.tool_interactions, 1):
                function_name = tool.get("function_name", "unknown")
                arguments = tool.get("arguments", "")
                tool_result = tool.get("result", {})
                status = tool_result.get("status", "unknown")
                
                print(f"  {i}. Function: {function_name}")
                print(f"     Arguments: {arguments}")
                print(f"     Status: {status}")
                
                if tool_result.get("is_error"):
                    error_content = tool_result.get("content", "Unknown error")
                    print(f"     Error: {error_content}")
                else:
                    result_content = tool_result.get("content", "")
                    preview = result_content[:100] + "..." if len(result_content) > 100 else result_content
                    print(f"     Result: {preview}")
        
        # Agent responses
        if result.agent_responses:
            print(f"\n🤖 Agent Responses:")
            for i, response in enumerate(result.agent_responses, 1):
                print(f"  Response {i} ({len(response)} chars):")
                # Show first few lines
                lines = response.split('\n')[:5]
                for line in lines:
                    print(f"    {line}")
                if len(response.split('\n')) > 5:
                    print(f"    ... ({len(response.split('\n')) - 5} more lines)")
        
        # Summary
        summary = result.structured_output.get("summary", {})
        print(f"\n📊 Summary:")
        print(f"  • Tool calls: {summary.get('total_tool_calls', 0)}")
        print(f"  • Agent responses: {summary.get('total_agent_responses', 0)}")
        print(f"  • Has errors: {summary.get('has_errors', False)}")
        print(f"  • Response length: {summary.get('response_length', 0)} chars")
        print(f"  • Tools used: {', '.join(summary.get('tools_used', []))}")
        
        # Evaluation results
        print(f"\n🏆 Evaluation Results:")
        status_val = result.status.value if hasattr(result.status, 'value') else str(result.status)
        print(f"  • Final Status: {status_val}")
        print(f"  • Final Score: {result.final_score:.2f}")
        print(f"  • Rule Score: {result.rule_based_score:.2f}")
        print(f"  • LLM Score: {result.llm_quality_score:.2f}")
        print(f"  • Confidence: {result.confidence_level}")
        
        if result.improvement_suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in result.improvement_suggestions[:3]:
                print(f"  • {suggestion}")

async def main():
    """Main entry point"""
    runner = HybridCarbonAgentEvaluationRunner()
    
    print("🔄 Hybrid Agent Evaluation System")
    print("=" * 40)
    print("1. Run Rule-based evaluation only")
    print("2. Run LLM-only evaluation") 
    print("3. Run Hybrid evaluation (recommended)")
    print("4. Start prompt engineering cycle")
    print("5. Quick test")
    print("6. Validate environment")
    print("7. View conversation details (after running evaluation)")
    
    try:
        choice = input("\nSelect option (1-7): ").strip()
        
        last_report = None  # Store last report for viewing
        
        if choice == "1":
            last_report = await runner.run_evaluation(mode="rule_based_only", runs_per_test=3)
        elif choice == "2":
            last_report = await runner.run_evaluation(mode="llm_only", runs_per_test=3)
        elif choice == "3":
            last_report = await runner.run_evaluation(mode="hybrid", runs_per_test=3)
        elif choice == "4":
            mode = input("Prompt engineering mode (rule_based/llm_only/hybrid) [hybrid]: ").strip() or "hybrid"
            last_report = await runner.run_prompt_engineering_cycle(mode=mode)
        elif choice == "5":
            custom_query = input("Enter query (or press Enter for default): ").strip()
            runner.quick_test(custom_query if custom_query else None)
        elif choice == "6":
            runner.validate_environment()
        elif choice == "7":
            test_id = input("Enter test case ID to view (or press Enter for first test): ").strip()
            # For demo purposes, we'd need a stored report
            print("💡 Run an evaluation first (options 1-4) to generate conversation details to view")
            print("This feature shows detailed conversation flow, tool interactions, and agent responses")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    asyncio.run(main())