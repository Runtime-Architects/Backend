import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
import traceback
from enum import Enum
import opentelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"

class AgentBehavior(Enum):
    CORRECT_FUNCTION_CALL = "correct_function_call"
    INCORRECT_FUNCTION_CALL = "incorrect_function_call"
    MISSING_FUNCTION_CALL = "missing_function_call"
    CONSISTENT_OUTPUT = "consistent_output"
    INCONSISTENT_OUTPUT = "inconsistent_output"
    PROPER_ERROR_HANDLING = "proper_error_handling"

@dataclass
class TestCase:
    """Represents a single test case for agent evaluation"""
    id: str
    query: str
    expected_functions: List[str]
    expected_behavior: List[AgentBehavior]
    expected_output_keywords: List[str]
    category: str
    priority: str = "medium"
    timeout_seconds: int = 60
    
@dataclass
class EvaluationResult:
    """Results from a single test case execution"""
    test_case_id: str
    status: EvaluationStatus
    execution_time: float
    functions_called: List[str]
    output_text: str
    behaviors_observed: List[AgentBehavior]
    consistency_score: float
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AgentEvaluationReport:
    """Comprehensive evaluation report for an agent"""
    agent_name: str
    evaluation_date: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    timeout_tests: int
    avg_execution_time: float
    consistency_score: float
    function_call_accuracy: float
    behavior_scores: Dict[str, float]
    detailed_results: List[EvaluationResult]
    recommendations: List[str]

class AgentEvaluator:
    """
    Systematic agent evaluation framework for testing consistency, 
    function calling accuracy, and behavioral patterns
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.test_cases: List[TestCase] = []
        self.results: List[EvaluationResult] = []
        self.setup_telemetry()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load evaluation configuration"""
        default_config = {
            "max_retries": 3,
            "timeout_seconds": 60,
            "consistency_threshold": 0.8,
            "telemetry_endpoint": None,
            "output_dir": "./evaluation_results"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def setup_telemetry(self):
        """Setup OpenTelemetry for agent behavior tracking"""
        if self.config.get("telemetry_endpoint"):
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config["telemetry_endpoint"]
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = tracer
        else:
            self.tracer = None
    
    def load_test_cases(self, test_cases_path: str):
        """Load test cases from JSON file"""
        with open(test_cases_path, 'r') as f:
            test_data = json.load(f)
            
        self.test_cases = [
            TestCase(**case) for case in test_data["test_cases"]
        ]
        logger.info(f"Loaded {len(self.test_cases)} test cases")
    
    def add_test_case(self, test_case: TestCase):
        """Add a single test case"""
        self.test_cases.append(test_case)
    
    async def evaluate_agent(self, agent_instance, runs_per_test: int = 3) -> AgentEvaluationReport:
        """
        Evaluate agent with multiple runs per test for consistency analysis
        """
        logger.info(f"Starting evaluation with {len(self.test_cases)} test cases")
        logger.info(f"Running {runs_per_test} iterations per test case")
        
        all_results = []
        
        for test_case in self.test_cases:
            logger.info(f"Executing test case: {test_case.id}")
            
            # Run the same test multiple times to check consistency
            test_runs = []
            for run in range(runs_per_test):
                logger.info(f"  Run {run + 1}/{runs_per_test}")
                result = await self._execute_test_case(agent_instance, test_case, run)
                test_runs.append(result)
                all_results.append(result)
                
                # Small delay between runs
                await asyncio.sleep(1)
            
            # Analyze consistency across runs
            consistency_score = self._calculate_consistency(test_runs)
            logger.info(f"  Consistency score: {consistency_score:.2f}")
            
            # Update all runs with consistency score
            for result in test_runs:
                result.consistency_score = consistency_score
        
        # Generate comprehensive report
        report = self._generate_report(agent_instance.__class__.__name__, all_results)
        
        # Save results
        self._save_results(report)
        
        return report
    
    async def _execute_test_case(self, agent_instance, test_case: TestCase, run_number: int) -> EvaluationResult:
        """Execute a single test case and capture detailed metrics"""
        
        start_time = time.time()
        
        if self.tracer:
            with self.tracer.start_as_current_span(f"test_case_{test_case.id}_run_{run_number}") as span:
                span.set_attribute("test_case.id", test_case.id)
                span.set_attribute("test_case.category", test_case.category)
                span.set_attribute("run_number", run_number)
        
        try:
            # Execute agent with timeout
            result = await asyncio.wait_for(
                self._run_agent_task(agent_instance, test_case.query),
                timeout=test_case.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            # Analyze the result
            functions_called = self._extract_functions_called(result)
            behaviors_observed = self._analyze_behaviors(result, test_case)
            status = self._determine_status(test_case, functions_called, behaviors_observed, result)
            
            return EvaluationResult(
                test_case_id=test_case.id,
                status=status,
                execution_time=execution_time,
                functions_called=functions_called,
                output_text=str(result),
                behaviors_observed=behaviors_observed,
                consistency_score=0.0  # Will be updated later
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return EvaluationResult(
                test_case_id=test_case.id,
                status=EvaluationStatus.TIMEOUT,
                execution_time=execution_time,
                functions_called=[],
                output_text="",
                behaviors_observed=[],
                consistency_score=0.0,
                error_message="Test case timeout"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                test_case_id=test_case.id,
                status=EvaluationStatus.ERROR,
                execution_time=execution_time,
                functions_called=[],
                output_text="",
                behaviors_observed=[],
                consistency_score=0.0,
                error_message=str(e)
            )
    
    async def _run_agent_task(self, agent_instance, query: str):
        """Run the agent task and capture output"""
        # This method needs to be adapted based on our specific agent implementation
        # For now, assuming the agent has a run method that returns results
        if hasattr(agent_instance, 'run_stream'):
            # Handle streaming agents
            results = []
            async for result in agent_instance.run_stream(task=query):
                results.append(result)
            return results
        elif hasattr(agent_instance, 'run'):
            return await agent_instance.run(query)
        else:
            # Fallback for basic callable agents
            return await agent_instance(query)
    
    def _extract_functions_called(self, result) -> List[str]:
        """Extract function calls from agent output"""
        functions_called = []
        
        # This is a simplified extraction - we'll need to adapt based on our agent's output format
        result_str = str(result)
        
        # Look for common function patterns
        function_patterns = [
            "get_emission_analysis",
            "PythonCodeExecutionTool",
            "emission_tool"
        ]
        
        for pattern in function_patterns:
            if pattern in result_str:
                functions_called.append(pattern)
        
        return functions_called
    
    def _analyze_behaviors(self, result, test_case: TestCase) -> List[AgentBehavior]:
        """Analyze agent behaviors from the execution"""
        behaviors = []
        result_str = str(result).lower()
        
        # Check if expected functions were called
        expected_functions = [f.lower() for f in test_case.expected_functions]
        called_functions = [f.lower() for f in self._extract_functions_called(result)]
        
        if all(func in str(called_functions) for func in expected_functions):
            behaviors.append(AgentBehavior.CORRECT_FUNCTION_CALL)
        else:
            behaviors.append(AgentBehavior.INCORRECT_FUNCTION_CALL)
        
        # Check for expected output keywords
        if all(keyword.lower() in result_str for keyword in test_case.expected_output_keywords):
            behaviors.append(AgentBehavior.CONSISTENT_OUTPUT)
        else:
            behaviors.append(AgentBehavior.INCONSISTENT_OUTPUT)
        
        # Check for error handling
        if "error" in result_str or "exception" in result_str:
            if "handled" in result_str or "resolved" in result_str:
                behaviors.append(AgentBehavior.PROPER_ERROR_HANDLING)
        
        return behaviors
    
    def _determine_status(self, test_case: TestCase, functions_called: List[str], 
                         behaviors_observed: List[AgentBehavior], result) -> EvaluationStatus:
        """Determine pass/fail status based on test criteria"""
        
        # Check if all expected functions were called
        expected_functions = test_case.expected_functions
        if expected_functions and not all(func in str(functions_called) for func in expected_functions):
            return EvaluationStatus.FAIL
        
        # Check if expected behaviors are present
        required_behaviors = test_case.expected_behavior
        if required_behaviors and not any(behavior in behaviors_observed for behavior in required_behaviors):
            return EvaluationStatus.FAIL
        
        # Check for output keywords
        result_str = str(result).lower()
        if test_case.expected_output_keywords:
            if not all(keyword.lower() in result_str for keyword in test_case.expected_output_keywords):
                return EvaluationStatus.FAIL
        
        return EvaluationStatus.PASS
    
    def _calculate_consistency(self, test_runs: List[EvaluationResult]) -> float:
        """Calculate consistency score across multiple runs"""
        if len(test_runs) < 2:
            return 1.0
        
        # Compare functions called
        function_consistency = self._compare_function_calls(test_runs)
        
        # Compare output similarity
        output_consistency = self._compare_outputs(test_runs)
        
        # Compare execution times (should be relatively stable)
        time_consistency = self._compare_execution_times(test_runs)
        
        # Weighted average
        return (function_consistency * 0.4 + output_consistency * 0.4 + time_consistency * 0.2)
    
    def _compare_function_calls(self, test_runs: List[EvaluationResult]) -> float:
        """Compare function calls across runs"""
        if not test_runs:
            return 0.0
        
        first_run_functions = set(test_runs[0].functions_called)
        consistent_count = 0
        
        for run in test_runs[1:]:
            if set(run.functions_called) == first_run_functions:
                consistent_count += 1
        
        return consistent_count / max(1, len(test_runs) - 1)
    
    def _compare_outputs(self, test_runs: List[EvaluationResult]) -> float:
        """Compare output similarity across runs"""
        if len(test_runs) < 2:
            return 1.0
        
        # Simple similarity based on common keywords
        # We might want to use more sophisticated text similarity metrics -> Need to improve by using an LLM - Hybrid approach
        outputs = [run.output_text.lower() for run in test_runs]
        
        # Get common words across all outputs
        word_sets = [set(output.split()) for output in outputs]
        common_words = set.intersection(*word_sets)
        total_unique_words = set.union(*word_sets)
        
        if not total_unique_words:
            return 1.0
        
        return len(common_words) / len(total_unique_words)
    
    def _compare_execution_times(self, test_runs: List[EvaluationResult]) -> float:
        """Compare execution time consistency"""
        if len(test_runs) < 2:
            return 1.0
        
        times = [run.execution_time for run in test_runs]
        avg_time = statistics.mean(times)
        
        if avg_time == 0:
            return 1.0
        
        # Calculate coefficient of variation (lower is more consistent)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        cv = std_dev / avg_time
        
        # Convert to consistency score (higher is better)
        return max(0, 1 - cv)
    
    def _generate_report(self, agent_name: str, results: List[EvaluationResult]) -> AgentEvaluationReport:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return AgentEvaluationReport(
                agent_name=agent_name,
                evaluation_date=datetime.now().isoformat(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=0,
                timeout_tests=0,
                avg_execution_time=0.0,
                consistency_score=0.0,
                function_call_accuracy=0.0,
                behavior_scores={},
                detailed_results=[],
                recommendations=[]
            )
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == EvaluationStatus.PASS])
        failed_tests = len([r for r in results if r.status == EvaluationStatus.FAIL])
        error_tests = len([r for r in results if r.status == EvaluationStatus.ERROR])
        timeout_tests = len([r for r in results if r.status == EvaluationStatus.TIMEOUT])
        
        avg_execution_time = statistics.mean([r.execution_time for r in results])
        avg_consistency_score = statistics.mean([r.consistency_score for r in results])
        
        # Calculate function call accuracy
        function_call_accuracy = len([r for r in results if AgentBehavior.CORRECT_FUNCTION_CALL in r.behaviors_observed]) / total_tests
        
        # Calculate behavior scores
        behavior_scores = {}
        for behavior in AgentBehavior:
            count = len([r for r in results if behavior in r.behaviors_observed])
            behavior_scores[behavior.value] = count / total_tests
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, behavior_scores)
        
        return AgentEvaluationReport(
            agent_name=agent_name,
            evaluation_date=datetime.now().isoformat(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            timeout_tests=timeout_tests,
            avg_execution_time=avg_execution_time,
            consistency_score=avg_consistency_score,
            function_call_accuracy=function_call_accuracy,
            behavior_scores=behavior_scores,
            detailed_results=results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: List[EvaluationResult], behavior_scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        recommendations = []
        
        # Function call accuracy recommendations
        if behavior_scores.get("correct_function_call", 0) < 0.8:
            recommendations.append(
                "Function call accuracy is low. Consider improving system prompts to better specify when and how to use tools."
            )
        
        # Consistency recommendations
        low_consistency_results = [r for r in results if r.consistency_score < 0.7]
        if len(low_consistency_results) > len(results) * 0.3:
            recommendations.append(
                "Agent shows inconsistent behavior across runs. Consider adding more specific instructions and examples to prompts."
            )
        
        # Timeout recommendations
        timeout_count = len([r for r in results if r.status == EvaluationStatus.TIMEOUT])
        if timeout_count > 0:
            recommendations.append(
                f"{timeout_count} tests timed out. Consider optimizing agent logic or increasing timeout limits."
            )
        
        # Error handling recommendations
        if behavior_scores.get("proper_error_handling", 0) < 0.5:
            recommendations.append(
                "Improve error handling in agent responses. Add more robust exception handling and user-friendly error messages."
            )
        
        return recommendations
    
    def _save_results(self, report: AgentEvaluationReport):
        """Save evaluation results to files"""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = output_dir / f"evaluation_report_{report.agent_name}_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save CSV for detailed analysis
        csv_file = output_dir / f"detailed_results_{report.agent_name}_{timestamp}.csv"
        df = pd.DataFrame([asdict(result) for result in report.detailed_results])
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {json_file} and {csv_file}")
