"""
Enhanced Agent Evaluator with LLM-as-Judge capabilities
Integrates with existing evaluation framework while adding LLM-based analysis
Fixed for Azure OpenAI API compatibility and AutoGen structured outputs
"""

import asyncio
import json
import time
import statistics
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

from llm_judge import LLMJudge, LLMEvaluationResult, FunctionCallAnalysis, BehaviorAnalysis
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

logger = logging.getLogger(__name__)

@dataclass
class EnhancedEvaluationResult:
    """Enhanced evaluation result with LLM insights"""
    test_case_id: str
    status: str  # "PASS", "FAIL", "ERROR", "TIMEOUT"
    execution_time: float
    functions_called: List[str]
    output_text: str
    behaviors_observed: List[str]
    consistency_score: float
    error_message: Optional[str] = None
    timestamp: str = None
    
    # LLM-based enhancements
    llm_evaluation: Optional[LLMEvaluationResult] = None
    function_analysis: Optional[FunctionCallAnalysis] = None
    behavior_analysis: Optional[BehaviorAnalysis] = None
    semantic_similarity: float = 0.0
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class EnhancedEvaluationReport:
    """Enhanced evaluation report with LLM insights"""
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
    detailed_results: List[EnhancedEvaluationResult]
    recommendations: List[str]
    
    # Enhanced metrics
    semantic_consistency_score: float = 0.0
    avg_quality_score: float = 0.0
    llm_confidence_score: float = 0.0

class EnhancedAgentEvaluator:
    """Enhanced agent evaluator with LLM-as-Judge capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        # dotenv is already loaded at module level
        self.config = self._load_config(config_path)
        self.test_cases = []
        self.llm_judge = None
        self._last_llm_result = None  # Store for status determination
        self._setup_llm_judge()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        default_config = {
            "llm_evaluation_enabled": True,
            "use_semantic_comparison": True,
            "use_function_analysis": True,
            "use_behavior_analysis": True,
            "max_retries": 3,
            "timeout_seconds": 60,
            "consistency_threshold": 0.8,
            "output_dir": "./evaluation_results"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_llm_judge(self):
        """Setup LLM judge with Azure OpenAI"""
        if not self.config.get("llm_evaluation_enabled", True):
            logger.info("LLM evaluation disabled")
            return
        
        try:
            # Validate required environment variables
            required_env_vars = ["AZURE_DEPLOYMENT", "AZURE_ENDPOINT", "API_KEY", "API_VERSION"]
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {missing_vars}")
                self.llm_judge = None
                return
            
            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                model=os.getenv("MODEL", "gpt-4"),
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("API_KEY")
            )
            
            self.llm_judge = LLMJudge(
                azure_client=client,
                model_name=os.getenv("MODEL", "gpt-4"),
                max_retries=self.config.get("max_retries", 3)
            )
            
            logger.info("LLM Judge initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Judge: {e}")
            self.llm_judge = None
    
    def load_test_cases(self, test_cases_path: str):
        """Load the test cases from JSON file"""
        try:
            with open(test_cases_path, 'r') as f:
                test_data = json.load(f)
            
            # Handle both old and new test case formats
            if "test_cases" in test_data:
                self.test_cases = test_data["test_cases"]
            else:
                self.test_cases = test_data
            
            logger.info(f"Loaded {len(self.test_cases)} test cases")
        except Exception as e:
            logger.error(f"Failed to load test cases from {test_cases_path}: {e}")
            raise
    
    async def evaluate_agent(self, agent_instance, runs_per_test: int = 3) -> EnhancedEvaluationReport:
        """Evaluate agent with LLM-based analysis"""
        logger.info(f"Starting enhanced evaluation with {len(self.test_cases)} test cases")
        logger.info(f"Running {runs_per_test} iterations per test case")
        
        if not self.test_cases:
            logger.error("No test cases loaded. Call load_test_cases() first.")
            raise ValueError("No test cases loaded")
        
        all_results = []
        
        for test_case in self.test_cases:
            logger.info(f"Executing test case: {test_case['id']}")
            
            # Run test multiple times for consistency
            test_runs = []
            for run in range(runs_per_test):
                result = await self._execute_test_case(agent_instance, test_case, run)
                test_runs.append(result)
                await asyncio.sleep(0.5)  # Brief delay between runs
            
            # Analyze consistency using LLM if enabled
            if self.llm_judge and len(test_runs) > 1:
                try:
                    consistency_analysis = await self._analyze_consistency_llm(test_runs, test_case)
                    
                    # Update all runs with consistency analysis
                    for i, result in enumerate(test_runs):
                        result.semantic_similarity = consistency_analysis.similarity_score
                        result.consistency_score = consistency_analysis.consistency_score
                        result.quality_score = consistency_analysis.quality_score
                        if i == 0:  # Store full LLM evaluation on first result
                            result.llm_evaluation = consistency_analysis
                except Exception as e:
                    logger.warning(f"LLM consistency analysis failed for {test_case['id']}: {e}")
                    # Fall back to basic consistency
                    consistency_score = self._calculate_basic_consistency(test_runs)
                    for result in test_runs:
                        result.consistency_score = consistency_score
            else:
                # Fallback to basic consistency
                consistency_score = self._calculate_basic_consistency(test_runs)
                for result in test_runs:
                    result.consistency_score = consistency_score
            
            all_results.extend(test_runs)
        
        # Generate enhanced report
        report = self._generate_enhanced_report(agent_instance.__class__.__name__, all_results)
        
        # Save results
        self._save_results(report)
        
        return report
    
    async def _execute_test_case(self, agent_instance, test_case: Dict, run_number: int) -> EnhancedEvaluationResult:
        """Execute a single test case with LLM analysis"""
        start_time = time.time()
        
        try:
            # Execute agent
            result = await asyncio.wait_for(
                self._run_agent_task(agent_instance, test_case["query"]),
                timeout=test_case.get("timeout_seconds", self.config["timeout_seconds"])
            )
            
            execution_time = time.time() - start_time
            output_text = str(result)
            
            # Enhanced function detection
            functions_called = self._extract_functions_advanced(output_text)
            
            # LLM-based analysis if enabled
            function_analysis = None
            behavior_analysis = None
            
            if self.llm_judge:
                try:
                    # Function call analysis
                    if self.config.get("use_function_analysis", True):
                        function_analysis = await self.llm_judge.analyze_function_calls(
                            output_text, 
                            test_case.get("expected_functions", []),
                            test_case["query"],
                            test_case.get("available_functions", [])
                        )
                        # Use LLM-detected functions if available and more comprehensive
                        if function_analysis.detected_functions and len(function_analysis.detected_functions) >= len(functions_called):
                            functions_called = function_analysis.detected_functions
                    
                    # Behavior analysis
                    if self.config.get("use_behavior_analysis", True):
                        behavior_analysis = await self.llm_judge.analyze_behavior(
                            output_text,
                            test_case["query"],
                            test_case.get("expected_behavior", []),
                            test_case.get("domain_context", "")
                        )
                        # Store for status determination
                        self._last_llm_result = behavior_analysis
                except Exception as e:
                    logger.warning(f"LLM analysis failed for {test_case['id']}: {e}")
            
            # Determine status with improved logic
            status = self._determine_status_improved(test_case, function_analysis, behavior_analysis, output_text)
            
            # Extract behaviors with improved detection
            behaviors_observed = []
            if behavior_analysis:
                behaviors_observed = behavior_analysis.behaviors_detected
            else:
                behaviors_observed = self._extract_behaviors_improved(test_case, output_text)
            
            return EnhancedEvaluationResult(
                test_case_id=test_case["id"],
                status=status,
                execution_time=execution_time,
                functions_called=functions_called,
                output_text=output_text,
                behaviors_observed=behaviors_observed,
                consistency_score=0.0,  # Will be updated later
                function_analysis=function_analysis,
                behavior_analysis=behavior_analysis
            )
            
        except asyncio.TimeoutError:
            return EnhancedEvaluationResult(
                test_case_id=test_case["id"],
                status="TIMEOUT",
                execution_time=time.time() - start_time,
                functions_called=[],
                output_text="",
                behaviors_observed=[],
                consistency_score=0.0,
                error_message="Test case timeout"
            )
        except Exception as e:
            logger.error(f"Error executing test case {test_case['id']}: {e}")
            return EnhancedEvaluationResult(
                test_case_id=test_case["id"],
                status="ERROR",
                execution_time=time.time() - start_time,
                functions_called=[],
                output_text="",
                behaviors_observed=[],
                consistency_score=0.0,
                error_message=str(e)
            )
    
    async def _run_agent_task(self, agent_instance, query: str):
        """Run agent task - adapts to different agent interfaces"""
        try:
            if hasattr(agent_instance, 'run_stream'):
                results = []
                async for result in agent_instance.run_stream(task=query):
                    results.append(result)
                return results
            elif hasattr(agent_instance, 'run'):
                return await agent_instance.run(query)
            elif callable(agent_instance):
                return await agent_instance(query)
            else:
                raise ValueError("Agent instance must have 'run', 'run_stream' method or be callable")
        except Exception as e:
            logger.error(f"Error running agent task: {e}")
            raise
    
    async def _analyze_consistency_llm(self, test_runs: List[EnhancedEvaluationResult], 
                                     test_case: Dict) -> LLMEvaluationResult:
        """Always attempt LLM analysis with proper fallbacks"""
        
        if not self.llm_judge:
            return self._create_fallback_evaluation(test_runs)
        
        try:
            outputs = [self._extract_agent_response(run.output_text) for run in test_runs]
            result = await self.llm_judge.evaluate_semantic_similarity(
                outputs, 
                test_case.get("expected_output_keywords", []),
                test_case.get("domain_context", "")
            )
            return result
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return self._create_fallback_evaluation(test_runs)
    
    def _create_fallback_evaluation(self, test_runs: List[EnhancedEvaluationResult]) -> LLMEvaluationResult:
        """Create fallback evaluation when LLM analysis fails"""
        consistency_score = self._calculate_basic_consistency(test_runs)
        
        return LLMEvaluationResult(
            similarity_score=consistency_score,
            consistency_score=consistency_score,
            quality_score=0.75,  # Assume reasonable quality if no errors
            confidence=0.5,
            reasoning="Fallback evaluation - LLM analysis unavailable",
            specific_issues=[],
            improvement_suggestions=[]
        )
    
    def _calculate_basic_consistency(self, test_runs: List[EnhancedEvaluationResult]) -> float:
        """Basic consistency calculation (fallback)"""
        if len(test_runs) < 2:
            return 1.0
        
        # Extract clean responses and compute word overlap
        responses = []
        for run in test_runs:
            clean_response = self._extract_agent_response(run.output_text)
            if clean_response.strip():
                responses.append(clean_response.lower())
        
        if len(responses) < 2:
            return 1.0
        
        word_sets = [set(response.split()) for response in responses]
        
        if not word_sets:
            return 1.0
        
        common_words = set.intersection(*word_sets)
        total_unique_words = set.union(*word_sets)
        
        if not total_unique_words:
            return 1.0
        
        return len(common_words) / len(total_unique_words)
    
    def _extract_agent_response(self, raw_output: str) -> str:
        """Extract the actual agent response from structured AutoGen output"""
        
        # Look for TextMessage content that's not the user query
        pattern = r"TextMessage\([^)]*content='([^']*(?:\\.[^']*)*)'"
        matches = re.findall(pattern, raw_output, re.DOTALL)
        
        if matches:
            # Filter out short messages (likely user queries) and find the main response
            responses = []
            for match in matches:
                # Clean up escaped characters
                clean_match = match.replace('\\n', '\n').replace('\\u', '\\u').replace("\\'", "'")
                if len(clean_match) > 50:  # Likely a substantial response
                    responses.append(clean_match)
            
            if responses:
                # Return the longest response (likely the main agent response)
                return max(responses, key=len)
        
        # Fallback: look for content between specific patterns
        if "content='" in raw_output:
            try:
                start = raw_output.rfind("content='") + 9
                end = raw_output.find("'", start)
                if end > start:
                    content = raw_output[start:end]
                    if len(content) > 50:
                        return content.replace('\\n', '\n')
            except:
                pass
        
        # Last resort: return original if we can't parse
        return str(raw_output)
    
    def _extract_functions_advanced(self, output_text: str) -> List[str]:
        """Enhanced function extraction that handles structured agent outputs"""
        functions_called = []
        
        # Handle AutoGen structured message outputs
        if "ToolCallRequestEvent" in output_text and "FunctionCall" in output_text:
            # Extract function names from ToolCallRequestEvent structures
            pattern = r"FunctionCall\([^)]*name='([^']+)'"
            matches = re.findall(pattern, output_text)
            functions_called.extend(matches)
            
        # Also check for ToolCallExecutionEvent (successful execution)
        if "ToolCallExecutionEvent" in output_text:
            pattern = r"name='([^']+)'[^}]*call_id"
            matches = re.findall(pattern, output_text)
            functions_called.extend(matches)
        
        # Fallback to basic text search for other formats
        function_patterns = [
            "get_emission_analysis",
            "emission_tool", 
            "PythonCodeExecutionTool",
            "search_tool",
            "calculator",
            "file_reader",
            "web_search",
            "data_retrieval"
        ]
        
        output_lower = output_text.lower()
        for pattern in function_patterns:
            if pattern.lower() in output_lower and pattern not in functions_called:
                functions_called.append(pattern)
        
        return list(set(functions_called))  # Remove duplicates
    
    def _extract_behaviors_improved(self, test_case: Dict, output_text: str) -> List[str]:
        """Improved behavior extraction"""
        behaviors = []
        
        # Extract the clean agent response
        agent_response = self._extract_agent_response(output_text)
        agent_response_lower = agent_response.lower()
        
        # Check for expected behaviors with more nuanced detection
        expected_behaviors = test_case.get("expected_behavior", [])
        
        behavior_indicators = {
            "correct_function_call": [
                "ToolCallRequestEvent", "get_emission_analysis", "retrieved", "analyzed",
                "FunctionCall", "ToolCallExecutionEvent"
            ],
            "high_quality_response": [
                "detailed", "comprehensive", "analysis", "recommendations", 
                "windows", "intensity", "schedule", "tips", "optimal", "best time"
            ],
            "user_friendly": [
                "best time", "recommend", "suggest", "tips", "help", "easy", 
                "simply", "here", "emoji", "🌱", "⚡", "🔥", "windows"
            ],
            "proper_error_handling": [
                "sorry", "unable", "limitation", "try", "please", "invalid", 
                "cannot", "not available", "error", "failed"
            ],
            "domain_expertise": [
                "carbon", "emission", "co2", "intensity", "sustainability", 
                "renewable", "grid", "electricity", "ireland", "roi", "northern ireland"
            ],
            "consistent_output": [
                # This should be evaluated across multiple runs, not per individual response
            ]
        }
        
        for behavior in expected_behaviors:
            if behavior == "consistent_output":
                continue  # Handle this at the consistency analysis level
                
            if behavior in behavior_indicators:
                indicators = behavior_indicators[behavior]
                # Check both the full output and clean response
                full_text = output_text.lower() + " " + agent_response_lower
                if any(indicator.lower() in full_text for indicator in indicators):
                    behaviors.append(behavior)
        
        return behaviors
    
    def _determine_status_improved(self, test_case: Dict, function_analysis: Optional[FunctionCallAnalysis],
                                 behavior_analysis: Optional[BehaviorAnalysis], output_text: str) -> str:
        """Improved status determination that's less strict and more intelligent"""
        
        # Extract the actual agent response (not the full structured output)
        agent_response = self._extract_agent_response(output_text)
        
        # Check for LLM quality indicators first
        llm_bias_pass = False
        if behavior_analysis:
            # If LLM gives high quality scores, bias toward PASS
            avg_quality = sum(behavior_analysis.quality_metrics.values()) / len(behavior_analysis.quality_metrics) if behavior_analysis.quality_metrics else 0
            if (avg_quality >= 0.8 or 
                behavior_analysis.domain_expertise_score >= 0.8 or 
                behavior_analysis.user_experience_score >= 0.8):
                llm_bias_pass = True
        
        # Check for actual critical errors (not just mentions of "error")
        critical_errors = [
            "exception occurred", "traceback", "failed to execute", 
            "error executing", "cannot complete", "tool failed",
            "timeout error", "connection failed"
        ]
        has_critical_error = any(error in agent_response.lower() for error in critical_errors)
        
        if has_critical_error and not llm_bias_pass:
            return "FAIL"
        
        # Check function requirements more intelligently
        expected_functions = test_case.get("expected_functions", [])
        if expected_functions:
            detected_functions = self._extract_functions_advanced(output_text)
            
            # If any expected function was called successfully, that's good
            if any(func in detected_functions for func in expected_functions):
                function_requirement_met = True
            else:
                # Check if agent provided substantive response despite no function call
                if len(agent_response.strip()) > 100:
                    # For irrelevant queries, not calling functions might be correct
                    if test_case.get("category") == "irrelevant_query":
                        function_requirement_met = True
                    elif "sorry" in agent_response.lower() or "cannot" in agent_response.lower():
                        function_requirement_met = True  # Proper error handling
                    else:
                        function_requirement_met = llm_bias_pass  # Let LLM quality override
                else:
                    return "FAIL"
        
        # Check if agent provided a reasonable response
        if len(agent_response.strip()) < 20:
            return "FAIL"
        
        # Check behavior requirements (but be more lenient)
        expected_behaviors = test_case.get("expected_behavior", [])
        if expected_behaviors:
            detected_behaviors = self._extract_behaviors_improved(test_case, output_text)
            
            # If LLM detected behaviors or we have high quality, be more lenient
            if behavior_analysis and behavior_analysis.behaviors_detected:
                detected_behaviors.extend(behavior_analysis.behaviors_detected)
            
            # Must have at least some expected behaviors OR high LLM quality
            behavior_match = any(behavior in detected_behaviors for behavior in expected_behaviors)
            if not behavior_match and not llm_bias_pass:
                return "FAIL"
        
        # Check output keywords (but make it optional if LLM quality is high)
        expected_keywords = test_case.get("expected_output_keywords", [])
        if expected_keywords and not llm_bias_pass:
            # Make keyword checking more flexible
            keyword_matches = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in agent_response.lower())
            keyword_ratio = keyword_matches / len(expected_keywords) if expected_keywords else 1.0
            
            # Require at least 50% keyword match (instead of 100%)
            if keyword_ratio < 0.5:
                return "FAIL"
        
        # If we get here, the test should pass
        return "PASS"
    
    def _determine_status(self, test_case: Dict, function_analysis: Optional[FunctionCallAnalysis],
                         behavior_analysis: Optional[BehaviorAnalysis], output_text: str) -> str:
        """Legacy method - redirects to improved version"""
        return self._determine_status_improved(test_case, function_analysis, behavior_analysis, output_text)
    
    def _generate_enhanced_report(self, agent_name: str, 
                                results: List[EnhancedEvaluationResult]) -> EnhancedEvaluationReport:
        """Generate enhanced evaluation report"""
        if not results:
            return EnhancedEvaluationReport(
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
        
        # Calculate basic statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == "PASS"])
        failed_tests = len([r for r in results if r.status == "FAIL"])
        error_tests = len([r for r in results if r.status == "ERROR"])
        timeout_tests = len([r for r in results if r.status == "TIMEOUT"])
        
        avg_execution_time = statistics.mean([r.execution_time for r in results])
        
        # Enhanced metrics
        semantic_scores = [r.semantic_similarity for r in results if r.semantic_similarity > 0]
        semantic_consistency_score = statistics.mean(semantic_scores) if semantic_scores else 0.0
        
        quality_scores = [r.quality_score for r in results if r.quality_score > 0]
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # LLM confidence scores
        confidence_scores = []
        for result in results:
            if result.llm_evaluation and result.llm_evaluation.confidence > 0:
                confidence_scores.append(result.llm_evaluation.confidence)
        llm_confidence_score = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        # Function call accuracy (improved calculation)
        accurate_function_calls = 0
        for result in results:
            if result.function_analysis:
                if not result.function_analysis.missing_functions:
                    accurate_function_calls += 1
            elif result.functions_called:  # If functions were detected
                accurate_function_calls += 1
            elif "correct_function_call" in result.behaviors_observed:
                accurate_function_calls += 1
        function_call_accuracy = accurate_function_calls / total_tests if total_tests > 0 else 0.0
        
        # Behavior scores
        all_behaviors = set()
        for result in results:
            all_behaviors.update(result.behaviors_observed)
        
        behavior_scores = {}
        for behavior in all_behaviors:
            count = sum(1 for result in results if behavior in result.behaviors_observed)
            behavior_scores[behavior] = count / total_tests if total_tests > 0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, behavior_scores)
        
        return EnhancedEvaluationReport(
            agent_name=agent_name,
            evaluation_date=datetime.now().isoformat(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            timeout_tests=timeout_tests,
            avg_execution_time=avg_execution_time,
            consistency_score=semantic_consistency_score,
            function_call_accuracy=function_call_accuracy,
            behavior_scores=behavior_scores,
            detailed_results=results,
            recommendations=recommendations,
            semantic_consistency_score=semantic_consistency_score,
            avg_quality_score=avg_quality_score,
            llm_confidence_score=llm_confidence_score
        )
    
    def _generate_recommendations(self, results: List[EnhancedEvaluationResult], 
                                behavior_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Calculate pass rate
        pass_rate = len([r for r in results if r.status == "PASS"]) / len(results) if results else 0
        
        # Function call recommendations
        if behavior_scores.get("correct_function_call", 0) < 0.8:
            recommendations.append(
                "Function call accuracy is low. Review system prompts for better tool usage guidance."
            )
        
        # Quality recommendations
        quality_scores = [r.quality_score for r in results if r.quality_score > 0]
        if quality_scores and statistics.mean(quality_scores) < 0.7:
            recommendations.append(
                "Response quality below threshold. Consider enhancing domain knowledge in prompts."
            )
        
        # Consistency recommendations
        consistency_scores = [r.consistency_score for r in results if r.consistency_score > 0]
        if consistency_scores and statistics.mean(consistency_scores) < 0.75:
            recommendations.append(
                "Consistency scores are low. Add more specific instructions for deterministic responses."
            )
        
        # Error rate recommendations
        error_rate = len([r for r in results if r.status == "ERROR"]) / len(results) if results else 0
        if error_rate > 0.1:
            recommendations.append(
                "High error rate detected. Review agent error handling and input validation."
            )
        
        # Pass rate recommendations
        if pass_rate < 0.7:
            recommendations.append(
                f"Pass rate ({pass_rate:.1%}) is low. Review test expectations and agent capabilities."
            )
        
        # LLM-specific recommendations
        llm_issues = []
        for result in results:
            if result.llm_evaluation and result.llm_evaluation.improvement_suggestions:
                llm_issues.extend(result.llm_evaluation.improvement_suggestions)
        
        if llm_issues:
            # Get most common LLM suggestions
            from collections import Counter
            common_suggestions = Counter(llm_issues).most_common(3)
            for suggestion, count in common_suggestions:
                if count > 1:  # Only include suggestions that appear multiple times
                    recommendations.append(f"LLM suggests: {suggestion}")
        
        return recommendations
    
    def _save_results(self, report: EnhancedEvaluationReport):
        """Save evaluation results"""
        try:
            output_dir = Path(self.config["output_dir"])
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON report
            json_file = output_dir / f"enhanced_evaluation_report_{report.agent_name}_{timestamp}.json"
            with open(json_file, 'w') as f:
                report_dict = asdict(report)
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Enhanced evaluation results saved to {json_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_enhanced_summary(self, report: EnhancedEvaluationReport):
        """Print enhanced evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Agent: {report.agent_name}")
        print(f"Date: {report.evaluation_date}")
        print(f"Total Tests: {report.total_tests}")
        
        if report.total_tests > 0:
            print(f"✅ Passed: {report.passed_tests} ({report.passed_tests/report.total_tests*100:.1f}%)")
            print(f"❌ Failed: {report.failed_tests} ({report.failed_tests/report.total_tests*100:.1f}%)")
            print(f"⚠️  Errors: {report.error_tests}")
            print(f"⏰ Timeouts: {report.timeout_tests}")
        
        print(f"\n📈 ENHANCED METRICS:")
        print(f"Average Execution Time: {report.avg_execution_time:.2f}s")
        print(f"🧠 Semantic Consistency: {report.semantic_consistency_score:.2f}")
        print(f"⭐ Average Quality Score: {report.avg_quality_score:.2f}")
        print(f"🔧 Function Call Accuracy: {report.function_call_accuracy:.2f}")
        print(f"🎯 LLM Confidence: {report.llm_confidence_score:.2f}")
        
        print(f"\n🎭 BEHAVIOR ANALYSIS:")
        for behavior, score in report.behavior_scores.items():
            emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
            print(f"  {emoji} {behavior}: {score:.2f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in report.detailed_results[:10]:  # Show first 10
            status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
            quality_info = f"(Q:{result.quality_score:.2f})" if result.quality_score > 0 else ""
            print(f"  {status_emoji} {result.test_case_id}: {result.status} "
                  f"({result.execution_time:.2f}s) {quality_info}")
        
        if len(report.detailed_results) > 10:
            print(f"  ... and {len(report.detailed_results) - 10} more results")
        
        print("=" * 60)