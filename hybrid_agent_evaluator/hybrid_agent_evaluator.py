"""
Hybrid Agent Evaluator - Combines both the rule-based and LLM-as-judge approaches
for comprehensive agent evaluation
"""

import asyncio
import json
import time
import statistics
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

from evaluation_strategies import RuleBasedStrategy, LLMJudgeStrategy, EvaluationMode
from result_combiner import ResultCombiner
from llm_judge import LLMJudge
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

logger = logging.getLogger(__name__)

class EvaluationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

@dataclass
class HybridEvaluationResult:
    """Enhanced evaluation result with structured output"""
    test_case_id: str
    status: EvaluationStatus
    execution_time: float
    
    # Rule-based results
    rule_based_status: EvaluationStatus
    rule_based_score: float
    functions_called: List[str]
    behaviors_observed: List[str]
    keyword_matches: int
    consistency_score: float
    
    # LLM-based results
    llm_status: Optional[EvaluationStatus] = None
    llm_quality_score: float = 0.0
    llm_confidence: float = 0.0
    semantic_similarity: float = 0.0
    llm_reasoning: str = ""
    llm_feedback: List[str] = None
    
    # Combined results
    final_score: float = 0.0
    confidence_level: str = "low"
    detailed_feedback: List[str] = None
    improvement_suggestions: List[str] = None
    
    # Structured output data
    raw_output: str = ""
    structured_output: Dict[str, Any] = None
    conversation_flow: List[Dict[str, Any]] = None
    tool_interactions: List[Dict[str, Any]] = None
    agent_responses: List[str] = None
    
    # Metadata
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.llm_feedback is None:
            self.llm_feedback = []
        if self.detailed_feedback is None:
            self.detailed_feedback = []
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []
        if self.structured_output is None:
            self.structured_output = {}
        if self.conversation_flow is None:
            self.conversation_flow = []
        if self.tool_interactions is None:
            self.tool_interactions = []
        if self.agent_responses is None:
            self.agent_responses = []

@dataclass
class HybridEvaluationReport:
    """Comprehensive evaluation report"""
    agent_name: str
    evaluation_date: str
    evaluation_mode: str
    total_tests: int
    
    # Pass/fail statistics
    passed_tests: int
    failed_tests: int
    error_tests: int
    timeout_tests: int
    
    # Performance metrics
    avg_execution_time: float
    rule_based_accuracy: float
    llm_agreement_rate: float
    
    # Quality metrics
    avg_rule_score: float
    avg_llm_score: float
    avg_combined_score: float
    consistency_score: float
    
    # Detailed results
    detailed_results: List[HybridEvaluationResult]
    
    # Insights
    recommendations: List[str]
    llm_insights: List[str]
    performance_bottlenecks: List[str]
    
    def pass_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

class AgentOutputParser:
    """Parser for structuring agent output into readable format"""
    
    @staticmethod
    def parse_agent_output(raw_output: str) -> Dict[str, Any]:
        """Parse raw agent output into structured format with robust error handling"""
        if not raw_output or not isinstance(raw_output, str):
            return AgentOutputParser._create_fallback_structure(raw_output or "")
        
        try:
            return AgentOutputParser._parse_autogen_output(raw_output)
        except Exception as e:
            logger.warning(f"Failed to parse agent output: {e}")
            return AgentOutputParser._create_fallback_structure(raw_output)
    
    @staticmethod
    def _parse_autogen_output(raw_output: str) -> Dict[str, Any]:
        """Parse AutoGen-specific output format with improved error handling and better content extraction"""
        import re
        
        # Initialize structure with safe defaults
        structured = {
            "conversation_flow": [],
            "tool_interactions": [],
            "agent_responses": [],
            "user_messages": [],
            "errors": [],
            "summary": {
                "total_user_messages": 0,
                "total_tool_calls": 0,
                "total_agent_responses": 0,
                "total_errors": 0,
                "tools_used": [],
                "has_errors": False,
                "response_length": 0,
                "conversation_turns": 0
            }
        }
        
        try:
            # Extract user messages - improved pattern to handle various formats
            user_patterns = [
                r"TextMessage\([^)]*source='user'[^)]*content='([^']*(?:\\.[^']*)*)'"  # Single quotes
            ]
            
            for pattern in user_patterns:
                user_matches = re.findall(pattern, raw_output, re.DOTALL)
                for i, match in enumerate(user_matches, 1):
                    try:
                        clean_content = AgentOutputParser._clean_content(match)
                        structured["user_messages"].append({
                            "message_id": f"user_{i}",
                            "content": clean_content,
                            "type": "user_input"
                        })
                    except Exception as e:
                        logger.debug(f"Error processing user message {i}: {e}")
            
            # Extract tool call requests - improved pattern
            tool_request_patterns = [
                r"ToolCallRequestEvent\([^)]*content=\[FunctionCall\([^)]*name='([^']+)'[^)]*arguments='([^']*)",
                r"FunctionCall\([^)]*name='([^']+)'[^)]*arguments='([^']*)"
            ]
            
            tool_requests = []
            for pattern in tool_request_patterns:
                matches = re.findall(pattern, raw_output)
                tool_requests.extend(matches)
            
            # Extract tool execution results - improved pattern
            tool_result_patterns = [
                r"ToolCallExecutionEvent\([^)]*content=\[FunctionExecutionResult\([^)]*content=\"([^\"]*(?:\\.[^\"]*)*)\"[^)]*name='([^']+)'[^)]*is_error=([^,)]+)",
                r"ToolCallExecutionEvent\([^)]*content=\[FunctionExecutionResult\([^)]*content='([^']*(?:\\.[^']*)*)'[^)]*name='([^']+)'[^)]*is_error=([^,)]+)",
                r"FunctionExecutionResult\([^)]*content=\"([^\"]*(?:\\.[^\"]*)*)\"[^)]*name='([^']+)'[^)]*is_error=([^,)]+)",
                r"FunctionExecutionResult\([^)]*content='([^']*(?:\\.[^']*)*)'[^)]*name='([^']+)'[^)]*is_error=([^,)]+)"
            ]
            
            tool_results = []
            for pattern in tool_result_patterns:
                matches = re.findall(pattern, raw_output)
                tool_results.extend(matches)
            
            # Process tool interactions
            for i, (func_name, arguments) in enumerate(tool_requests, 1):
                try:
                    tool_interaction = {
                        "interaction_id": f"tool_{i}",
                        "function_name": func_name,
                        "arguments": AgentOutputParser._clean_content(arguments),
                        "type": "tool_call"
                    }
                    
                    # Find corresponding result
                    for result_content, result_func, is_error in tool_results:
                        if result_func == func_name:
                            tool_interaction["result"] = {
                                "content": AgentOutputParser._clean_content(result_content),
                                "is_error": is_error.strip() == 'True',
                                "status": "error" if is_error.strip() == 'True' else "success"
                            }
                            if is_error.strip() == 'True':
                                structured["errors"].append({
                                    "function": func_name,
                                    "error": result_content,
                                    "type": "tool_execution_error"
                                })
                            break
                    
                    structured["tool_interactions"].append(tool_interaction)
                except Exception as e:
                    logger.debug(f"Error processing tool interaction {i}: {e}")
            
            # Extract agent responses - improved patterns to handle various source formats
            agent_patterns = [
                r"TextMessage\([^)]*source='[^']*assistant[^']*'[^)]*content='([^']*(?:\\.[^']*)*)'"  # Single quotes
            ]
            
            agent_responses_raw = []
            for pattern in agent_patterns:
                matches = re.findall(pattern, raw_output, re.DOTALL)
                agent_responses_raw.extend(matches)
            
            # Process agent responses with better cleaning and formatting
            for i, match in enumerate(agent_responses_raw, 1):
                try:
                    clean_content = AgentOutputParser._clean_content(match)
                    
                    # Only add substantial responses (filter out very short ones)
                    if len(clean_content.strip()) > 10:
                        structured["agent_responses"].append({
                            "response_id": f"Response{i:02d}",
                            "content": clean_content,
                            "type": "agent_response",
                            "length": len(clean_content),
                            "has_recommendations": "recommend" in clean_content.lower() or "suggest" in clean_content.lower(),
                            "has_time_info": any(time_word in clean_content.lower() for time_word in ["time", "hour", "morning", "evening", "overnight"]),
                            "has_data": "data" in clean_content.lower() or "intensity" in clean_content.lower(),
                            "formatted_content": AgentOutputParser._format_for_display(clean_content)
                        })
                except Exception as e:
                    logger.debug(f"Error processing agent response {i}: {e}")
            
            # Build conversation flow with better structure
            flow_counter = 1
            for user_msg in structured["user_messages"]:
                try:
                    structured["conversation_flow"].append({
                        "step": flow_counter,
                        "type": "user_input",
                        "content": user_msg["content"],
                        "preview": user_msg["content"][:100] + "..." if len(user_msg["content"]) > 100 else user_msg["content"]
                    })
                    flow_counter += 1
                except Exception as e:
                    logger.debug(f"Error adding user message to flow: {e}")
            
            for tool in structured["tool_interactions"]:
                try:
                    structured["conversation_flow"].append({
                        "step": flow_counter,
                        "type": "tool_call",
                        "function": tool["function_name"],
                        "arguments": tool.get("arguments", ""),
                        "status": tool.get("result", {}).get("status", "unknown"),
                        "result_preview": tool.get("result", {}).get("content", "")[:100] + "..." if len(tool.get("result", {}).get("content", "")) > 100 else tool.get("result", {}).get("content", "")
                    })
                    flow_counter += 1
                except Exception as e:
                    logger.debug(f"Error adding tool to flow: {e}")
            
            for agent_resp in structured["agent_responses"]:
                try:
                    structured["conversation_flow"].append({
                        "step": flow_counter,
                        "type": "agent_response",
                        "response_id": agent_resp["response_id"],
                        "length": agent_resp["length"],
                        "preview": agent_resp["content"][:200] + "..." if len(agent_resp["content"]) > 200 else agent_resp["content"]
                    })
                    flow_counter += 1
                except Exception as e:
                    logger.debug(f"Error adding agent response to flow: {e}")
            
            # Generate summary
            tools_used = []
            for tool in structured["tool_interactions"]:
                try:
                    func_name = tool.get("function_name")
                    if func_name:
                        tools_used.append(func_name)
                except Exception:
                    pass
            
            response_length = sum(resp.get("length", 0) for resp in structured["agent_responses"])
            
            structured["summary"] = {
                "total_user_messages": len(structured["user_messages"]),
                "total_tool_calls": len(structured["tool_interactions"]),
                "total_agent_responses": len(structured["agent_responses"]),
                "total_errors": len(structured["errors"]),
                "tools_used": list(set(tools_used)),
                "has_errors": len(structured["errors"]) > 0,
                "response_length": response_length,
                "conversation_turns": len(structured["conversation_flow"])
            }
            
        except Exception as e:
            logger.warning(f"Error in _parse_autogen_output: {e}")
            # Return partial structure with error info
            structured["errors"].append({
                "type": "parsing_error",
                "message": f"Parsing failed: {str(e)}"
            })
            structured["summary"]["has_errors"] = True
            structured["summary"]["total_errors"] = len(structured["errors"])
        
        return structured
    
    @staticmethod
    def _clean_content(content: str) -> str:
        """Clean and normalize content from agent output"""
        if not content:
            return ""
        
        # Replace escape sequences
        clean_content = content.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
        
        # Clean up unicode escapes
        clean_content = AgentOutputParser._clean_unicode(clean_content)
        
        return clean_content.strip()
    
    @staticmethod
    def _format_for_display(content: str) -> str:
        """Format content for better display in JSON with proper line breaks"""
        if not content:
            return ""
        
        # Split into lines and clean up
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Only add non-empty lines
                formatted_lines.append(line)
        
        # Join with proper line breaks and add some structure
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def _clean_unicode(text: str) -> str:
        """Clean unicode escape sequences"""
        replacements = {
            '\\u2019': "'",
            '\\u2013': '–', 
            '\\u2014': '—',
            '\\u2082': '₂',
            '\\u00b0': '°',
            '\\u201c': '"',
            '\\u201d': '"',
            '\\ud83c\\udf31': '🌱',
            '\\ud83d\\udd25': '🔥',
            '\\ud83d\\udd51': '🕑',
            '\\ud83d\\udd0c': '🔌',
            '\\ud83c\\udf0d': '🌍',
            '\\ud83d\\udc9a': '💚',
            '\\ud83d\\udd0b': '🔋',
            '\\u23f1\\ufe0f': '⏱️',
            '\\ud83d\\udcf1': '📱',
            '\\u26a1': '⚡',
            '\\u2010': '-'
        }
        
        for escape, char in replacements.items():
            text = text.replace(escape, char)
        
        return text
    
    @staticmethod
    def _create_fallback_structure(raw_output: str) -> Dict[str, Any]:
        """Create safe fallback structure when parsing fails"""
        return {
            "conversation_flow": [{"step": 1, "type": "raw_output", "content": "Failed to parse - see raw_output"}],
            "tool_interactions": [],
            "agent_responses": [{"response_id": "fallback", "content": raw_output[:500] + "..." if len(raw_output) > 500 else raw_output, "length": len(raw_output)}],
            "user_messages": [],
            "errors": [{"type": "parsing_error", "message": "Could not parse agent output"}],
            "summary": {
                "total_user_messages": 0,
                "total_tool_calls": 0,
                "total_agent_responses": 1,
                "total_errors": 1,
                "tools_used": [],
                "has_errors": True,
                "response_length": len(raw_output),
                "conversation_turns": 1
            }
        }

class HybridAgentEvaluator:
    """
    Hybrid Agent Evaluator that combines rule-based and LLM-as-judge approaches
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.test_cases = []
        
        # Initialize strategies
        self.rule_strategy = RuleBasedStrategy(self.config)
        self.llm_strategy = None
        
        # Initialize LLM if enabled
        if self.config.get("llm_evaluation_enabled", True):
            self._setup_llm_strategy()
        
        # Result combiner
        self.result_combiner = ResultCombiner(self.config)
        
        logger.info(f"Hybrid evaluator initialized with mode: {self.config.get('evaluation_mode', 'hybrid')}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration with intelligent defaults"""
        default_config = {
            # Evaluation mode
            "evaluation_mode": "hybrid",  # "rule_based", "llm_only", "hybrid"
            
            # Basic settings
            "max_retries": 3,
            "timeout_seconds": 60,
            "output_dir": "./evaluation_results",
            
            # Rule-based settings
            "consistency_threshold": 0.8,
            "keyword_weight": 0.3,
            "function_weight": 0.4,
            "behavior_weight": 0.3,
            
            # LLM settings
            "llm_evaluation_enabled": True,
            "llm_on_failures_only": False,  # If True, only run LLM on failed rule-based tests
            "llm_quality_threshold": 0.7,
            "llm_confidence_threshold": 0.6,
            
            # Hybrid settings
            "rule_weight": 0.4,  # Weight for rule-based score in final score
            "llm_weight": 0.6,   # Weight for LLM score in final score
            "llm_override_enabled": True,  # Allow LLM to override rule-based decisions
            "llm_override_threshold": 0.8,  # LLM confidence needed to override
            
            # Performance settings
            "parallel_llm_calls": False,
            "max_llm_tokens": 2000,
            "llm_timeout": 30
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_llm_strategy(self):
        """Setup LLM strategy with Azure OpenAI"""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            required_vars = ["AZURE_DEPLOYMENT", "AZURE_ENDPOINT", "API_KEY", "API_VERSION"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing environment variables for LLM: {missing_vars}")
                self.llm_strategy = None
                return
            
            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                model=os.getenv("MODEL", "gpt-4"),
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("API_KEY")
            )
            
            llm_judge = LLMJudge(
                azure_client=client,
                model_name=os.getenv("MODEL", "gpt-4"),
                max_retries=self.config.get("max_retries", 3)
            )
            
            self.llm_strategy = LLMJudgeStrategy(llm_judge, self.config)
            logger.info("LLM strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM strategy: {e}")
            self.llm_strategy = None
    
    def load_test_cases(self, test_cases_path: str):
        """Load test cases from JSON file"""
        try:
            with open(test_cases_path, 'r') as f:
                test_data = json.load(f)
            
            if "test_cases" in test_data:
                self.test_cases = test_data["test_cases"]
            else:
                self.test_cases = test_data
            
            logger.info(f"Loaded {len(self.test_cases)} test cases")
            
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            raise
    
    async def evaluate_agent(self, agent_instance, runs_per_test: int = 3) -> HybridEvaluationReport:
        """
        Main evaluation method that combines rule-based and LLM approaches
        """
        logger.info(f"Starting hybrid evaluation with {len(self.test_cases)} test cases")
        logger.info(f"Evaluation mode: {self.config['evaluation_mode']}")
        logger.info(f"Runs per test: {runs_per_test}")
        
        if not self.test_cases:
            raise ValueError("No test cases loaded. Call load_test_cases() first.")
        
        all_results = []
        evaluation_mode = EvaluationMode(self.config["evaluation_mode"])
        
        for test_case in self.test_cases:
            logger.info(f"Evaluating test case: {test_case['id']}")
            
            # Run multiple iterations for consistency analysis
            test_runs = []
            for run in range(runs_per_test):
                result = await self._evaluate_single_test(agent_instance, test_case, run, evaluation_mode)
                test_runs.append(result)
                await asyncio.sleep(0.5)  # Brief delay between runs
            
            # Calculate consistency across runs
            self._update_consistency_scores(test_runs)
            all_results.extend(test_runs)
        
        # Generate comprehensive report
        report = self._generate_report(agent_instance, all_results, evaluation_mode)
        
        # Add debug information
        logger.info(f"Evaluation completed: {len(all_results)} total results")
        logger.info(f"Status breakdown: PASS={report.passed_tests}, FAIL={report.failed_tests}, ERROR={report.error_tests}, TIMEOUT={report.timeout_tests}")
        logger.info(f"Rule-LLM agreement: {report.llm_agreement_rate:.2f}")
        
        # Save results
        self._save_results(report)
        
        return report
    
    async def _evaluate_single_test(self, agent_instance, test_case: Dict, 
                                  run_number: int, mode: EvaluationMode) -> HybridEvaluationResult:
        """Evaluate a single test case with the specified mode"""
        
        start_time = time.time()
        
        try:
            # Execute the agent
            result = await asyncio.wait_for(
                self._run_agent_task(agent_instance, test_case["query"]),
                timeout=test_case.get("timeout_seconds", self.config["timeout_seconds"])
            )
            
            execution_time = time.time() - start_time
            output_text = str(result)
            
            # Parse the output structure
            structured_output = AgentOutputParser.parse_agent_output(output_text)
            
            # Extract agent responses for LLM evaluation
            agent_responses = []
            for resp_data in structured_output.get("agent_responses", []):
                if isinstance(resp_data, dict):
                    content = resp_data.get("content", "")
                    if content:
                        agent_responses.append(content)
                else:
                    agent_responses.append(str(resp_data))
            
            # Get main response for LLM evaluation
            main_response = ""
            if agent_responses:
                # Use the longest response as the main response
                main_response = max(agent_responses, key=len)
            else:
                # Fallback: extract from raw output
                main_response = self._extract_agent_response(output_text)
            
            # Run rule-based evaluation
            rule_result = await self.rule_strategy.evaluate(test_case, output_text)
            
            # Initialize hybrid result with rule-based data and structured output
            hybrid_result = HybridEvaluationResult(
                test_case_id=test_case["id"],
                status=rule_result.status,
                execution_time=execution_time,
                rule_based_status=rule_result.status,
                rule_based_score=rule_result.score,
                functions_called=rule_result.functions_called,
                behaviors_observed=rule_result.behaviors_observed,
                keyword_matches=rule_result.keyword_matches,
                consistency_score=0.0,  # Updated later
                raw_output=output_text,
                structured_output=structured_output,
                conversation_flow=structured_output.get("conversation_flow", []),
                tool_interactions=structured_output.get("tool_interactions", []),
                agent_responses=agent_responses
            )
            
            # Run LLM evaluation based on mode
            if self._should_run_llm_evaluation(mode, rule_result):
                try:
                    llm_result = await self.llm_strategy.evaluate(test_case, main_response, rule_result)
                    
                    # Update with LLM results
                    hybrid_result.llm_status = llm_result.status
                    hybrid_result.llm_quality_score = llm_result.quality_score
                    hybrid_result.llm_confidence = llm_result.confidence
                    hybrid_result.semantic_similarity = llm_result.semantic_similarity
                    hybrid_result.llm_reasoning = llm_result.reasoning
                    hybrid_result.llm_feedback = llm_result.feedback
                    
                except Exception as e:
                    logger.warning(f"LLM evaluation failed for {test_case['id']}: {e}")
            
            # Combine results
            hybrid_result = self.result_combiner.combine_results(hybrid_result, test_case, mode)
            
            return hybrid_result
            
        except asyncio.TimeoutError:
            return HybridEvaluationResult(
                test_case_id=test_case["id"],
                status=EvaluationStatus.TIMEOUT,
                execution_time=time.time() - start_time,
                rule_based_status=EvaluationStatus.TIMEOUT,
                rule_based_score=0.0,
                functions_called=[],
                behaviors_observed=[],
                keyword_matches=0,
                consistency_score=0.0,
                error_message="Test case timeout",
                raw_output="",
                structured_output={"summary": {"has_errors": True, "total_errors": 1}, "errors": [{"type": "timeout", "message": "Test case timeout"}]},
                conversation_flow=[],
                tool_interactions=[],
                agent_responses=[]
            )
        except Exception as e:
            logger.error(f"Error in test {test_case['id']}: {e}")
            return HybridEvaluationResult(
                test_case_id=test_case["id"],
                status=EvaluationStatus.ERROR,
                execution_time=time.time() - start_time,
                rule_based_status=EvaluationStatus.ERROR,
                rule_based_score=0.0,
                functions_called=[],
                behaviors_observed=[],
                keyword_matches=0,
                consistency_score=0.0,
                error_message=str(e),
                raw_output="",
                structured_output={"summary": {"has_errors": True, "total_errors": 1}, "errors": [{"type": "execution_error", "message": str(e)}]},
                conversation_flow=[],
                tool_interactions=[],
                agent_responses=[]
            )
    
    def _extract_agent_response(self, raw_output: str) -> str:
        """Extract the actual agent response from structured output"""
        # Handle AutoGen TextMessage patterns
        pattern = r"TextMessage\([^)]*content='([^']*(?:\\.[^']*)*)'"
        matches = re.findall(pattern, raw_output, re.DOTALL)
        
        if matches:
            responses = []
            for match in matches:
                clean_match = match.replace('\\n', '\n').replace("\\'", "'")
                if len(clean_match) > 50:  # Substantial response
                    responses.append(clean_match)
            
            if responses:
                return max(responses, key=len)  # Return longest response
        
        # Fallback extraction
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
        
        return str(raw_output)
    
    def _should_run_llm_evaluation(self, mode: EvaluationMode, rule_result) -> bool:
        """Determine if LLM evaluation should be run"""
        if mode == EvaluationMode.RULE_BASED_ONLY:
            return False
        elif mode == EvaluationMode.LLM_ONLY or mode == EvaluationMode.HYBRID:
            if self.config.get("llm_on_failures_only", False):
                return rule_result.status == EvaluationStatus.FAIL
            return True
        return False
    
    async def _run_agent_task(self, agent_instance, query: str):
        """Run agent task with multiple interface support"""
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
    
    def _update_consistency_scores(self, test_runs: List[HybridEvaluationResult]):
        """Update consistency scores across multiple runs"""
        if len(test_runs) < 2:
            for result in test_runs:
                result.consistency_score = 1.0
            return
        
        # Calculate consistency based on multiple factors
        function_consistency = self._calculate_function_consistency(test_runs)
        output_consistency = self._calculate_output_consistency(test_runs)
        status_consistency = self._calculate_status_consistency(test_runs)
        
        # Weighted average
        for result in test_runs:
            result.consistency_score = (
                function_consistency * 0.3 +
                output_consistency * 0.4 +
                status_consistency * 0.3
            )
    
    def _calculate_function_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate consistency of function calls across runs"""
        if not test_runs:
            return 1.0
        
        function_sets = [set(result.functions_called) for result in test_runs]
        if not function_sets:
            return 1.0
        
        # Count how many runs have the same function set as the first
        first_set = function_sets[0]
        consistent_count = sum(1 for func_set in function_sets if func_set == first_set)
        
        return consistent_count / len(function_sets)
    
    def _calculate_output_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate consistency of outputs across runs using structured data"""
        if len(test_runs) < 2:
            return 1.0
        
        # Use semantic similarity if available from LLM
        semantic_scores = [result.semantic_similarity for result in test_runs if result.semantic_similarity > 0]
        if semantic_scores:
            return statistics.mean(semantic_scores)
        
        # Use structured agent responses for comparison
        agent_responses = []
        for result in test_runs:
            if result.agent_responses and len(result.agent_responses) > 0:
                # Use the main agent response
                agent_responses.append(result.agent_responses[0].lower())
            elif result.raw_output:
                # Fallback to extracting from raw output
                agent_response = self._extract_agent_response(result.raw_output)
                agent_responses.append(agent_response.lower())
        
        if len(agent_responses) < 2:
            return 1.0
        
        # Calculate word overlap consistency
        word_sets = [set(response.split()) for response in agent_responses]
        common_words = set.intersection(*word_sets)
        total_words = set.union(*word_sets)
        
        return len(common_words) / len(total_words) if total_words else 1.0
    
    def _calculate_status_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate consistency of test statuses across runs"""
        if not test_runs:
            return 1.0
        
        first_status = test_runs[0].status
        consistent_count = sum(1 for result in test_runs if result.status == first_status)
        
        return consistent_count / len(test_runs)
    
    def _generate_report(self, agent_instance, results: List[HybridEvaluationResult], 
                        mode: EvaluationMode) -> HybridEvaluationReport:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return HybridEvaluationReport(
                agent_name=agent_instance.__class__.__name__,
                evaluation_date=datetime.now().isoformat(),
                evaluation_mode=mode.value,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=0,
                timeout_tests=0,
                avg_execution_time=0.0,
                rule_based_accuracy=0.0,
                llm_agreement_rate=0.0,
                avg_rule_score=0.0,
                avg_llm_score=0.0,
                avg_combined_score=0.0,
                consistency_score=0.0,
                detailed_results=[],
                recommendations=[],
                llm_insights=[],
                performance_bottlenecks=[]
            )
        
        # Calculate basic statistics
        total_tests = len(results)
        
        # Handle both enum and string status values
        def count_status(target_status):
            count = 0
            for r in results:
                if hasattr(r.status, 'value'):
                    # Handle enum
                    if r.status.value == target_status or r.status == target_status:
                        count += 1
                else:
                    # Handle string
                    if r.status == target_status:
                        count += 1
            return count
        
        passed_tests = count_status("PASS")
        failed_tests = count_status("FAIL") 
        error_tests = count_status("ERROR")
        timeout_tests = count_status("TIMEOUT")
        
        logger.info(f"Status counts: PASS={passed_tests}, FAIL={failed_tests}, ERROR={error_tests}, TIMEOUT={timeout_tests}")
        
        # Performance metrics
        avg_execution_time = statistics.mean([r.execution_time for r in results])
        
        # Rule-based accuracy using consistent status counting
        rule_passed = count_status("PASS")  # Use the helper function that we defined above earlier
        rule_based_accuracy = rule_passed / total_tests if total_tests > 0 else 0.0
        
        # LLM agreement rate (when both rule and LLM ran)
        llm_results = [r for r in results if r.llm_status is not None]
        if llm_results:
            def status_matches(status1, status2):
                """Helper to compare status values that might be enum or string"""
                val1 = status1.value if hasattr(status1, 'value') else str(status1)
                val2 = status2.value if hasattr(status2, 'value') else str(status2)
                return val1 == val2
            
            agreement_count = len([r for r in llm_results if status_matches(r.rule_based_status, r.llm_status)])
            llm_agreement_rate = agreement_count / len(llm_results)
        else:
            llm_agreement_rate = 0.0
        
        # Quality scores
        avg_rule_score = statistics.mean([r.rule_based_score for r in results])
        llm_scores = [r.llm_quality_score for r in results if r.llm_quality_score > 0]
        avg_llm_score = statistics.mean(llm_scores) if llm_scores else 0.0
        combined_scores = [r.final_score for r in results if r.final_score > 0]
        avg_combined_score = statistics.mean(combined_scores) if combined_scores else 0.0
        
        # Consistency
        consistency_scores = [r.consistency_score for r in results if r.consistency_score > 0]
        consistency_score = statistics.mean(consistency_scores) if consistency_scores else 0.0
        
        # Generate insights
        recommendations = self._generate_recommendations(results)
        llm_insights = self._extract_llm_insights(results)
        performance_bottlenecks = self._identify_bottlenecks(results)
        
        return HybridEvaluationReport(
            agent_name=agent_instance.__class__.__name__,
            evaluation_date=datetime.now().isoformat(),
            evaluation_mode=mode.value,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            timeout_tests=timeout_tests,
            avg_execution_time=avg_execution_time,
            rule_based_accuracy=rule_based_accuracy,
            llm_agreement_rate=llm_agreement_rate,
            avg_rule_score=avg_rule_score,
            avg_llm_score=avg_llm_score,
            avg_combined_score=avg_combined_score,
            consistency_score=consistency_score,
            detailed_results=results,
            recommendations=recommendations,
            llm_insights=llm_insights,
            performance_bottlenecks=performance_bottlenecks
        )
    
    def _generate_recommendations(self, results: List[HybridEvaluationResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Pass rate analysis
        pass_rate = len([r for r in results if r.status == EvaluationStatus.PASS]) / len(results)
        if pass_rate < 0.7:
            recommendations.append(f"Low pass rate ({pass_rate:.1%}). Review agent capabilities and test expectations.")
        
        # Function call analysis
        function_issues = len([r for r in results if not r.functions_called and r.status == EvaluationStatus.FAIL])
        if function_issues > 0:
            recommendations.append(f"{function_issues} tests failed due to missing function calls. Review tool usage prompts.")
        
        # Consistency analysis
        consistency_scores = [r.consistency_score for r in results if r.consistency_score > 0]
        if consistency_scores and statistics.mean(consistency_scores) < 0.7:
            recommendations.append("Low consistency across runs. Add more specific instructions for deterministic behavior.")
        
        # LLM vs Rule disagreement
        llm_results = [r for r in results if r.llm_status is not None]
        if llm_results:
            disagreements = [r for r in llm_results if r.rule_based_status != r.llm_status]
            if len(disagreements) > len(llm_results) * 0.3:
                recommendations.append("High disagreement between rule-based and LLM evaluation. Review evaluation criteria.")
        
        # Performance issues
        slow_tests = [r for r in results if r.execution_time > 30]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests are slow (>30s). Consider optimization or increasing timeouts.")
        
        return recommendations
    
    def _extract_llm_insights(self, results: List[HybridEvaluationResult]) -> List[str]:
        """Extract insights from LLM feedback"""
        insights = []
        
        # Collect all LLM feedback
        all_feedback = []
        for result in results:
            if result.llm_feedback:
                all_feedback.extend(result.llm_feedback)
        
        # Find common themes (simple frequency analysis)
        if all_feedback:
            from collections import Counter
            feedback_counter = Counter(all_feedback)
            common_feedback = feedback_counter.most_common(3)
            
            for feedback, count in common_feedback:
                if count > 1:
                    insights.append(f"Common LLM feedback: {feedback} (mentioned {count} times)")
        
        # High confidence LLM overrides
        llm_overrides = [r for r in results if r.llm_status != r.rule_based_status and r.llm_confidence > 0.8]
        if llm_overrides:
            insights.append(f"LLM overrode rule-based decisions in {len(llm_overrides)} high-confidence cases")
        
        return insights
    
    def _identify_bottlenecks(self, results: List[HybridEvaluationResult]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Execution time analysis
        times = [r.execution_time for r in results]
        if times:
            avg_time = statistics.mean(times)
            if avg_time > 20:
                bottlenecks.append(f"High average execution time: {avg_time:.1f}s")
            
            # Find outliers
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            outliers = [r for r in results if abs(r.execution_time - avg_time) > 2 * std_dev]
            if outliers:
                bottlenecks.append(f"{len(outliers)} tests with inconsistent timing")
        
        # Timeout analysis
        timeouts = [r for r in results if r.status == EvaluationStatus.TIMEOUT]
        if timeouts:
            bottlenecks.append(f"{len(timeouts)} tests timed out")
        
        # Error analysis
        errors = [r for r in results if r.status == EvaluationStatus.ERROR]
        if errors:
            bottlenecks.append(f"{len(errors)} tests encountered errors")
        
        return bottlenecks
    
    def _format_raw_output_for_display(self, raw_output: str) -> Dict[str, Any]:
        """Format raw output for better readability in JSON - FIXED VERSION"""
        if not raw_output:
            return {"formatted": "No output", "components": []}
        
        formatted_data = {
            "formatted": "",
            "components": []
        }
        
        try:
            components = []
            
            # 1. Extract user messages - FIXED pattern
            user_pattern = r"TextMessage\([^)]*source='user'[^)]*content='([^']*(?:\\.[^']*)*)'"
            user_matches = re.findall(user_pattern, raw_output, re.DOTALL)
            for i, match in enumerate(user_matches, 1):
                clean_content = AgentOutputParser._clean_content(match)
                components.append({
                    "type": "user_message", 
                    "number": i,
                    "content": clean_content
                })
            
            # 2. Extract tool calls - FIXED to handle escaped JSON properly
            tool_pattern = r"ToolCallRequestEvent\([^)]*content=\[FunctionCall\([^)]*name='([^']+)'[^)]*arguments='([^']*(?:\\.[^']*)*)'"
            tool_matches = re.findall(tool_pattern, raw_output, re.DOTALL)
            for i, (func_name, arguments) in enumerate(tool_matches, 1):
                # Clean up escaped JSON arguments
                clean_args = arguments.replace('\\"', '"')
                components.append({
                    "type": "tool_call",
                    "number": i,
                    "function": func_name,
                    "arguments": clean_args
                })
            
            # 3. Extract tool results - FIXED to handle both quote styles
            result_patterns = [
                r"ToolCallExecutionEvent\([^)]*content=\[FunctionExecutionResult\([^)]*content=\"([^\"]*(?:\\.[^\"]*)*)\"[^)]*name='([^']+)'",
                r"ToolCallExecutionEvent\([^)]*content=\[FunctionExecutionResult\([^)]*content='([^']*(?:\\.[^']*)*)'[^)]*name='([^']+)'"
            ]
            
            for pattern in result_patterns:
                matches = re.findall(pattern, raw_output, re.DOTALL)
                for i, (content, func_name) in enumerate(matches, 1):
                    clean_content = AgentOutputParser._clean_content(content)
                    components.append({
                        "type": "tool_result",
                        "number": i,
                        "function": func_name,
                        "result": clean_content
                    })
            
            # 4. Extract agent responses - FIXED to capture source and handle hybrid_carbon_assistant
            agent_pattern = r"TextMessage\([^)]*source='([^']*assistant[^']*)'[^)]*content='([^']*(?:\\.[^']*)*)'"
            agent_matches = re.findall(agent_pattern, raw_output, re.DOTALL)
            for i, (source, content) in enumerate(agent_matches, 1):
                clean_content = AgentOutputParser._clean_content(content)
                if len(clean_content.strip()) > 10:  # Only substantial responses
                    components.append({
                        "type": "agent_response",
                        "number": i,
                        "source": source,
                        "content": clean_content,
                        "formatted_content": AgentOutputParser._format_for_display(clean_content)
                    })
            
            formatted_data["components"] = components
            
            # 5. Create formatted string - FIXED to handle empty components
            if components:
                formatted_parts = []
                for comp in components:
                    if comp["type"] == "user_message":
                        formatted_parts.append(f"👤 USER MESSAGE {comp['number']}:\n{comp['content']}")
                    elif comp["type"] == "tool_call":
                        formatted_parts.append(f"🔧 TOOL CALL {comp['number']} - {comp['function']}:\nArguments: {comp['arguments']}")
                    elif comp["type"] == "tool_result":
                        formatted_parts.append(f"📊 TOOL RESULT {comp['number']} - {comp['function']}:\n{comp['result']}")
                    elif comp["type"] == "agent_response":
                        formatted_parts.append(f"🤖 AGENT RESPONSE {comp['number']} ({comp.get('source', 'assistant')}):\n{comp['formatted_content']}")
                
                separator = "=" * 80
                section_separator = "-" * 40
                formatted_data["formatted"] = f"\n{separator}\n" + f"\n{section_separator}\n".join(formatted_parts) + f"\n{separator}"
            else:
                # FIXED: Provide meaningful fallback when no components found
                formatted_data["formatted"] = f"⚠️ PARSING FAILED - No components extracted\n\nRaw output preview:\n{raw_output[:1000]}{'...[TRUNCATED]' if len(raw_output) > 1000 else ''}"
                
                # Add debug info
                formatted_data["debug_info"] = {
                    "raw_length": len(raw_output),
                    "contains_textmessage": "TextMessage(" in raw_output,
                    "contains_toolcall": "ToolCallRequestEvent(" in raw_output,
                    "contains_assistant": "assistant" in raw_output.lower(),
                    "sample_start": raw_output[:200]
                }
            
        except Exception as e:
            logger.warning(f"Error formatting raw output: {e}")
            formatted_data["formatted"] = f"❌ ERROR formatting output: {str(e)}\n\nRaw output preview:\n{raw_output[:500]}..."
            formatted_data["components"] = []
            formatted_data["error"] = str(e)
        
        return formatted_data
    
    def _save_results(self, report: HybridEvaluationReport):
        """Save evaluation results with improved JSON formatting"""
        try:
            output_dir = Path(self.config["output_dir"])
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save comprehensive JSON report with custom serialization
            json_file = output_dir / f"hybrid_evaluation_{report.agent_name}_{timestamp}.json"
            
            # Convert to dict with custom formatting
            report_dict = self._format_report_for_json(report)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=str, ensure_ascii=False)
            
            # Save CSV for analysis (simplified for spreadsheet use)
            csv_file = output_dir / f"detailed_results_{report.agent_name}_{timestamp}.csv"
            
            # Create simplified CSV data
            csv_data = []
            for result in report.detailed_results:
                csv_row = {
                    'test_case_id': result.test_case_id,
                    'status': result.status.value if hasattr(result.status, 'value') else str(result.status),
                    'execution_time': result.execution_time,
                    'rule_score': result.rule_based_score,
                    'llm_score': result.llm_quality_score,
                    'final_score': result.final_score,
                    'functions_called': ', '.join(result.functions_called),
                    'has_errors': result.structured_output.get('summary', {}).get('has_errors', False),
                    'tool_calls': result.structured_output.get('summary', {}).get('total_tool_calls', 0),
                    'response_length': result.structured_output.get('summary', {}).get('response_length', 0),
                    'main_response_preview': result.agent_responses[0][:100] + '...' if result.agent_responses else 'No response'
                }
                csv_data.append(csv_row)
            
            import pandas as pd
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False)
            
            # Save separate detailed conversation logs
            conversations_file = output_dir / f"conversation_logs_{report.agent_name}_{timestamp}.json"
            conversation_data = {
                "evaluation_metadata": {
                    "agent_name": report.agent_name,
                    "evaluation_date": report.evaluation_date,
                    "total_tests": report.total_tests
                },
                "conversations": []
            }
            
            for result in report.detailed_results:
                conversation_data["conversations"].append({
                    "test_case_id": result.test_case_id,
                    "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                    "conversation_flow": result.conversation_flow,
                    "tool_interactions": result.tool_interactions,
                    "agent_responses": result.agent_responses,
                    "structured_summary": result.structured_output.get("summary", {}),
                    "errors": result.structured_output.get("errors", [])
                })
            
            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Results saved to:")
            logger.info(f"  - Main report: {json_file}")
            logger.info(f"  - CSV summary: {csv_file}")
            logger.info(f"  - Conversation logs: {conversations_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _format_report_for_json(self, report: HybridEvaluationReport) -> Dict:
        """Format report for JSON with better structure"""
        
        # Convert main report
        report_dict = asdict(report)
        
        # Enhance detailed results formatting
        enhanced_results = []
        for result in report.detailed_results:
            result_dict = {
                "test_metadata": {
                    "test_case_id": result.test_case_id,
                    "timestamp": result.timestamp,
                    "execution_time": result.execution_time
                },
                "evaluation_results": {
                    "final_status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                    "final_score": result.final_score,
                    "confidence_level": result.confidence_level,
                    "rule_based": {
                        "status": result.rule_based_status.value if hasattr(result.rule_based_status, 'value') else str(result.rule_based_status),
                        "score": result.rule_based_score,
                        "functions_called": result.functions_called,
                        "behaviors_observed": result.behaviors_observed,
                        "keyword_matches": result.keyword_matches
                    },
                    "llm_based": {
                        "status": result.llm_status.value if result.llm_status and hasattr(result.llm_status, 'value') else str(result.llm_status) if result.llm_status else None,
                        "quality_score": result.llm_quality_score,
                        "confidence": result.llm_confidence,
                        "reasoning": result.llm_reasoning,
                        "feedback": result.llm_feedback
                    },
                    "consistency_score": result.consistency_score
                },
                "agent_interaction": {
                    "conversation_summary": result.structured_output.get("summary", {}),
                    "conversation_flow": result.conversation_flow,
                    "tool_interactions": result.tool_interactions,
                    "agent_responses": [
                        {
                            "response_id": resp_data.get("response_id", f"Response{i+1:02d}") if isinstance(resp_data, dict) else f"Response{i+1:02d}",
                            "content": resp_data.get("content", str(resp_data)) if isinstance(resp_data, dict) else str(resp_data),
                            "formatted_content": resp_data.get("formatted_content", resp_data.get("content", str(resp_data))) if isinstance(resp_data, dict) else str(resp_data),
                            "length": resp_data.get("length", len(str(resp_data))) if isinstance(resp_data, dict) else len(str(resp_data)),
                            "has_recommendations": resp_data.get("has_recommendations", False) if isinstance(resp_data, dict) else False,
                            "has_time_info": resp_data.get("has_time_info", False) if isinstance(resp_data, dict) else False,
                            "has_data": resp_data.get("has_data", False) if isinstance(resp_data, dict) else False
                        } for i, resp_data in enumerate(result.agent_responses)
                    ],
                    "errors": result.structured_output.get("errors", [])
                },
                "feedback_and_suggestions": {
                    "detailed_feedback": result.detailed_feedback,
                    "improvement_suggestions": result.improvement_suggestions
                },
                "raw_data": {
                    "raw_output_preview": result.raw_output[:1000] + "\n...[TRUNCATED]..." if len(result.raw_output) > 1000 else result.raw_output,
                    "raw_output_length": len(result.raw_output),
                    "formatted_raw_output": self._format_raw_output_for_display(result.raw_output),
                    "error_message": result.error_message
                }
            }
            enhanced_results.append(result_dict)
        
        # Replace detailed_results with enhanced version
        report_dict["detailed_results"] = enhanced_results
        
        return report_dict
    
    def print_report_summary(self, report: HybridEvaluationReport):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 80)
        print("🔄 HYBRID AGENT EVALUATION REPORT")
        print("=" * 80)
        
        # Add debugging section for development
        if logger.level <= logging.DEBUG:
            print(f"\n🔧 DEBUG INFO:")
            status_counts = {}
            for result in report.detailed_results:
                status_val = result.status.value if hasattr(result.status, 'value') else str(result.status)
                status_counts[status_val] = status_counts.get(status_val, 0) + 1
            print(f"Status counts from detailed results: {status_counts}")
            
            if report.evaluation_mode == "hybrid":
                rule_llm_comparison = []
                for result in report.detailed_results:
                    if result.llm_status is not None:
                        rule_val = result.rule_based_status.value if hasattr(result.rule_based_status, 'value') else str(result.rule_based_status)
                        llm_val = result.llm_status.value if hasattr(result.llm_status, 'value') else str(result.llm_status)
                        final_val = result.status.value if hasattr(result.status, 'value') else str(result.status)
                        rule_llm_comparison.append(f"{rule_val}->{llm_val}->{final_val}")
                print(f"Sample rule->llm->final progressions: {rule_llm_comparison[:5]}")
        
        print("=" * 80)
        
        # Basic info
        print(f"🤖 Agent: {report.agent_name}")
        print(f"📅 Date: {report.evaluation_date}")
        print(f"⚙️  Mode: {report.evaluation_mode}")
        print(f"🧪 Total Tests: {report.total_tests}")
        
        # Results overview
        if report.total_tests > 0:
            print(f"\n📊 RESULTS OVERVIEW:")
            print(f"✅ Passed: {report.passed_tests} ({report.pass_rate():.1%})")
            print(f"❌ Failed: {report.failed_tests} ({report.failed_tests/report.total_tests:.1%})")
            print(f"⚠️  Errors: {report.error_tests}")
            print(f"⏰ Timeouts: {report.timeout_tests}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"Average Execution Time: {report.avg_execution_time:.2f}s")
        print(f"Rule-based Accuracy: {report.rule_based_accuracy:.2f}")
        print(f"LLM Agreement Rate: {report.llm_agreement_rate:.2f}")
        print(f"Consistency Score: {report.consistency_score:.2f}")
        
        # Quality scores
        print(f"\n⭐ QUALITY SCORES:")
        print(f"Rule-based Score: {report.avg_rule_score:.2f}")
        print(f"LLM Quality Score: {report.avg_llm_score:.2f}")
        print(f"Combined Score: {report.avg_combined_score:.2f}")
        
        # Recommendations
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # LLM insights
        if report.llm_insights:
            print(f"\n🧠 LLM INSIGHTS:")
            for insight in report.llm_insights[:3]:
                print(f"  • {insight}")
        
        # Performance bottlenecks
        if report.performance_bottlenecks:
            print(f"\n🚨 PERFORMANCE ISSUES:")
            for bottleneck in report.performance_bottlenecks:
                print(f"  • {bottleneck}")
        
        # Detailed results sample
        print(f"\n📋 SAMPLE DETAILED RESULTS:")
        for result in report.detailed_results[:5]:
            # Fix emoji logic to handle both enum and string status
            status_val = result.status.value if hasattr(result.status, 'value') else str(result.status)
            status_emoji = "✅" if status_val == "PASS" else "❌" if status_val == "FAIL" else "⚠️"
            
            confidence = f"({result.confidence_level})" if result.confidence_level != "low" else ""
            print(f"  {status_emoji} {result.test_case_id}: {status_val} "
                  f"[R:{result.rule_based_score:.2f}|L:{result.llm_quality_score:.2f}] "
                  f"{confidence}")
        
        if len(report.detailed_results) > 5:
            print(f"  ... and {len(report.detailed_results) - 5} more results")
        
        print("=" * 80)