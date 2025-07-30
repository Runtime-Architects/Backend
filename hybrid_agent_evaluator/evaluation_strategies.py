"""
Evaluation strategies with ground truth support and LLM evaluation
Contains rule-based and LLM-based evaluation approaches for agent testing
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Import LLM judge
from llm_judge import LLMJudge, GroundTruthEvaluationResult

logger = logging.getLogger(__name__)

class EvaluationMode(Enum):
    RULE_BASED_ONLY = "rule_based_only"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"

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
class RuleBasedResult:
    """Result from rule-based evaluation"""
    status: EvaluationStatus
    score: float
    functions_called: List[str]
    behaviors_observed: List[str]
    keyword_matches: int
    total_keywords: int
    reasoning: str
    issues_found: List[str]

@dataclass
class LLMResult:
    """LLM evaluation result with ground truth comparison"""
    status: EvaluationStatus
    quality_score: float
    confidence: float
    semantic_similarity: float
    reasoning: str
    feedback: List[str]
    improvement_suggestions: List[str]
    
    # Additional fields for ground truth evaluation
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    actionability_score: float = 0.0
    format_score: float = 0.0
    
    # Ground truth specific
    ground_truth_used: bool = False
    specific_matches: List[str] = None
    specific_gaps: List[str] = None
    penalties_applied: List[str] = None
    
    function_analysis: Optional[Dict] = None
    behavior_analysis: Optional[Dict] = None

class BaseEvaluationStrategy(ABC):
    """Base class for evaluation strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractmethod
    async def evaluate(self, test_case: Dict, output_text: str, **kwargs) -> Any:
        """Evaluate the test case"""
        pass

class RuleBasedStrategy(BaseEvaluationStrategy):
    """Rule-based evaluation strategy - keeping existing implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.function_weight = config.get("function_weight", 0.4)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        self.behavior_weight = config.get("behavior_weight", 0.3)
    
    async def evaluate(self, test_case: Dict, output_text: str, conversation_log: List = None) -> RuleBasedResult:
        """Perform rule-based evaluation with robust error handling"""
        
        # Validate inputs
        if not test_case or not isinstance(test_case, dict):
            return RuleBasedResult(
                status=EvaluationStatus.ERROR,
                score=0.0,
                functions_called=[],
                behaviors_observed=[],
                keyword_matches=0,
                total_keywords=0,
                reasoning="Invalid test case",
                issues_found=["Invalid test case provided"]
            )
        
        if not output_text or not isinstance(output_text, str):
            return RuleBasedResult(
                status=EvaluationStatus.ERROR,
                score=0.0,
                functions_called=[],
                behaviors_observed=[],
                keyword_matches=0,
                total_keywords=0,
                reasoning="Empty or invalid output text",
                issues_found=["No output to evaluate"]
            )
        
        try:
            # Extract functions - prefer conversation log if available
            if conversation_log:
                functions_called = self._extract_functions_from_log(conversation_log)
            else:
                functions_called = self._extract_functions(output_text)
            
            # Check behaviors
            behaviors_observed = self._analyze_behaviors(test_case, output_text)
            
            # Check keywords
            keyword_matches, total_keywords = self._check_keywords(test_case, output_text)
            
            # Calculate score
            score = self._calculate_score(test_case, functions_called, behaviors_observed, 
                                        keyword_matches, total_keywords)
            
            # Determine status
            status, reasoning, issues = self._determine_status(test_case, functions_called, 
                                                             behaviors_observed, keyword_matches, 
                                                             total_keywords, output_text)
            
            logger.debug(f"Rule-based evaluation for {test_case.get('id', 'unknown')}: "
                        f"functions={functions_called}, behaviors={behaviors_observed}, "
                        f"keywords={keyword_matches}/{total_keywords}, score={score:.2f}")
            
            return RuleBasedResult(
                status=status,
                score=score,
                functions_called=functions_called,
                behaviors_observed=behaviors_observed,
                keyword_matches=keyword_matches,
                total_keywords=total_keywords,
                reasoning=reasoning,
                issues_found=issues
            )
            
        except Exception as e:
            logger.error(f"Error in rule-based evaluation: {e}")
            return RuleBasedResult(
                status=EvaluationStatus.ERROR,
                score=0.0,
                functions_called=[],
                behaviors_observed=[],
                keyword_matches=0,
                total_keywords=0,
                reasoning=f"Evaluation error: {str(e)}",
                issues_found=[f"Evaluation failed: {str(e)}"]
            )
    
    # ... (keep all the existing rule-based methods unchanged)
    def _extract_functions(self, output_text: str) -> List[str]:
        """Extract function calls from output"""
        if not output_text or not isinstance(output_text, str):
            return []
        
        functions_called = []
        
        try:
            # Look for emission tool patterns
            emission_patterns = [
                r"get_emission_analysis",
                r"emission_tool",
                r"emission analysis",
                r"calling.*emission",
                r"executing.*emission",
                r"CO2 intensity",
                r"carbon intensity", 
                r"emission.*data",
                r"best time.*appliances",
                r"\d{1,2}:\d{2}.*(?:AM|PM|am|pm|hours?)",
            ]
            
            output_lower = output_text.lower()
            emission_evidence_count = 0
            
            for pattern in emission_patterns:
                if re.search(pattern, output_lower, re.IGNORECASE):
                    emission_evidence_count += 1
            
            if emission_evidence_count >= 2:
                if "get_emission_analysis" not in functions_called:
                    functions_called.append("get_emission_analysis")
                    
        except Exception as e:
            logger.debug(f"Error in function extraction: {e}")
        
        return functions_called
    
    def _extract_functions_from_log(self, conversation_log: List) -> List[str]:
        """Extract actual function calls from conversation log"""
        functions_called = []
        
        try:
            for event in conversation_log:
                # Check for ToolCallRequestEvent or similar
                if hasattr(event, 'content'):
                    content = event.content
                    if isinstance(content, list):
                        for item in content:
                            # Check for FunctionCall objects
                            if hasattr(item, 'name'):
                                functions_called.append(item.name)
                            # Check for dict with function name
                            elif isinstance(item, dict) and 'name' in item:
                                functions_called.append(item['name'])
                
                # Check for function execution results
                if hasattr(event, 'type') and 'ToolCall' in str(event.type):
                    # Try to extract function name from event metadata
                    if hasattr(event, 'metadata') and isinstance(event.metadata, dict):
                        if 'function_name' in event.metadata:
                            functions_called.append(event.metadata['function_name'])
        
        except Exception as e:
            logger.debug(f"Error extracting functions from log: {e}")
            # Fall back to text-based extraction
            return self._extract_functions(str(conversation_log))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_functions = []
        for func in functions_called:
            if func not in seen:
                seen.add(func)
                unique_functions.append(func)
        
        logger.debug(f"Extracted functions from conversation log: {unique_functions}")
        return unique_functions
    
    def _analyze_behaviors(self, test_case: Dict, output_text: str) -> List[str]:
        """Analyze behaviors in output"""
        behaviors = []
        
        output_lower = output_text.lower()
        
        behavior_indicators = {
            "correct_function_call": [
                "emission", "carbon", "co2", "intensity", "data", "analysis",
                "best time", "optimal", "recommend"
            ],
            "high_quality_response": [
                "recommend", "suggest", "analysis", "best", "optimal", 
                "carbon", "emission", "schedule", "time"
            ],
            "user_friendly": [
                "best time", "recommend", "suggest", "should", "you can",
                "morning", "evening", "🌱", "⚡", "🔥"
            ],
            "proper_error_handling": [
                "sorry", "unable", "cannot", "error", "failed", "apologize"
            ],
            "domain_expertise": [
                "carbon intensity", "co2", "emission", "renewable", "grid",
                "peak", "off-peak", "ireland", "roi"
            ]
        }
        
        expected_behaviors = test_case.get("expected_behavior", [])
        
        for behavior in expected_behaviors:
            if behavior in behavior_indicators:
                indicators = behavior_indicators[behavior]
                matches = sum(1 for indicator in indicators if indicator in output_lower)
                threshold = max(1, len(indicators) // 6)
                
                if matches >= threshold:
                    behaviors.append(behavior)
        
        return behaviors
    
    def _check_keywords(self, test_case: Dict, output_text: str) -> tuple[int, int]:
        """Check for expected keywords"""
        expected_keywords = test_case.get("expected_output_keywords", [])
        if not expected_keywords:
            return 0, 0
        
        output_lower = output_text.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in output_lower)
        
        return matches, len(expected_keywords)
    
    def _calculate_score(self, test_case: Dict, functions_called: List[str], 
                        behaviors_observed: List[str], keyword_matches: int, 
                        total_keywords: int) -> float:
        """Calculate weighted score"""
        expected_functions = test_case.get("expected_functions", [])
        if expected_functions:
            function_score = len([f for f in expected_functions if f in functions_called]) / len(expected_functions)
        else:
            function_score = 1.0 if not functions_called else 0.8
        
        expected_behaviors = test_case.get("expected_behavior", [])
        if expected_behaviors:
            behavior_score = len([b for b in expected_behaviors if b in behaviors_observed]) / len(expected_behaviors)
        else:
            behavior_score = 1.0
        
        keyword_score = keyword_matches / total_keywords if total_keywords > 0 else 1.0
        
        total_score = (
            function_score * self.function_weight +
            behavior_score * self.behavior_weight +
            keyword_score * self.keyword_weight
        )
        
        return min(1.0, total_score)
    
    def _determine_status(self, test_case: Dict, functions_called: List[str], 
                         behaviors_observed: List[str], keyword_matches: int, 
                         total_keywords: int, output_text: str) -> tuple[EvaluationStatus, str, List[str]]:
        """Determine pass/fail status with consistent criteria"""
        issues = []
        reasoning_parts = []
        
        # Check for critical errors
        if len(output_text.strip()) < 10:
            issues.append("Response too short")
            return EvaluationStatus.FAIL, "Response too short", issues
        
        # Function requirements - STRICT CHECK
        expected_functions = test_case.get("expected_functions", [])
        if expected_functions:
            missing_functions = [f for f in expected_functions if f not in functions_called]
            if missing_functions:
                if test_case.get("category") != "irrelevant_query":
                    issues.append(f"Missing required functions: {missing_functions}")
                    reasoning_parts.append(f"Missing functions: {', '.join(missing_functions)}")
        
        # Calculate overall score for threshold check
        score = self._calculate_score(test_case, functions_called, behaviors_observed, 
                                    keyword_matches, total_keywords)
        
        # Apply consistent score threshold (matching LLM evaluator)
        score_threshold = self.config.get("rule_based_min_score", 0.6)
        if score < score_threshold:
            issues.append(f"Score below threshold: {score:.2f} < {score_threshold}")
            reasoning_parts.append(f"Insufficient score: {score:.2f}")
        
        # Behavior requirements check
        expected_behaviors = test_case.get("expected_behavior", [])
        if expected_behaviors:
            missing_behaviors = [b for b in expected_behaviors if b not in behaviors_observed]
            if len(missing_behaviors) > len(expected_behaviors) * 0.5:  # Missing more than half
                issues.append(f"Missing critical behaviors: {missing_behaviors}")
                reasoning_parts.append("Insufficient behavioral indicators")
        
        # Keyword coverage check
        if total_keywords > 0:
            keyword_coverage = keyword_matches / total_keywords
            if keyword_coverage < 0.4:  # Less than 40% keyword coverage
                issues.append(f"Low keyword coverage: {keyword_coverage:.1%}")
                reasoning_parts.append("Insufficient keyword matches")
        
        # Determine final status
        if len(issues) == 0:
            return EvaluationStatus.PASS, "All criteria met", issues
        elif len(issues) == 1 and score >= score_threshold * 0.9:  # Allow ONE minor issue if score is close
            return EvaluationStatus.PASS, "Minor issues but acceptable", issues
        else:
            return EvaluationStatus.FAIL, "; ".join(reasoning_parts) or "Multiple criteria not met", issues

class LLMJudgeStrategy(BaseEvaluationStrategy):
    """LLM-based evaluation strategy with ground truth support"""
    
    def __init__(self, llm_judge, config: Dict):
        super().__init__(config)
        self.llm_judge = llm_judge
        self.quality_threshold = config.get("llm_quality_threshold", 0.7)
        self.confidence_threshold = config.get("llm_confidence_threshold", 0.6)
        
        # Ensure we have a valid LLM judge
        if not self.llm_judge:
            logger.error("No LLM judge provided to LLMJudgeStrategy")
    
    async def evaluate(self, test_case: Dict, output_text: str, 
                      rule_result: Optional[RuleBasedResult] = None,
                      compressed_co2_data: Dict = None) -> LLMResult:
        """Perform LLM-based evaluation with ground truth comparison"""
        
        # Validate inputs
        if not test_case or not isinstance(test_case, dict):
            logger.error("Invalid test case provided to LLM evaluation")
            return self._create_error_result("Invalid test case provided")
        
        if not output_text or not isinstance(output_text, str):
            logger.error("Invalid output text provided to LLM evaluation")
            return self._create_error_result("Empty or invalid output text")
        
        # Check if LLM judge is available
        if not self.llm_judge:
            logger.error("LLM judge not available")
            return self._create_error_result("LLM judge not available")
        
        # Check for ground truth
        ground_truth = test_case.get("ground_truth")
        if not ground_truth:
            error_msg = f"No ground truth available for test case {test_case.get('id', 'unknown')}. Cannot perform LLM evaluation without reference data."
            logger.error(error_msg)
            return self._create_error_result(error_msg)
        
        try:
            logger.info(f"Starting LLM evaluation with ground truth for {test_case.get('id', 'unknown')}")
            
            # Use LLM judge with ground truth
            gt_result = await self.llm_judge.evaluate_with_ground_truth(
                agent_output=output_text,
                ground_truth=ground_truth,
                test_case=test_case,
                compressed_co2_data=compressed_co2_data
            )
            
            # Convert to LLMResult format
            llm_result = self._convert_ground_truth_result(gt_result, test_case)
            
            logger.info(f"LLM evaluation completed for {test_case.get('id', 'unknown')}: "
                       f"{llm_result.status.value} (score: {llm_result.quality_score:.2f})")
            
            return llm_result
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for {test_case.get('id', 'unknown')}: {e}")
            return self._create_error_result(f"LLM evaluation failed: {str(e)}")
    
    def _convert_ground_truth_result(self, gt_result: GroundTruthEvaluationResult, 
                                   test_case: Dict) -> LLMResult:
        """Convert ground truth result to LLM result format"""
        
        # Determine status based on overall score
        status = self._determine_status_from_score(gt_result.overall_score, gt_result.confidence)
        
        # Create comprehensive feedback
        feedback = [gt_result.reasoning]
        if gt_result.specific_matches:
            feedback.append(f"Matches found: {', '.join(gt_result.specific_matches[:3])}")
        if gt_result.specific_gaps:
            feedback.append(f"Missing elements: {', '.join(gt_result.specific_gaps[:3])}")
        if gt_result.penalties_applied:
            feedback.extend(gt_result.penalties_applied[:2])
        
        return LLMResult(
            status=status,
            quality_score=gt_result.overall_score,
            confidence=gt_result.confidence,
            semantic_similarity=gt_result.similarity_score,
            reasoning=gt_result.reasoning,
            feedback=feedback,
            improvement_suggestions=gt_result.improvement_suggestions,
            
            # Detailed scores
            accuracy_score=gt_result.accuracy_score,
            completeness_score=gt_result.completeness_score,
            clarity_score=gt_result.clarity_score,
            actionability_score=gt_result.actionability_score,
            format_score=gt_result.format_score,
            
            # Ground truth specific
            ground_truth_used=True,
            specific_matches=gt_result.specific_matches,
            specific_gaps=gt_result.specific_gaps,
            penalties_applied=gt_result.penalties_applied
        )
    
    def _determine_status_from_score(self, overall_score: float, confidence: float) -> EvaluationStatus:
        """Determine evaluation status from scores with calibrated thresholds"""
        
        # Primary threshold check (matching rule-based at 0.6)
        if overall_score >= self.quality_threshold:
            return EvaluationStatus.PASS
        
        # Close to threshold with good confidence - allow some flexibility
        elif overall_score >= self.quality_threshold * 0.9 and confidence >= self.confidence_threshold:
            # Score is within 10% of threshold and confidence is good
            return EvaluationStatus.PASS
        
        # Medium quality with very high confidence - semantic equivalence cases
        elif overall_score >= 0.5 and confidence >= 0.85:
            return EvaluationStatus.PASS
        
        # Clear fail cases
        elif overall_score < 0.4:
            return EvaluationStatus.FAIL
        
        # Borderline cases (0.4-0.6) - use confidence as tiebreaker
        else:
            # If confidence is high enough, give benefit of doubt
            if confidence >= 0.7 and overall_score >= 0.45:
                return EvaluationStatus.PASS
            else:
                return EvaluationStatus.FAIL
    
    # Removed _fallback_evaluation method - no more fake evaluations
    
    def _create_error_result(self, error_message: str) -> LLMResult:
        """Create error result"""
        return LLMResult(
            status=EvaluationStatus.ERROR,
            quality_score=0.0,
            confidence=0.0,
            semantic_similarity=0.0,
            reasoning=error_message,
            feedback=[f"Evaluation error: {error_message}"],
            improvement_suggestions=["Check LLM service configuration"],
            ground_truth_used=False
        )

# Note: LLMJudgeStrategy is the main class name