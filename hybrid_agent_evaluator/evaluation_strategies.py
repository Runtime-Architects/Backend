"""
Evaluation strategies for the hybrid agent evaluator
Separates rule-based and LLM-based evaluation logic
"""

import re
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

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
    """Result from LLM evaluation"""
    status: EvaluationStatus
    quality_score: float
    confidence: float
    semantic_similarity: float
    reasoning: str
    feedback: List[str]
    improvement_suggestions: List[str]
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
    """Rule-based evaluation strategy with improved logic"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.function_weight = config.get("function_weight", 0.4)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        self.behavior_weight = config.get("behavior_weight", 0.3)
    
    async def evaluate(self, test_case: Dict, output_text: str) -> RuleBasedResult:
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
            # Extract functions
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
    
    def _extract_functions(self, output_text: str) -> List[str]:
        """Extract function calls from output with improved detection and error handling"""
        
        if not output_text or not isinstance(output_text, str):
            return []
        
        functions_called = []
        
        try:
            # Handle AutoGen structured message outputs
            if "ToolCallRequestEvent" in output_text and "FunctionCall" in output_text:
                pattern = r"FunctionCall\([^)]*name='([^']+)'"
                matches = re.findall(pattern, output_text)
                functions_called.extend(matches)
            
            # Check for ToolCallExecutionEvent (successful execution)
            if "ToolCallExecutionEvent" in output_text:
                pattern = r"name='([^']+)'[^}]*call_id"
                matches = re.findall(pattern, output_text)
                functions_called.extend(matches)
            
            # Fallback patterns for different agent frameworks
            function_patterns = [
                r"calling function[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)",
                r"executing[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)",
                r"tool[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)",
                r"function[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)\(",
            ]
            
            for pattern in function_patterns:
                try:
                    matches = re.findall(pattern, output_text, re.IGNORECASE)
                    functions_called.extend(matches)
                except Exception as e:
                    logger.debug(f"Error in pattern {pattern}: {e}")
                    continue
            
            # Specific function name detection
            known_functions = [
                "get_emission_analysis", "emission_tool", "PythonCodeExecutionTool",
                "search_tool", "calculator", "file_reader", "web_search", "data_retrieval"
            ]
            
            output_lower = output_text.lower()
            for func in known_functions:
                try:
                    if func.lower() in output_lower and func not in functions_called:
                        functions_called.append(func)
                except Exception as e:
                    logger.debug(f"Error checking function {func}: {e}")
                    continue
            
        except Exception as e:
            logger.debug(f"Error in function extraction: {e}")
            # Return what we have so far
        
        return list(set(functions_called))  # Remove duplicates
    
    def _analyze_behaviors(self, test_case: Dict, output_text: str) -> List[str]:
        """Analyze behaviours with enhanced detection"""
        behaviors = []
        output_lower = output_text.lower()
        
        # Extract clean agent response
        agent_response = self._extract_agent_response(output_text)
        agent_response_lower = agent_response.lower()
        
        # Behaviour indicators
        behavior_indicators = {
            "correct_function_call": [
                "ToolCallRequestEvent", "get_emission_analysis", "retrieved", "analyzed",
                "FunctionCall", "ToolCallExecutionEvent", "data shows", "analysis reveals"
            ],
            "high_quality_response": [
                "detailed", "comprehensive", "analysis", "recommendations", 
                "windows", "intensity", "schedule", "tips", "optimal", "best time",
                "summary", "statistics", "insights"
            ],
            "user_friendly": [
                "best time", "recommend", "suggest", "tips", "help", "easy", 
                "simply", "here", "emoji", "🌱", "⚡", "🔥", "windows",
                "you should", "i recommend", "here's what"
            ],
            "proper_error_handling": [
                "sorry", "unable", "limitation", "try", "please", "invalid", 
                "cannot", "not available", "error", "failed", "apologize",
                "unfortunately", "however"
            ],
            "domain_expertise": [
                "carbon", "emission", "co2", "intensity", "sustainability", 
                "renewable", "grid", "electricity", "ireland", "roi", 
                "northern ireland", "kwh", "megawatt", "fossil fuel"
            ],
            "consistent_output": [
                # This is evaluated at the consistency analysis level
            ]
        }
        
        expected_behaviors = test_case.get("expected_behavior", [])
        
        for behavior in expected_behaviors:
            if behavior == "consistent_output":
                continue  # Handled at consistency level
            
            if behavior in behavior_indicators:
                indicators = behavior_indicators[behavior]
                # Check both full output and clean response
                full_text = output_text.lower() + " " + agent_response_lower
                
                # Count matches and require a threshold
                matches = sum(1 for indicator in indicators if indicator.lower() in full_text)
                threshold = max(1, len(indicators) // 3)  # At least 1/3 of indicators
                
                if matches >= threshold:
                    behaviors.append(behavior)
        
        return behaviors
    
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
    
    def _check_keywords(self, test_case: Dict, output_text: str) -> tuple[int, int]:
        """Check for expected keywords with fuzzy matching"""
        expected_keywords = test_case.get("expected_output_keywords", [])
        if not expected_keywords:
            return 0, 0
        
        # Clean text for matching
        clean_output = self._extract_agent_response(output_text).lower()
        
        matches = 0
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            
            # Direct match
            if keyword_lower in clean_output:
                matches += 1
                continue
            
            # Fuzzy matching for common variations
            variations = self._generate_keyword_variations(keyword_lower)
            if any(var in clean_output for var in variations):
                matches += 1
        
        return matches, len(expected_keywords)
    
    def _generate_keyword_variations(self, keyword: str) -> List[str]:
        """Generate variations of a keyword for fuzzy matching"""
        variations = [keyword]
        
        # Common plurals/singulars
        if keyword.endswith('s'):
            variations.append(keyword[:-1])
        else:
            variations.append(keyword + 's')
        
        # Common word replacements
        replacements = {
            'co2': ['carbon dioxide', 'carbon', 'emissions'],
            'ireland': ['roi', 'republic of ireland', 'irish'],
            'time': ['timing', 'schedule', 'when'],
            'appliance': ['device', 'equipment'],
            'charge': ['charging', 'load'],
            'intensity': ['level', 'rate', 'amount']
        }
        
        for original, alts in replacements.items():
            if original in keyword:
                for alt in alts:
                    variations.append(keyword.replace(original, alt))
            if keyword in alts:
                variations.append(original)
        
        return variations
    
    def _calculate_score(self, test_case: Dict, functions_called: List[str], 
                        behaviors_observed: List[str], keyword_matches: int, 
                        total_keywords: int) -> float:
        """Calculate weighted score based on multiple factors"""
        
        # Function score
        expected_functions = test_case.get("expected_functions", [])
        if expected_functions:
            function_score = len([f for f in expected_functions if f in functions_called]) / len(expected_functions)
        else:
            # If no functions expected, check if we correctly didn't call any
            function_score = 1.0 if not functions_called else 0.8
        
        # Behavior score
        expected_behaviors = test_case.get("expected_behavior", [])
        if expected_behaviors:
            behavior_score = len([b for b in expected_behaviors if b in behaviors_observed]) / len(expected_behaviors)
        else:
            behavior_score = 1.0
        
        # Keyword score
        keyword_score = keyword_matches / total_keywords if total_keywords > 0 else 1.0
        
        # Weighted combination
        total_score = (
            function_score * self.function_weight +
            behavior_score * self.behavior_weight +
            keyword_score * self.keyword_weight
        )
        
        return min(1.0, total_score)  # Cap at 1.0
    
    def _determine_status(self, test_case: Dict, functions_called: List[str], 
                         behaviors_observed: List[str], keyword_matches: int, 
                         total_keywords: int, output_text: str) -> tuple[EvaluationStatus, str, List[str]]:
        """Determine pass/fail status with improved and more balanced logic"""
        
        issues = []
        reasoning_parts = []
        
        # Check for critical errors
        critical_errors = [
            "exception occurred", "traceback", "failed to execute",
            "error executing", "cannot complete", "tool failed"
        ]
        
        agent_response = self._extract_agent_response(output_text).lower()
        has_critical_error = any(error in agent_response for error in critical_errors)
        
        if has_critical_error:
            issues.append("Critical execution error detected")
            reasoning_parts.append("Failed due to critical execution error")
            return EvaluationStatus.FAIL, "; ".join(reasoning_parts), issues
        
        # Check minimum response length (more lenient)
        if len(agent_response.strip()) < 10:  # Reduced from 20
            issues.append("Response too short")
            reasoning_parts.append("Response length insufficient")
            return EvaluationStatus.FAIL, "; ".join(reasoning_parts), issues
        
        # Function requirements (more flexible)
        expected_functions = test_case.get("expected_functions", [])
        missing_functions = []
        
        if expected_functions:
            missing_functions = [f for f in expected_functions if f not in functions_called]
            if missing_functions:
                issues.append(f"Missing functions: {', '.join(missing_functions)}")
                reasoning_parts.append(f"Missing {len(missing_functions)} expected functions")
                
                # More lenient function checking
                category = test_case.get("category", "")
                if category == "irrelevant_query":
                    # For irrelevant queries, not calling functions is often correct
                    reasoning_parts.append("Acceptable for irrelevant query")
                elif len(missing_functions) == len(expected_functions):
                    # Only fail if ALL expected functions are missing
                    return EvaluationStatus.FAIL, "; ".join(reasoning_parts), issues
                # Otherwise, note the issue but don't automatically fail
        
        # Behavior requirements (more flexible)
        expected_behaviors = test_case.get("expected_behavior", [])
        missing_behaviors = []
        
        if expected_behaviors:
            missing_behaviors = [b for b in expected_behaviors if b not in behaviors_observed]
            if missing_behaviors:
                issues.append(f"Missing behaviors: {', '.join(missing_behaviors)}")
                reasoning_parts.append(f"Missing {len(missing_behaviors)} expected behaviors")
                
                # Only fail on critical missing behaviors
                critical_behaviors = ["correct_function_call"]
                critical_missing = [b for b in missing_behaviors if b in critical_behaviors]
                
                # Allow for proper error handling as acceptable behavior
                if "proper_error_handling" in behaviors_observed:
                    reasoning_parts.append("Has proper error handling")
                elif critical_missing and len(agent_response) > 50:
                    # Only fail if missing critical behaviors AND response is substantial
                    if not any(phrase in agent_response for phrase in ["sorry", "cannot", "unable", "limitation"]):
                        return EvaluationStatus.FAIL, "; ".join(reasoning_parts), issues
        
        # Keyword requirements (more lenient threshold)
        keyword_issues = []
        if total_keywords > 0:
            keyword_ratio = keyword_matches / total_keywords
            if keyword_ratio < 0.3:  # Reduced from 0.4
                keyword_issues.append(f"Low keyword match: {keyword_matches}/{total_keywords}")
                reasoning_parts.append(f"Only {keyword_ratio:.1%} keyword match")
                
                # Don't fail purely on keywords unless very low
                if keyword_ratio < 0.1 and len(agent_response) < 100:
                    issues.extend(keyword_issues)
                    return EvaluationStatus.FAIL, "; ".join(reasoning_parts), issues
        
        # Overall assessment
        total_issues = len(issues)
        
        if total_issues == 0:
            reasoning_parts.append("All criteria met")
            return EvaluationStatus.PASS, "; ".join(reasoning_parts), issues
        elif total_issues == 1 and any("keyword" in issue.lower() for issue in issues):
            reasoning_parts.append("Minor keyword issues only")
            return EvaluationStatus.PASS, "; ".join(reasoning_parts), issues
        elif total_issues <= 2 and len(agent_response) > 100:
            # If response is substantial, be more lenient
            reasoning_parts.append("Minor issues but substantial response")
            return EvaluationStatus.PASS, "; ".join(reasoning_parts), issues
        else:
            return EvaluationStatus.FAIL, "; ".join(reasoning_parts), issues

class LLMJudgeStrategy(BaseEvaluationStrategy):
    """LLM-based evaluation strategy"""
    
    def __init__(self, llm_judge, config: Dict):
        super().__init__(config)
        self.llm_judge = llm_judge
        self.quality_threshold = config.get("llm_quality_threshold", 0.7)
        self.confidence_threshold = config.get("llm_confidence_threshold", 0.6)
    
    async def evaluate(self, test_case: Dict, output_text: str, 
                      rule_result: Optional[RuleBasedResult] = None) -> LLMResult:
        """Perform LLM-based evaluation with robust error handling"""
        
        # Validate inputs
        if not test_case or not isinstance(test_case, dict):
            return LLMResult(
                status=EvaluationStatus.ERROR,
                quality_score=0.0,
                confidence=0.0,
                semantic_similarity=0.0,
                reasoning="Invalid test case provided",
                feedback=[],
                improvement_suggestions=[]
            )
        
        if not output_text or not isinstance(output_text, str):
            return LLMResult(
                status=EvaluationStatus.ERROR,
                quality_score=0.0,
                confidence=0.0,
                semantic_similarity=0.0,
                reasoning="Empty or invalid output text",
                feedback=[],
                improvement_suggestions=[]
            )
        
        try:
            # Get clean agent response
            agent_response = self._extract_agent_response(output_text)
            
            if not agent_response or len(agent_response.strip()) < 10:
                return LLMResult(
                    status=EvaluationStatus.FAIL,
                    quality_score=0.0,
                    confidence=0.8,
                    semantic_similarity=0.0,
                    reasoning="Response too short or empty",
                    feedback=["Response is too short to evaluate"],
                    improvement_suggestions=["Provide more detailed responses"]
                )
            
            # Semantic similarity analysis (if multiple outputs available)
            semantic_similarity = 0.0
            
            # Function analysis
            function_analysis = None
            if self.config.get("use_function_analysis", True):
                try:
                    function_analysis = await self.llm_judge.analyze_function_calls(
                        agent_response,
                        test_case.get("expected_functions", []),
                        test_case["query"],
                        test_case.get("available_functions", [])
                    )
                except Exception as e:
                    logger.warning(f"Function analysis failed: {e}")
            
            # Behavior analysis
            behavior_analysis = None
            if self.config.get("use_behavior_analysis", True):
                try:
                    behavior_analysis = await self.llm_judge.analyze_behavior(
                        agent_response,
                        test_case["query"],
                        test_case.get("expected_behavior", []),
                        test_case.get("domain_context", "")
                    )
                except Exception as e:
                    logger.warning(f"Behavior analysis failed: {e}")
            
            # Overall quality evaluation
            try:
                quality_result = await self.llm_judge.evaluate_semantic_similarity(
                    [agent_response],
                    test_case.get("expected_output_keywords", []),
                    test_case.get("domain_context", "")
                )
            except Exception as e:
                logger.warning(f"Quality evaluation failed: {e}")
                quality_result = type('obj', (object,), {
                    'quality_score': 0.5,
                    'confidence': 0.3,
                    'reasoning': f"LLM evaluation failed: {str(e)}",
                    'improvement_suggestions': []
                })()
            
            # Determine status based on LLM analysis
            status = self._determine_llm_status(function_analysis, behavior_analysis, quality_result)
            
            # Combine feedback
            feedback = []
            if function_analysis and hasattr(function_analysis, 'reasoning'):
                feedback.append(f"Function analysis: {function_analysis.reasoning}")
            if behavior_analysis and hasattr(behavior_analysis, 'specific_feedback'):
                feedback.extend(behavior_analysis.specific_feedback)
            
            improvement_suggestions = []
            if hasattr(quality_result, 'improvement_suggestions') and quality_result.improvement_suggestions:
                improvement_suggestions.extend(quality_result.improvement_suggestions)
            
            return LLMResult(
                status=status,
                quality_score=getattr(quality_result, 'quality_score', 0.0),
                confidence=getattr(quality_result, 'confidence', 0.0),
                semantic_similarity=semantic_similarity,
                reasoning=getattr(quality_result, 'reasoning', 'No reasoning available'),
                feedback=feedback,
                improvement_suggestions=improvement_suggestions,
                function_analysis=function_analysis.__dict__ if function_analysis else None,
                behavior_analysis=behavior_analysis.__dict__ if behavior_analysis else None
            )
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return LLMResult(
                status=EvaluationStatus.ERROR,
                quality_score=0.0,
                confidence=0.0,
                semantic_similarity=0.0,
                reasoning=f"LLM evaluation failed: {str(e)}",
                feedback=[],
                improvement_suggestions=[]
            )
    
    def _extract_agent_response(self, raw_output: str) -> str:
        """Extract agent response - same logic as rule-based strategy"""
        pattern = r"TextMessage\([^)]*content='([^']*(?:\\.[^']*)*)'"
        matches = re.findall(pattern, raw_output, re.DOTALL)
        
        if matches:
            responses = []
            for match in matches:
                clean_match = match.replace('\\n', '\n').replace("\\'", "'")
                if len(clean_match) > 50:
                    responses.append(clean_match)
            
            if responses:
                return max(responses, key=len)
        
        # Fallback
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
    
    def _determine_llm_status(self, function_analysis, behavior_analysis, quality_result) -> EvaluationStatus:
        """Determine status based on LLM analysis"""
        
        # High confidence override
        if quality_result.confidence >= 0.9 and quality_result.quality_score >= 0.8:
            return EvaluationStatus.PASS
        
        # Quality-based determination
        if quality_result.quality_score >= self.quality_threshold:
            return EvaluationStatus.PASS
        elif quality_result.quality_score < 0.3:
            return EvaluationStatus.FAIL
        
        # Function analysis consideration
        if function_analysis and hasattr(function_analysis, 'missing_functions'):
            if function_analysis.missing_functions and len(function_analysis.missing_functions) > 0:
                # Check confidence - if LLM is confident functions are missing, lean toward fail
                avg_confidence = 0.0
                if hasattr(function_analysis, 'confidence_scores') and function_analysis.confidence_scores:
                    avg_confidence = sum(function_analysis.confidence_scores.values()) / len(function_analysis.confidence_scores)
                
                if avg_confidence >= 0.8:
                    return EvaluationStatus.FAIL
        
        # Behavior analysis consideration
        if behavior_analysis and hasattr(behavior_analysis, 'quality_metrics'):
            if behavior_analysis.quality_metrics:
                avg_behavior_quality = sum(behavior_analysis.quality_metrics.values()) / len(behavior_analysis.quality_metrics)
                if avg_behavior_quality >= 0.7:
                    return EvaluationStatus.PASS
        
        # Default based on overall confidence and quality
        if quality_result.confidence >= self.confidence_threshold and quality_result.quality_score >= 0.5:
            return EvaluationStatus.PASS
        else:
            return EvaluationStatus.FAIL