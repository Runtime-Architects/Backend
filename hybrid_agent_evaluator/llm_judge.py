"""
LLM-as-Judge implementation for our hybrid agent evaluation
Provides semantic evaluation capabilities using Azure OpenAI
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

logger = logging.getLogger(__name__)

@dataclass
class LLMEvaluationResult:
    """Results from LLM-based evaluation"""
    similarity_score: float
    consistency_score: float
    quality_score: float
    confidence: float
    reasoning: str
    specific_issues: List[str]
    improvement_suggestions: List[str]
    execution_time: float = 0.0

@dataclass
class FunctionCallAnalysis:
    """Analysis of function calls using LLM"""
    detected_functions: List[str]
    attempted_functions: List[str]
    missing_functions: List[str]
    confidence_scores: Dict[str, float]
    evidence: Dict[str, str]
    reasoning: str

@dataclass
class BehaviorAnalysis:
    """Behavior analysis using LLM"""
    behaviors_detected: List[str]
    quality_metrics: Dict[str, float]
    domain_expertise_score: float
    user_experience_score: float
    error_handling_score: float
    reasoning: str
    specific_feedback: List[str]

class LLMJudge:
    """LLM-as-Judge implementation for carbon agent evaluation"""
    
    def __init__(self, azure_client: AzureOpenAIChatCompletionClient, 
                 model_name: str = "gpt-4", max_retries: int = 3):
        self.client = azure_client
        self.model_name = model_name
        self.max_retries = max_retries
        self._test_connection()
    
    def _test_connection(self):
        """Test the LLM connection during initialization"""
        try:
            # Simple test to validate client is working
            logger.info(f"Testing LLM connection with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"LLM connection test failed: {e}")
    
    async def evaluate_semantic_similarity(self, outputs: List[str], 
                                         expected_keywords: List[str] = None,
                                         domain_context: str = "") -> LLMEvaluationResult:
        """Evaluate semantic similarity and quality of outputs"""
        
        if not outputs or all(not output.strip() for output in outputs):
            return LLMEvaluationResult(
                similarity_score=0.0, consistency_score=0.0, quality_score=0.0,
                confidence=1.0, reasoning="No valid outputs provided",
                specific_issues=["No content to evaluate"], improvement_suggestions=[]
            )
        
        prompt = self._build_enhanced_similarity_prompt(outputs, expected_keywords, domain_context)
        
        start_time = time.time()
        try:
            response = await self._call_llm_with_retry(prompt)
            result = self._parse_similarity_response(response)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"LLM similarity evaluation failed: {e}")
            return LLMEvaluationResult(
                similarity_score=0.0, consistency_score=0.0, quality_score=0.0,
                confidence=0.0, reasoning=f"Evaluation failed: {str(e)}",
                specific_issues=[f"LLM error: {str(e)}"], improvement_suggestions=[],
                execution_time=time.time() - start_time
            )
    
    async def analyze_function_calls(self, output: str, expected_functions: List[str],
                                   query: str, available_functions: List[str] = None) -> FunctionCallAnalysis:
        """Analyze function calls in agent output with enhanced detection"""
        
        if not output.strip():
            return FunctionCallAnalysis(
                detected_functions=[], attempted_functions=[], missing_functions=expected_functions,
                confidence_scores={}, evidence={}, reasoning="No output to analyze"
            )
        
        prompt = self._build_enhanced_function_prompt(output, expected_functions, query, available_functions)
        
        try:
            response = await self._call_llm_with_retry(prompt)
            return self._parse_function_analysis_response(response)
        except Exception as e:
            logger.error(f"LLM function analysis failed: {e}")
            return FunctionCallAnalysis(
                detected_functions=[], attempted_functions=[], missing_functions=expected_functions,
                confidence_scores={}, evidence={}, reasoning=f"Analysis failed: {str(e)}"
            )
    
    async def analyze_behavior(self, output: str, query: str, 
                             expected_behaviors: List[str],
                             domain_context: str = "") -> BehaviorAnalysis:
        """Analyze agent behavior and response quality with domain awareness"""
        
        if not output.strip():
            return BehaviorAnalysis(
                behaviors_detected=[], quality_metrics={}, domain_expertise_score=0.0,
                user_experience_score=0.0, error_handling_score=0.0,
                reasoning="No output to analyze", specific_feedback=[]
            )
        
        prompt = self._build_enhanced_behavior_prompt(output, query, expected_behaviors, domain_context)
        
        try:
            response = await self._call_llm_with_retry(prompt)
            return self._parse_behavior_analysis_response(response)
        except Exception as e:
            logger.error(f"LLM behavior analysis failed: {e}")
            return BehaviorAnalysis(
                behaviors_detected=[], quality_metrics={}, domain_expertise_score=0.0,
                user_experience_score=0.0, error_handling_score=0.0,
                reasoning=f"Analysis failed: {str(e)}", specific_feedback=[]
            )
    
    def _build_enhanced_similarity_prompt(self, outputs: List[str], 
                                        expected_keywords: List[str],
                                        domain_context: str) -> str:
        """Build enhanced prompt for semantic similarity evaluation"""
        
        # Clean outputs
        clean_outputs = [output.strip() for output in outputs if output.strip()]
        
        if len(clean_outputs) == 1:
            # Single output quality evaluation
            evaluation_text = f"AGENT OUTPUT TO EVALUATE:\n{clean_outputs[0]}"
            task_description = "Evaluate the quality and completeness of this agent output."
        else:
            # Multiple outputs consistency evaluation
            evaluation_text = "\n".join([f"Output {i+1}:\n{output}\n" for i, output in enumerate(clean_outputs)])
            task_description = "Evaluate the semantic consistency and quality across these agent outputs."
        
        keywords_section = ""
        if expected_keywords:
            keywords_section = f"\nEXPECTED CONTENT: The response should cover: {', '.join(expected_keywords)}"
        
        context_section = ""
        if domain_context:
            context_section = f"\nDOMAIN CONTEXT: {domain_context}"
        
        return f"""You are an expert AI agent evaluator. Your job is to assess agent outputs objectively and provide actionable feedback.

{task_description}
{context_section}

{evaluation_text}
{keywords_section}

EVALUATION CRITERIA:
1. SEMANTIC SIMILARITY (0-1): For multiple outputs, how similar are they in meaning? For single output, rate as 1.0.
2. CONSISTENCY (0-1): Are facts, recommendations, and style consistent across outputs?
3. QUALITY (0-1): Overall response quality considering:
   - Completeness: Addresses the query fully
   - Accuracy: Information appears correct and reliable
   - Helpfulness: Provides actionable, useful information
   - Clarity: Well-organized and easy to understand
4. CONFIDENCE (0-1): How confident are you in this evaluation?

IMPORTANT GUIDELINES:
- Focus on CONTENT and MEANING rather than exact wording
- Different phrasing of same information should score highly
- Consider domain expertise and context appropriateness
- Be objective but recognize that some variation in AI responses is normal

Respond in valid JSON format only:
{{
    "similarity_score": 0.0,
    "consistency_score": 0.0,
    "quality_score": 0.0,
    "confidence": 0.0,
    "reasoning": "Clear explanation of your evaluation",
    "specific_issues": ["issue1", "issue2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""
    
    def _build_enhanced_function_prompt(self, output: str, expected_functions: List[str], 
                                      query: str, available_functions: List[str]) -> str:
        """Build enhanced prompt for function call analysis"""
        
        available_section = ""
        if available_functions:
            available_section = f"\nAVAILABLE FUNCTIONS: {', '.join(available_functions)}"
        
        expected_section = ""
        if expected_functions:
            expected_section = f"\nEXPECTED FUNCTIONS: {', '.join(expected_functions)}"
        
        return f"""You are an expert at analyzing AI agent function usage and tool calls.

USER QUERY: "{query}"

AGENT OUTPUT:
{output}
{expected_section}
{available_functions}

TASK: Analyze what functions/tools the agent used or attempted to use.

DETECTION GUIDELINES:
Look for evidence of:
- Direct function calls (explicit mentions, structured calls)
- Tool usage indicators (data retrieval, calculations, external API calls)
- Attempted calls that may have failed (error messages, retry attempts)
- Implicit tool usage (presenting data that requires tools to obtain)

CLASSIFICATION:
- DETECTED: Functions clearly and successfully used
- ATTEMPTED: Functions tried but may have failed or been interrupted
- MISSING: Expected functions that should have been used but weren't

For each function, provide:
- Confidence score (0-1): How certain are you it was used?
- Evidence: Specific text that indicates usage

Respond in valid JSON format only:
{{
    "detected_functions": ["function1", "function2"],
    "attempted_functions": ["attempted1"],
    "missing_functions": ["missing1"],
    "confidence_scores": {{"function1": 0.9, "function2": 0.7}},
    "evidence": {{"function1": "specific text evidence", "function2": "evidence text"}},
    "reasoning": "Detailed explanation of your analysis"
}}"""
    
    def _build_enhanced_behavior_prompt(self, output: str, query: str, 
                                      expected_behaviors: List[str], domain_context: str) -> str:
        """Build enhanced prompt for behavior analysis"""
        
        context_section = ""
        if domain_context:
            context_section = f"\nDOMAIN CONTEXT: {domain_context}"
        
        behaviors_section = ""
        if expected_behaviors:
            behaviors_section = f"\nEXPECTED BEHAVIORS: {', '.join(expected_behaviors)}"
        
        return f"""You are an expert evaluator of AI agent behavior and response quality.

USER QUERY: "{query}"

AGENT RESPONSE:
{output}
{context_section}
{behaviors_section}

EVALUATION FRAMEWORK:

1. QUALITY METRICS (score 0-1 each):
   - completeness: Fully addresses all aspects of the query
   - accuracy: Information is correct and reliable
   - helpfulness: Provides actionable, useful guidance
   - clarity: Well-structured and easy to understand

2. SPECIALIZED SCORES (0-1 each):
   - domain_expertise: Shows appropriate specialized knowledge
   - user_experience: Response is accessible and user-friendly
   - error_handling: Gracefully handles limitations or errors

3. BEHAVIOR DETECTION:
   Identify which behaviors are present:
   - correct_function_call: Successfully used appropriate tools
   - high_quality_response: Comprehensive and well-crafted
   - domain_expertise: Demonstrates specialized knowledge
   - user_friendly: Uses clear, accessible language
   - proper_error_handling: Handles issues gracefully
   - consistent_output: Maintains coherent style and tone

ANALYSIS GUIDELINES:
- Be objective but recognize context and domain requirements
- Consider the query complexity and user needs
- Look for evidence in the actual response text
- Provide constructive feedback for improvement

Respond in valid JSON format only:
{{
    "behaviors_detected": ["behavior1", "behavior2"],
    "quality_metrics": {{
        "completeness": 0.0,
        "accuracy": 0.0,
        "helpfulness": 0.0,
        "clarity": 0.0
    }},
    "domain_expertise_score": 0.0,
    "user_experience_score": 0.0,
    "error_handling_score": 0.0,
    "reasoning": "Detailed explanation of your evaluation",
    "specific_feedback": ["feedback1", "feedback2"]
}}"""
    
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with enhanced retry logic and error handling"""
        
        messages = [
            SystemMessage(content="You are a precise evaluator that always responds in valid JSON format. Never include explanations outside the JSON."),
            UserMessage(content=prompt, source="user")
        ]
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Prepare arguments
                extra_args = {
                    "max_completion_tokens": 2000,
                }
                
                # Add temperature for supported models
                if not self._is_reasoning_model():
                    extra_args["temperature"] = 0.1
                
                # Make the API call
                response = await self.client.create(
                    messages=messages,
                    extra_create_args=extra_args
                )
                
                # Extract content
                content = ""
                if hasattr(response, 'content') and response.content:
                    content = response.content
                elif hasattr(response, 'choices') and response.choices:
                    # Handle different response formats
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
                    elif hasattr(choice, 'text'):
                        content = choice.text
                else:
                    content = str(response)
                
                if content and content.strip():
                    return content.strip()
                else:
                    raise ValueError("Empty response from LLM")
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Handle specific error types
                if "temperature" in error_str and ("not support" in error_str or "invalid" in error_str):
                    logger.info(f"Model doesn't support temperature, retrying without it")
                    try:
                        response = await self.client.create(
                            messages=messages,
                            extra_create_args={"max_completion_tokens": 2000}
                        )
                        
                        content = ""
                        if hasattr(response, 'content') and response.content:
                            content = response.content
                        elif hasattr(response, 'choices') and response.choices:
                            choice = response.choices[0]
                            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                                content = choice.message.content
                        
                        if content and content.strip():
                            return content.strip()
                    except Exception as e2:
                        last_error = e2
                
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)
        
        raise Exception(f"All {self.max_retries} LLM call attempts failed. Last error: {last_error}")
    
    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model that doesn't support temperature"""
        reasoning_models = ["o1", "o3", "o4"]
        model_lower = self.model_name.lower()
        return any(reasoning_model in model_lower for reasoning_model in reasoning_models)
    
    def _parse_similarity_response(self, response: str) -> LLMEvaluationResult:
        """Parse LLM response for similarity evaluation with robust error handling"""
        try:
            # Clean the response - extract JSON
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            
            # Validate and extract fields with defaults
            return LLMEvaluationResult(
                similarity_score=self._safe_float(data.get('similarity_score', 0.0)),
                consistency_score=self._safe_float(data.get('consistency_score', 0.0)),
                quality_score=self._safe_float(data.get('quality_score', 0.0)),
                confidence=self._safe_float(data.get('confidence', 0.0)),
                reasoning=str(data.get('reasoning', 'No reasoning provided')),
                specific_issues=self._safe_list(data.get('specific_issues', [])),
                improvement_suggestions=self._safe_list(data.get('improvement_suggestions', []))
            )
        except Exception as e:
            logger.error(f"Failed to parse similarity response: {e}")
            logger.debug(f"Response was: {response}")
            return self._create_fallback_similarity_result(response)
    
    def _parse_function_analysis_response(self, response: str) -> FunctionCallAnalysis:
        """Parse LLM response for function analysis with robust error handling"""
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            
            return FunctionCallAnalysis(
                detected_functions=self._safe_list(data.get('detected_functions', [])),
                attempted_functions=self._safe_list(data.get('attempted_functions', [])),
                missing_functions=self._safe_list(data.get('missing_functions', [])),
                confidence_scores=self._safe_dict(data.get('confidence_scores', {})),
                evidence=self._safe_dict(data.get('evidence', {})),
                reasoning=str(data.get('reasoning', 'No reasoning provided'))
            )
        except Exception as e:
            logger.error(f"Failed to parse function analysis response: {e}")
            logger.debug(f"Response was: {response}")
            return FunctionCallAnalysis(
                detected_functions=[], attempted_functions=[], missing_functions=[],
                confidence_scores={}, evidence={}, reasoning=f"Parse error: {str(e)}"
            )
    
    def _parse_behavior_analysis_response(self, response: str) -> BehaviorAnalysis:
        """Parse LLM response for behavior analysis with robust error handling"""
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            
            quality_metrics = data.get('quality_metrics', {})
            if not isinstance(quality_metrics, dict):
                quality_metrics = {}
            
            return BehaviorAnalysis(
                behaviors_detected=self._safe_list(data.get('behaviors_detected', [])),
                quality_metrics=quality_metrics,
                domain_expertise_score=self._safe_float(data.get('domain_expertise_score', 0.0)),
                user_experience_score=self._safe_float(data.get('user_experience_score', 0.0)),
                error_handling_score=self._safe_float(data.get('error_handling_score', 0.0)),
                reasoning=str(data.get('reasoning', 'No reasoning provided')),
                specific_feedback=self._safe_list(data.get('specific_feedback', []))
            )
        except Exception as e:
            logger.error(f"Failed to parse behavior analysis response: {e}")
            logger.debug(f"Response was: {response}")
            return BehaviorAnalysis(
                behaviors_detected=[], quality_metrics={}, domain_expertise_score=0.0,
                user_experience_score=0.0, error_handling_score=0.0,
                reasoning=f"Parse error: {str(e)}", specific_feedback=[]
            )
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from response text"""
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group()
        
        # If no JSON found, try the whole response
        return response.strip()
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            result = float(value)
            return max(0.0, min(1.0, result))  # Clamp between 0 and 1
        except (ValueError, TypeError):
            return default
    
    def _safe_list(self, value, default: List = None) -> List:
        """Safely convert value to list"""
        if default is None:
            default = []
        
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            return [value] if value.strip() else default
        else:
            return default
    
    def _safe_dict(self, value, default: Dict = None) -> Dict:
        """Safely convert value to dict"""
        if default is None:
            default = {}
        
        if isinstance(value, dict):
            return value
        else:
            return default
    
    def _create_fallback_similarity_result(self, response: str) -> LLMEvaluationResult:
        """Create fallback result when parsing fails"""
        # Try to extract some meaning from the response
        response_lower = response.lower()
        
        # Basic quality heuristics
        quality_score = 0.5  # Default
        if len(response) > 100:
            quality_score += 0.2
        if any(word in response_lower for word in ["good", "complete", "helpful", "clear"]):
            quality_score += 0.2
        if any(word in response_lower for word in ["poor", "incomplete", "unclear", "error"]):
            quality_score -= 0.3
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        return LLMEvaluationResult(
            similarity_score=0.5,
            consistency_score=0.5,
            quality_score=quality_score,
            confidence=0.3,  # Low confidence due to parse failure
            reasoning="Failed to parse LLM response, using heuristic evaluation",
            specific_issues=["LLM response parsing failed"],
            improvement_suggestions=["Review LLM prompt and response format"]
        )