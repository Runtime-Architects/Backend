"""
Core LLM-as-Judge implementation for agent evaluation
Could work with any agent type and integrates with existing evaluation frameworks
Fixed for Azure OpenAI API compatibility
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Azure OpenAI imports
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage
import os

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
    """Behaviour analysis using LLM"""
    behaviors_detected: List[str]
    quality_metrics: Dict[str, float]
    domain_expertise_score: float
    user_experience_score: float
    error_handling_score: float
    reasoning: str
    specific_feedback: List[str]

class LLMJudge:
    """Core LLM-as-Judge implementation"""
    
    def __init__(self, azure_client: AzureOpenAIChatCompletionClient, 
                 model_name: str = "gpt-4", max_retries: int = 3):
        self.client = azure_client
        self.model_name = model_name
        self.max_retries = max_retries
        
    async def evaluate_semantic_similarity(self, outputs: List[str], 
                                         expected_keywords: List[str] = None,
                                         domain_context: str = "") -> LLMEvaluationResult:
        """Evaluate semantic similarity across multiple outputs"""
        
        prompt = self._build_similarity_prompt(outputs, expected_keywords, domain_context)
        
        start_time = time.time()
        try:
            response = await self._call_llm_with_retry(prompt)
            result = self._parse_similarity_response(response)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"LLM similarity evaluation failed: {e}")
            return LLMEvaluationResult(
                similarity_score=0.0,
                consistency_score=0.0,
                quality_score=0.0,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                specific_issues=[],
                improvement_suggestions=[],
                execution_time=time.time() - start_time
            )
    
    async def analyze_function_calls(self, output: str, expected_functions: List[str],
                                   query: str, available_functions: List[str] = None) -> FunctionCallAnalysis:
        """Analyze function calls in agent output"""
        
        prompt = self._build_function_analysis_prompt(output, expected_functions, query, available_functions)
        
        try:
            response = await self._call_llm_with_retry(prompt)
            return self._parse_function_analysis_response(response)
        except Exception as e:
            logger.error(f"LLM function analysis failed: {e}")
            return FunctionCallAnalysis(
                detected_functions=[],
                attempted_functions=[],
                missing_functions=expected_functions,
                confidence_scores={},
                evidence={},
                reasoning=f"Analysis failed: {str(e)}"
            )
    
    async def analyze_behavior(self, output: str, query: str, 
                             expected_behaviors: List[str],
                             domain_context: str = "") -> BehaviorAnalysis:
        """Analyze agent behavior and response quality"""
        
        prompt = self._build_behavior_analysis_prompt(output, query, expected_behaviors, domain_context)
        
        try:
            response = await self._call_llm_with_retry(prompt)
            return self._parse_behavior_analysis_response(response)
        except Exception as e:
            logger.error(f"LLM behavior analysis failed: {e}")
            return BehaviorAnalysis(
                behaviors_detected=[],
                quality_metrics={},
                domain_expertise_score=0.0,
                user_experience_score=0.0,
                error_handling_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                specific_feedback=[]
            )
    
    def _build_similarity_prompt(self, outputs: List[str], 
                               expected_keywords: List[str],
                               domain_context: str) -> str:
        """Build prompt for semantic similarity evaluation"""
        
        keywords_section = ""
        if expected_keywords:
            keywords_section = f"\nEXPECTED KEYWORDS: {', '.join(expected_keywords)}"
        
        context_section = ""
        if domain_context:
            context_section = f"\nDOMAIN CONTEXT: {domain_context}"
        
        return f"""You are an expert evaluator assessing the semantic consistency of AI agent outputs.
{context_section}

TASK: Evaluate the semantic similarity and consistency of these agent outputs from multiple test runs:

OUTPUTS TO COMPARE:
{chr(10).join([f"Output {i+1}: {output}" for i, output in enumerate(outputs)])}
{keywords_section}

EVALUATION CRITERIA:
1. SEMANTIC SIMILARITY (0-1): Do outputs convey the same core information?
2. FACTUAL CONSISTENCY (0-1): Are facts and data consistent across outputs?
3. RESPONSE QUALITY (0-1): Overall quality, completeness, and helpfulness
4. CONFIDENCE (0-1): How confident are you in this evaluation?

Focus on MEANING rather than exact wording. Different phrasing of same information should score highly.

Respond in valid JSON format:
{{
    "similarity_score": 0.0,
    "consistency_score": 0.0,
    "quality_score": 0.0,
    "confidence": 0.0,
    "reasoning": "Explanation of evaluation",
    "specific_issues": ["issue1", "issue2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""
    
    def _build_function_analysis_prompt(self, output: str, expected_functions: List[str], 
                                      query: str, available_functions: List[str]) -> str:
        """Build prompt for function call analysis"""
        
        available_section = ""
        if available_functions:
            available_section = f"\nAVAILABLE FUNCTIONS: {', '.join(available_functions)}"
        
        return f"""You are an expert at analyzing AI agent behavior and function usage.

USER QUERY: {query}
AGENT OUTPUT: {output}
EXPECTED FUNCTIONS: {', '.join(expected_functions) if expected_functions else "None specified"}
{available_section}

TASK: Analyze function usage in the agent's response.

DETECTION CRITERIA:
- DETECTED: Functions clearly used (explicit calls, data that requires tools)
- ATTEMPTED: Functions tried but may have failed (error messages, references to attempts)
- MISSING: Expected functions that should have been used but weren't
- EVIDENCE: Specific text showing function usage or absence

Look for:
- Direct function calls or tool usage
- References to data retrieval or analysis
- Error messages about function calls
- Implicit function usage (getting data without explicit calls)

Respond in valid JSON format:
{{
    "detected_functions": ["function1", "function2"],
    "attempted_functions": ["attempted1"],
    "missing_functions": ["missing1"],
    "confidence_scores": {{"function1": 0.9, "function2": 0.7}},
    "evidence": {{"function1": "specific text evidence", "function2": "evidence text"}},
    "reasoning": "Detailed explanation of analysis"
}}"""
    
    def _build_behavior_analysis_prompt(self, output: str, query: str, 
                                      expected_behaviors: List[str], domain_context: str) -> str:
        """Build prompt for behaviour analysis"""
        
        context_section = ""
        if domain_context:
            context_section = f"\nDOMAIN CONTEXT: {domain_context}"
        
        return f"""You are an expert evaluator of AI agent behavior and response quality.

USER QUERY: {query}
AGENT RESPONSE: {output}
EXPECTED BEHAVIORS: {', '.join(expected_behaviors)}
{context_section}

EVALUATION AREAS:
1. RESPONSE QUALITY METRICS (0-1 each):
   - Completeness: Addresses all aspects of query
   - Accuracy: Information is correct and reliable
   - Helpfulness: Provides actionable, useful information
   - Clarity: Easy to understand and well-organized

2. DOMAIN EXPERTISE (0-1): Shows specialized knowledge in the domain
3. USER EXPERIENCE (0-1): Response is accessible and user-friendly
4. ERROR HANDLING (0-1): Graceful handling of errors or limitations

BEHAVIOR DETECTION:
- correct_function_call: Used appropriate tools effectively
- high_quality_response: Comprehensive and helpful
- domain_expertise: Shows specialized knowledge
- user_friendly: Clear, accessible language
- proper_error_handling: Handles issues gracefully
- consistent_output: Maintains consistent style

Respond in valid JSON format:
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
    "reasoning": "Detailed explanation of evaluation",
    "specific_feedback": ["feedback1", "feedback2"]
}}"""
    
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with retry logic - Fixed for Azure OpenAI API and model compatibility"""
        
        messages = [
            SystemMessage(content="You are a precise evaluator that always responds in valid JSON format."),
            UserMessage(content=prompt, source="user")
        ]
        
        for attempt in range(self.max_retries):
            try:
                # Try with temperature first (for models that support it)
                extra_args = {"max_completion_tokens": 2000}
                
                # Add temperature for models that support it
                if not self._is_reasoning_model():
                    extra_args["temperature"] = 0.1
                
                response = await self.client.create(
                    messages=messages,
                    extra_create_args=extra_args
                )
                
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # If temperature is not supported, retry without it
                if "temperature" in error_str and "not support" in error_str:
                    logger.info(f"Temperature not supported by model {self.model_name}, retrying without temperature")
                    try:
                        response = await self.client.create(
                            messages=messages,
                            extra_create_args={"max_completion_tokens": 2000}
                        )
                        
                        if hasattr(response, 'content'):
                            return response.content
                        else:
                            return str(response)
                    except Exception as e2:
                        logger.warning(f"LLM call attempt {attempt + 1} failed even without temperature: {e2}")
                else:
                    logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        raise Exception("All LLM call attempts failed")
    
    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model that doesn't support temperature"""
        reasoning_models = ["o1", "o3", "o4"]  # I think these are the models that typically don't support temperature (could change)
        model_lower = self.model_name.lower()
        return any(reasoning_model in model_lower for reasoning_model in reasoning_models)
    
    def _parse_similarity_response(self, response: str) -> LLMEvaluationResult:
        """Parse LLM response for similarity evaluation"""
        try:
            # Handle both direct JSON and JSON embedded in text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return LLMEvaluationResult(
                similarity_score=float(data.get('similarity_score', 0.0)),
                consistency_score=float(data.get('consistency_score', 0.0)),
                quality_score=float(data.get('quality_score', 0.0)),
                confidence=float(data.get('confidence', 0.0)),
                reasoning=data.get('reasoning', ''),
                specific_issues=data.get('specific_issues', []),
                improvement_suggestions=data.get('improvement_suggestions', [])
            )
        except Exception as e:
            logger.error(f"Failed to parse similarity response: {e}")
            logger.debug(f"Response was: {response}")
            return LLMEvaluationResult(
                similarity_score=0.0,
                consistency_score=0.0,
                quality_score=0.0,
                confidence=0.0,
                reasoning=f"Parse error: {str(e)}",
                specific_issues=[],
                improvement_suggestions=[]
            )
    
    def _parse_function_analysis_response(self, response: str) -> FunctionCallAnalysis:
        """Parse LLM response for function analysis"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return FunctionCallAnalysis(
                detected_functions=data.get('detected_functions', []),
                attempted_functions=data.get('attempted_functions', []),
                missing_functions=data.get('missing_functions', []),
                confidence_scores=data.get('confidence_scores', {}),
                evidence=data.get('evidence', {}),
                reasoning=data.get('reasoning', '')
            )
        except Exception as e:
            logger.error(f"Failed to parse function analysis response: {e}")
            logger.debug(f"Response was: {response}")
            return FunctionCallAnalysis(
                detected_functions=[],
                attempted_functions=[],
                missing_functions=[],
                confidence_scores={},
                evidence={},
                reasoning=f"Parse error: {str(e)}"
            )
    
    def _parse_behavior_analysis_response(self, response: str) -> BehaviorAnalysis:
        """Parse LLM response for behaviour analysis"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return BehaviorAnalysis(
                behaviors_detected=data.get('behaviors_detected', []),
                quality_metrics=data.get('quality_metrics', {}),
                domain_expertise_score=float(data.get('domain_expertise_score', 0.0)),
                user_experience_score=float(data.get('user_experience_score', 0.0)),
                error_handling_score=float(data.get('error_handling_score', 0.0)),
                reasoning=data.get('reasoning', ''),
                specific_feedback=data.get('specific_feedback', [])
            )
        except Exception as e:
            logger.error(f"Failed to parse behavior analysis response: {e}")
            logger.debug(f"Response was: {response}")
            return BehaviorAnalysis(
                behaviors_detected=[],
                quality_metrics={},
                domain_expertise_score=0.0,
                user_experience_score=0.0,
                error_handling_score=0.0,
                reasoning=f"Parse error: {str(e)}",
                specific_feedback=[]
            )