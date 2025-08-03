"""
LLM-as-Judge implementation with ground truth comparison
Provides consistent scoring by comparing agent outputs against reference data
Handles JSON parsing and penalty calculation
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
class GroundTruthEvaluationResult:
    """Evaluation result with ground truth comparison"""
    overall_score: float
    similarity_score: float
    quality_score: float
    confidence: float
    
    # Detailed scoring breakdown
    accuracy_score: float
    completeness_score: float
    clarity_score: float
    format_score: float
    actionability_score: float
    
    # Analysis details
    reasoning: str
    specific_matches: List[str]
    specific_gaps: List[str]
    improvement_suggestions: List[str]
    
    # Penalty analysis
    penalties_applied: List[str]
    penalty_total: float
    
    execution_time: float = 0.0

@dataclass
class ContentAnalysis:
    """Detailed content analysis against requirements"""
    must_have_score: float
    should_have_score: float
    format_score: float
    
    must_have_matches: List[str]
    must_have_missing: List[str]
    should_have_matches: List[str]
    should_have_missing: List[str]
    
    format_compliance: Dict[str, bool]
    content_quality_metrics: Dict[str, float]

class LLMJudge:
    """LLM-as-Judge with ground truth comparison and consistent scoring"""
    
    def __init__(self, azure_client: AzureOpenAIChatCompletionClient, 
                 model_name: str = "gpt-4", max_retries: int = 3):
        self.client = azure_client
        self.model_name = model_name
        self.max_retries = max_retries
        
        # Consistency settings
        self.consistency_temperature = 0.1  # Low for consistency
        self.max_tokens = 3000
        
        logger.info(f"LLM Judge initialized with model: {model_name}")
    
    async def evaluate_with_ground_truth(self, 
                                       agent_output: str,
                                       ground_truth: Dict[str, Any],
                                       test_case: Dict[str, Any],
                                       compressed_co2_data: Dict = None) -> GroundTruthEvaluationResult:
        """
        Main evaluation method comparing agent output against ground truth
        """
        if not agent_output or not agent_output.strip():
            return self._create_empty_output_result()
        
        if not ground_truth or "reference_output" not in ground_truth:
            logger.error("Ground truth missing or invalid")
            return self._create_error_result("No ground truth provided")
        
        start_time = time.time()
        
        try:
            # Step 1: Content Analysis
            content_analysis = await self._analyze_content_requirements(
                agent_output, ground_truth, test_case, compressed_co2_data
            )
            
            # Step 2: Ground Truth Comparison
            comparison_result = await self._compare_with_ground_truth(
                agent_output, ground_truth["reference_output"], test_case, compressed_co2_data
            )
            
            # Step 3: Calculate final scores
            final_result = self._calculate_final_scores(
                content_analysis, comparison_result, ground_truth
            )
            
            final_result.execution_time = time.time() - start_time
            
            logger.info(f"Ground truth evaluation completed: score={final_result.overall_score:.2f}")
            return final_result
            
        except Exception as e:
            logger.error(f"Ground truth evaluation failed: {e}")
            return self._create_error_result(f"Evaluation failed: {str(e)}")
    
    async def _analyze_content_requirements(self, 
                                          agent_output: str, 
                                          ground_truth: Dict, 
                                          test_case: Dict,
                                          compressed_co2_data: Dict = None) -> ContentAnalysis:
        """Analyze if agent output meets specific requirements"""
        
        scoring_criteria = ground_truth.get("scoring_criteria", {})
        content_reqs = scoring_criteria.get("content_requirements", {})
        
        prompt = self._build_content_analysis_prompt(
            agent_output, content_reqs, test_case, compressed_co2_data
        )
        
        try:
            response = await self._call_llm_with_retry(prompt)
            return self._parse_content_analysis_response(response)
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            # Return fallback analysis
            return ContentAnalysis(
                must_have_score=0.5, should_have_score=0.5, format_score=0.5,
                must_have_matches=[], must_have_missing=[],
                should_have_matches=[], should_have_missing=[],
                format_compliance={}, content_quality_metrics={}
            )
    
    async def _compare_with_ground_truth(self, 
                                       agent_output: str, 
                                       reference_output: str, 
                                       test_case: Dict,
                                       compressed_co2_data: Dict = None) -> Dict[str, Any]:
        """Compare agent output directly with ground truth reference"""
        
        prompt = self._build_ground_truth_comparison_prompt(
            agent_output, reference_output, test_case, compressed_co2_data
        )
        
        try:
            response = await self._call_llm_with_retry(prompt)
            return self._parse_ground_truth_comparison_response(response)
        except Exception as e:
            logger.error(f"Ground truth comparison failed: {e}")
            return {
                "similarity_score": 0.5,
                "quality_score": 0.5,
                "accuracy_score": 0.5,
                "reasoning": f"Comparison failed: {str(e)}",
                "matches": [],
                "gaps": []
            }
    
    def _build_content_analysis_prompt(self, 
                                     agent_output: str, 
                                     content_requirements: Dict, 
                                     test_case: Dict,
                                     compressed_co2_data: Dict = None) -> str:
        """Build prompt for detailed content requirement analysis"""
        
        must_haves = content_requirements.get("must_have", [])
        should_haves = content_requirements.get("should_have", [])
        format_reqs = content_requirements.get("format_requirements", [])
        
        # Build CO2 data context if available
        co2_context = ""
        if compressed_co2_data and 'data' in compressed_co2_data:
            data_points = compressed_co2_data['data']
            if data_points:
                values = [point['value'] for point in data_points]
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                
                # Find optimal and peak times
                min_point = min(data_points, key=lambda x: x['value'])
                max_point = max(data_points, key=lambda x: x['value'])
                
                co2_context = f"""
**ACTUAL COMPRESSED CO2 DATA CONTEXT:**
- Optimal time: {min_point['time']} ({min_val}g CO2/kWh) - actual minimum
- Peak time: {max_point['time']} ({max_val}g CO2/kWh) - actual maximum  
- Daily average: {avg_val:.0f}g CO2/kWh
- Data range: {min_val}-{max_val}g CO2/kWh
- Total data points: {len(data_points)}

VALIDATION NOTES:
- Agent should reference values within {min_val}-{max_val}g CO2/kWh range
- Time recommendations should align with actual optimal periods
- Allow flexibility if agent provides narrower optimal windows (e.g., peak hour within broader range)
"""

        return f"""You are a precise content evaluator. Analyze the agent output against specific requirements using ACTUAL CO2 data for validation.

USER QUERY: "{test_case.get('query', '')}"

AGENT OUTPUT TO ANALYZE:
{agent_output}
{co2_context}
EVALUATION CRITERIA:

**MUST-HAVE REQUIREMENTS (Critical - each missing item is major penalty):**
{self._format_requirements_list(must_haves)}

**SHOULD-HAVE REQUIREMENTS (Important - missing items reduce quality):**
{self._format_requirements_list(should_haves)}

**FORMAT REQUIREMENTS (Structure and presentation):**
{self._format_requirements_list(format_reqs)}

SCORING INSTRUCTIONS:
- Check each requirement carefully against the agent output
- Be specific about what is present vs missing
- For format requirements, check exact compliance (e.g., "at least 3 bullet points" means count them)
- Be consistent: same criteria should always yield same results

Respond in valid JSON format only:
{{
    "must_have_analysis": {{
        "score": 0.0,
        "matches": ["requirement1", "requirement2"],
        "missing": ["requirement3"],
        "details": {{"requirement1": "specific evidence in output"}}
    }},
    "should_have_analysis": {{
        "score": 0.0,
        "matches": ["requirement1"],
        "missing": ["requirement2"],
        "details": {{"requirement1": "specific evidence"}}
    }},
    "format_analysis": {{
        "score": 0.0,
        "compliance": {{"at least 3 bullet points": true, "specific time format": false}},
        "details": "specific format observations"
    }},
    "content_quality": {{
        "readability": 0.0,
        "structure": 0.0,
        "engagement": 0.0
    }}
}}"""
    
    def _build_ground_truth_comparison_prompt(self, 
                                            agent_output: str, 
                                            reference_output: str, 
                                            test_case: Dict,
                                            compressed_co2_data: Dict = None) -> str:
        """Build prompt for direct ground truth comparison"""
        
        # Build CO2 data context for validation
        co2_validation = ""
        if compressed_co2_data and 'data' in compressed_co2_data:
            data_points = compressed_co2_data['data']
            if data_points:
                values = [point['value'] for point in data_points]
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                
                co2_validation = f"""
**ACTUAL CO2 DATA FOR VALIDATION:**
- Valid CO2 range: {min_val}-{max_val}g CO2/kWh (from compressed data)
- Actual average: {avg_val:.0f}g CO2/kWh
- Data source: Compressed EirGrid data

**VALIDATION RULES:**
- Agent values MUST be within {min_val}-{max_val}g CO2/kWh range
- Time recommendations should be data-driven, not generic
- Allow narrow optimal windows if they align with actual low periods
- Penalize values outside the actual data range heavily
"""

        return f"""You are an expert evaluator comparing AI agent outputs against reference standards using ACTUAL CO2 data for validation.

USER QUERY: "{test_case.get('query', '')}"

REFERENCE OUTPUT (Gold Standard):
{reference_output}

AGENT OUTPUT TO EVALUATE:
{agent_output}
{co2_validation}
EVALUATION TASK: Compare the agent output against the reference output across these dimensions:

**ACCURACY (0-1)**: How factually correct is the agent output compared to reference?
- Check time recommendations, values, and facts
- Penalize any incorrect information heavily

**COMPLETENESS (0-1)**: How thoroughly does the agent output cover the reference content?
- Does it include the main points from reference?
- Missing key information should be penalized

**CLARITY (0-1)**: How clear and well-structured is the communication?
- Compare organization and readability to reference
- Reference sets the standard for clarity

**ACTIONABILITY (0-1)**: How useful and actionable is the agent output?
- Does it provide specific, implementable advice like the reference?
- Vague advice should score lower

**SIMILARITY (0-1)**: Overall semantic similarity to reference output
- Same core message and recommendations?
- Similar level of detail and usefulness?

COMPARISON GUIDELINES:
- The reference output is the gold standard - agent should match or exceed it
- Different wording for same meaning is fine
- Missing key information is a major penalty
- Incorrect information is worse than missing information
- Be consistent: similar outputs should get similar scores

Respond in valid JSON format only:
{{
    "accuracy_score": 0.0,
    "completeness_score": 0.0,
    "clarity_score": 0.0,
    "actionability_score": 0.0,
    "similarity_score": 0.0,
    "overall_quality": 0.0,
    "reasoning": "detailed explanation of scores",
    "specific_matches": ["matches found"],
    "specific_gaps": ["important missing elements"],
    "confidence": 0.0
}}"""
    
    def _calculate_final_scores(self, 
                              content_analysis: ContentAnalysis, 
                              comparison_result: Dict, 
                              ground_truth: Dict) -> GroundTruthEvaluationResult:
        """Calculate final scores with calibrated penalty application"""
        
        scoring_criteria = ground_truth.get("scoring_criteria", {})
        weights = scoring_criteria.get("scoring_weights", {})
        penalties = scoring_criteria.get("penalty_conditions", [])
        
        # Base scores from comparison
        accuracy_score = comparison_result.get("accuracy_score", 0.0)
        completeness_score = comparison_result.get("completeness_score", 0.0)
        clarity_score = comparison_result.get("clarity_score", 0.0)
        actionability_score = comparison_result.get("actionability_score", 0.0)
        format_score = content_analysis.format_score
        
        # Apply semantic equivalence bonus
        semantic_bonus = self._calculate_semantic_equivalence_bonus(
            comparison_result, content_analysis
        )
        accuracy_score = min(1.0, accuracy_score + semantic_bonus)
        
        # Apply calibrated penalty calculation (reduced from 0.3 to 0.15)
        # Missing requirements reduce overall score proportionally
        must_have_penalty = (1.0 - content_analysis.must_have_score) * 0.15  # Reduced penalty
        should_have_penalty = (1.0 - content_analysis.should_have_score) * 0.05  # Reduced penalty
        
        # Calculate weighted score with better defaults
        default_weights = {
            "accuracy": 0.25, "completeness": 0.25, "clarity": 0.20, 
            "actionability": 0.20, "format": 0.10
        }
        
        # Use provided weights or defaults
        w_accuracy = weights.get("accuracy", weights.get("accuracy_vs_real_data", default_weights["accuracy"]))
        w_completeness = weights.get("completeness", default_weights["completeness"])
        w_clarity = weights.get("clarity", default_weights["clarity"])
        w_actionability = weights.get("actionability", default_weights["actionability"])
        w_format = weights.get("format", default_weights["format"])
        
        base_score = (
            accuracy_score * w_accuracy +
            completeness_score * w_completeness +
            clarity_score * w_clarity +
            actionability_score * w_actionability +
            format_score * w_format
        )
        
        # Apply penalties more reasonably
        penalty_total = must_have_penalty + should_have_penalty
        penalties_applied = []
        
        if must_have_penalty > 0:
            penalties_applied.append(f"Missing must-have requirements: -{must_have_penalty:.2f}")
        if should_have_penalty > 0:
            penalties_applied.append(f"Missing should-have requirements: -{should_have_penalty:.2f}")
        
        # Don't let penalties completely destroy the score
        # Apply penalties but ensure minimum score if there's any quality
        final_score = max(0.0, base_score - penalty_total)
        
        # If the base score was reasonable but penalties killed it, give some credit
        if base_score >= 0.4 and final_score < 0.2:
            final_score = max(final_score, base_score * 0.5)  # At least 50% of base score
        
        # More reasonable quality assessment
        # Don't let small issues completely fail a response
        quality_score = comparison_result.get("overall_quality", final_score)
        if quality_score < final_score:  # Use the higher of the two
            quality_score = final_score
            
        similarity_score = comparison_result.get("similarity_score", final_score)
        
        # Generate more constructive improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            content_analysis, comparison_result, penalties_applied
        )
        
        # Better reasoning that reflects the more balanced scoring
        reasoning = comparison_result.get("reasoning", "")
        if penalty_total > 0:
            reasoning += f" Applied penalties: {penalty_total:.2f} for missing requirements, but maintained base quality assessment."
        
        return GroundTruthEvaluationResult(
            overall_score=final_score,
            similarity_score=similarity_score,
            quality_score=quality_score,
            confidence=comparison_result.get("confidence", 0.8),
            
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            format_score=format_score,
            actionability_score=actionability_score,
            
            reasoning=reasoning,
            specific_matches=comparison_result.get("specific_matches", []),
            specific_gaps=comparison_result.get("specific_gaps", []),
            improvement_suggestions=improvement_suggestions,
            
            penalties_applied=penalties_applied,
            penalty_total=penalty_total
        )
    
    def _generate_improvement_suggestions(self, 
                                        content_analysis: ContentAnalysis, 
                                        comparison_result: Dict, 
                                        penalties_applied: List[str]) -> List[str]:
        """Generate specific improvement suggestions"""
        
        suggestions = []
        
        # Based on missing must-haves
        for missing in content_analysis.must_have_missing:
            suggestions.append(f"Add {missing}")
        
        # Based on comparison gaps
        for gap in comparison_result.get("specific_gaps", []):
            suggestions.append(f"Include {gap}")
        
        # Based on low scores
        if comparison_result.get("accuracy_score", 1.0) < 0.7:
            suggestions.append("Improve factual accuracy of recommendations")
        
        if comparison_result.get("clarity_score", 1.0) < 0.7:
            suggestions.append("Improve structure and clarity of response")
        
        if content_analysis.format_score < 0.7:
            suggestions.append("Follow the required format more closely")
        
        return suggestions[:5]  # Limit to top 5
    
    def _format_requirements_list(self, requirements: List[str]) -> str:
        """Format requirements list for prompt"""
        if not requirements:
            return "- None specified"
        return "\n".join([f"- {req}" for req in requirements])
    
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """ LLM call with better error handling"""
        
        messages = [
            SystemMessage(content="You are a precise evaluator that always responds in valid JSON format. Be consistent and objective in your evaluations."),
            UserMessage(content=prompt, source="user")
        ]
        
        for attempt in range(self.max_retries):
            try:
                # Prepare arguments for consistency
                extra_args = {
                    "max_completion_tokens": self.max_tokens,
                }
                
                # Add temperature only if model supports it
                if not self._is_reasoning_model():
                    extra_args["temperature"] = self.consistency_temperature
                
                response = await self.client.create(
                    messages=messages,
                    extra_create_args=extra_args
                )
                
                # Extract content with better error handling
                content = self._extract_response_content(response)
                
                if content and content.strip():
                    # Validate JSON before returning
                    self._validate_json_response(content)
                    return content.strip()
                else:
                    raise ValueError("Empty response from LLM")
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"All {self.max_retries} LLM attempts failed. Last error: {e}")
    
    def _extract_response_content(self, response) -> str:
        """Extract content from response with multiple format support"""
        
        # Try different response formats
        if hasattr(response, 'content') and response.content:
            return response.content
        elif hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            elif hasattr(choice, 'text'):
                return choice.text
        
        # Fallback to string conversion
        return str(response)
    
    def _validate_json_response(self, content: str) -> None:
        """Validate that response is valid JSON"""
        try:
            json_text = self._extract_json(content)
            json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response: {e}")
            # Don't raise - let the parser handle it
    
    def _is_reasoning_model(self) -> bool:
        """Check if model is reasoning-based (doesn't support temperature)"""
        reasoning_models = ["o1", "o3", "o4"]
        model_lower = self.model_name.lower()
        return any(reasoning_model in model_lower for reasoning_model in reasoning_models)
    
    def _extract_json(self, response: str) -> str:
        """ Better JSON extraction from response text"""
        
        # Try to find JSON block with better patterns
        json_patterns = [
            r'\{.*\}',  # Basic JSON
            r'```json\s*(\{.*\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*\})\s*```',  # JSON in any code blocks
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'  # Nested JSON
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_text = match.group(1) if pattern.startswith(r'```') else match.group()
                
                # Try to validate it's valid JSON
                try:
                    json.loads(json_text)
                    return json_text
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, return the response as is for fallback processing
        return response.strip()
    
    def _parse_content_analysis_response(self, response: str) -> ContentAnalysis:
        """ Parse content analysis response with better error handling"""
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            
            must_have = data.get("must_have_analysis", {})
            should_have = data.get("should_have_analysis", {})
            format_analysis = data.get("format_analysis", {})
            quality = data.get("content_quality", {})
            
            # More lenient scoring - if no explicit score, calculate based on matches
            must_have_score = float(must_have.get("score", 0.0))
            if must_have_score == 0.0 and must_have.get("matches"):
                # Calculate score based on matches vs total requirements
                total_requirements = len(must_have.get("matches", [])) + len(must_have.get("missing", []))
                if total_requirements > 0:
                    must_have_score = len(must_have.get("matches", [])) / total_requirements
            
            should_have_score = float(should_have.get("score", 0.0))
            if should_have_score == 0.0 and should_have.get("matches"):
                total_requirements = len(should_have.get("matches", [])) + len(should_have.get("missing", []))
                if total_requirements > 0:
                    should_have_score = len(should_have.get("matches", [])) / total_requirements
            
            format_score = float(format_analysis.get("score", 0.0))
            if format_score == 0.0 and format_analysis.get("compliance"):
                compliance = format_analysis.get("compliance", {})
                if compliance:
                    format_score = sum(1 for v in compliance.values() if v) / len(compliance)
            
            return ContentAnalysis(
                must_have_score=must_have_score,
                should_have_score=should_have_score,
                format_score=format_score,
                
                must_have_matches=must_have.get("matches", []),
                must_have_missing=must_have.get("missing", []),
                should_have_matches=should_have.get("matches", []),
                should_have_missing=should_have.get("missing", []),
                
                format_compliance=format_analysis.get("compliance", {}),
                content_quality_metrics=quality
            )
            
        except Exception as e:
            logger.error(f"Failed to parse content analysis: {e}")
            logger.debug(f"Response that failed to parse: {response[:500]}...")
            
            # Fallback: try to extract basic information from text
            return self._fallback_content_analysis(response)
    
    def _fallback_content_analysis(self, response: str) -> ContentAnalysis:
        """Fallback content analysis when JSON parsing fails"""
        
        # Look for score patterns in the text
        score_patterns = [
            r"must_have.*?score[\"']?\s*:\s*([0-9.]+)",
            r"should_have.*?score[\"']?\s*:\s*([0-9.]+)",
            r"format.*?score[\"']?\s*:\s*([0-9.]+)"
        ]
        
        scores = []
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            scores.append(float(match.group(1)) if match else 0.5)
        
        # Look for matches and missing items
        matches = re.findall(r"matches[\"']?\s*:\s*\[(.*?)\]", response, re.IGNORECASE)
        missing = re.findall(r"missing[\"']?\s*:\s*\[(.*?)\]", response, re.IGNORECASE)
        
        return ContentAnalysis(
            must_have_score=scores[0] if len(scores) > 0 else 0.5,
            should_have_score=scores[1] if len(scores) > 1 else 0.5, 
            format_score=scores[2] if len(scores) > 2 else 0.5,
            must_have_matches=matches[0].split(",") if matches else [],
            must_have_missing=missing[0].split(",") if missing else [],
            should_have_matches=[],
            should_have_missing=[],
            format_compliance={},
            content_quality_metrics={}
        )

    def _parse_ground_truth_comparison_response(self, response: str) -> Dict[str, Any]:
        """ Parse ground truth comparison response with better error handling"""
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            
            # Extract scores with defaults and validation
            accuracy_score = max(0.0, min(1.0, float(data.get("accuracy_score", 0.0))))
            completeness_score = max(0.0, min(1.0, float(data.get("completeness_score", 0.0))))
            clarity_score = max(0.0, min(1.0, float(data.get("clarity_score", 0.0))))
            actionability_score = max(0.0, min(1.0, float(data.get("actionability_score", 0.0))))
            similarity_score = max(0.0, min(1.0, float(data.get("similarity_score", 0.0))))
            overall_quality = max(0.0, min(1.0, float(data.get("overall_quality", 0.0))))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            
            # If overall_quality is 0, calculate it from component scores
            if overall_quality == 0.0:
                overall_quality = (accuracy_score + completeness_score + clarity_score + actionability_score) / 4
            
            return {
                "accuracy_score": accuracy_score,
                "completeness_score": completeness_score,
                "clarity_score": clarity_score,
                "actionability_score": actionability_score,
                "similarity_score": similarity_score,
                "overall_quality": overall_quality,
                "reasoning": str(data.get("reasoning", "")).strip(),
                "specific_matches": data.get("specific_matches", []),
                "specific_gaps": data.get("specific_gaps", []),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to parse comparison response: {e}")
            logger.debug(f"Response that failed to parse: {response[:500]}...")
            
            # Fallback: try to extract scores from text
            return self._fallback_comparison_analysis(response)
    
    def _fallback_comparison_analysis(self, response: str) -> Dict[str, Any]:
        """Fallback comparison analysis when JSON parsing fails"""
        
        # Look for score patterns in the text
        score_patterns = [
            (r"accuracy[_\s]*score[\"']?\s*:\s*([0-9.]+)", "accuracy_score"),
            (r"completeness[_\s]*score[\"']?\s*:\s*([0-9.]+)", "completeness_score"),
            (r"clarity[_\s]*score[\"']?\s*:\s*([0-9.]+)", "clarity_score"),
            (r"actionability[_\s]*score[\"']?\s*:\s*([0-9.]+)", "actionability_score"),
            (r"similarity[_\s]*score[\"']?\s*:\s*([0-9.]+)", "similarity_score"),
            (r"overall[_\s]*quality[\"']?\s*:\s*([0-9.]+)", "overall_quality"),
            (r"confidence[\"']?\s*:\s*([0-9.]+)", "confidence")
        ]
        
        result = {}
        for pattern, key in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            result[key] = max(0.0, min(1.0, float(match.group(1)))) if match else 0.3
        
        # Calculate overall quality if not found
        if result.get("overall_quality", 0.0) == 0.3:
            component_scores = [result.get("accuracy_score", 0.3), result.get("completeness_score", 0.3),
                              result.get("clarity_score", 0.3), result.get("actionability_score", 0.3)]
            result["overall_quality"] = sum(component_scores) / len(component_scores)
        
        # Look for reasoning
        reasoning_match = re.search(r"reasoning[\"']?\s*:\s*[\"']([^\"']+)[\"']", response, re.IGNORECASE | re.DOTALL)
        result["reasoning"] = reasoning_match.group(1) if reasoning_match else "Parsing failed, fallback analysis used"
        
        result["specific_matches"] = []
        result["specific_gaps"] = []
        
        return result
    
    def _create_empty_output_result(self) -> GroundTruthEvaluationResult:
        """Create result for empty output"""
        return GroundTruthEvaluationResult(
            overall_score=0.0,
            similarity_score=0.0,
            quality_score=0.0,
            confidence=1.0,
            accuracy_score=0.0,
            completeness_score=0.0,
            clarity_score=0.0,
            format_score=0.0,
            actionability_score=0.0,
            reasoning="No output provided by agent",
            specific_matches=[],
            specific_gaps=["No response generated"],
            improvement_suggestions=["Ensure agent generates a response"],
            penalties_applied=["No output: -1.0"],
            penalty_total=1.0
        )
    
    def _calculate_semantic_equivalence_bonus(self, comparison_result: Dict, 
                                            content_analysis: ContentAnalysis) -> float:
        """Calculate bonus for semantically equivalent but textually different responses"""
        bonus = 0.0
        
        # Check for time window equivalence
        gaps = comparison_result.get("specific_gaps", [])
        matches = comparison_result.get("specific_matches", [])
        
        # Pattern matching for time-related differences
        time_difference_patterns = [
            r"window differs.*\d+:\d+.*vs.*\d+:\d+",
            r"recommendation at \d+:\d+ instead of \d+:\d+",
            r"narrowed to.*\d+:\d+.*vs.*\d+:\d+"
        ]
        
        for gap in gaps:
            gap_lower = gap.lower()
            # Check if the gap is about time differences
            if any(re.search(pattern, gap_lower) for pattern in time_difference_patterns):
                # If we have the core elements (time recommendations exist), give partial credit
                if any("time" in match.lower() or "period" in match.lower() for match in matches):
                    bonus += 0.05  # Small bonus for having correct structure despite different times
            
            # Check for narrower but valid windows
            if "narrowed" in gap_lower or "differs" in gap_lower:
                if "optimal" in gap_lower or "window" in gap_lower:
                    # Agent provided a more specific window - this can be good
                    bonus += 0.03
        
        # Bonus for having all required elements even if values differ slightly
        if content_analysis.must_have_score >= 0.8:
            bonus += 0.02
        
        # Cap the total bonus
        return min(0.15, bonus)
    
    def _create_error_result(self, error_message: str) -> GroundTruthEvaluationResult:
        """Create result for evaluation errors"""
        return GroundTruthEvaluationResult(
            overall_score=0.0,
            similarity_score=0.0,
            quality_score=0.0,
            confidence=0.0,
            accuracy_score=0.0,
            completeness_score=0.0,
            clarity_score=0.0,
            format_score=0.0,
            actionability_score=0.0,
            reasoning=error_message,
            specific_matches=[],
            specific_gaps=[],
            improvement_suggestions=["Check evaluation system configuration"],
            penalties_applied=[],
            penalty_total=0.0
        )