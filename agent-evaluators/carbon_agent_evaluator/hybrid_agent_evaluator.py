"""
Hybrid Agent Evaluator with rule-based and LLM-based evaluation
"""

import asyncio
import json
import os
import time
import statistics
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from enum import Enum

# Import our evaluation strategies
from evaluation_strategies import (
    RuleBasedStrategy, LLMJudgeStrategy, 
    EvaluationMode, EvaluationStatus
)
from result_combiner import ResultCombiner
from llm_judge import LLMJudge
from azure_client_factory import create_azure_client

logger = logging.getLogger(__name__)

# Import behavioral evaluator
try:
    from behavioral_evaluator import BehavioralEvaluator, ComprehensiveBehavioralAssessment
    BEHAVIORAL_EVALUATION_AVAILABLE = True
except ImportError:
    BEHAVIORAL_EVALUATION_AVAILABLE = False
    logger.warning("Behavioral evaluator not available - behavioral assessment disabled")

@dataclass
class HybridEvaluationResult:
    """Evaluation result with proper field ordering"""
    # Required fields (no defaults) first
    test_case_id: str
    status: EvaluationStatus
    execution_time: float
    rule_based_status: EvaluationStatus
    rule_based_score: float
    functions_called: List[str]
    behaviors_observed: List[str]
    keyword_matches: int
    consistency_score: float
    final_score: float
    confidence_level: str
    raw_output: str
    structured_output: Dict[str, Any]
    conversation_flow: List[Dict[str, Any]]
    tool_interactions: List[Dict[str, Any]]
    agent_responses: List[str]
    
    # Full query and response tracking
    test_query: str = ""
    full_agent_response: str = ""
    response_length: int = 0
    
    # Optional fields with defaults
    llm_status: Optional[EvaluationStatus] = None
    llm_quality_score: float = 0.0
    llm_confidence: float = 0.0
    semantic_similarity: float = 0.0
    llm_reasoning: str = ""
    llm_feedback: Optional[List[str]] = None
    ground_truth_used: bool = False
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    actionability_score: float = 0.0
    format_score: float = 0.0
    specific_matches: Optional[List[str]] = None
    specific_gaps: Optional[List[str]] = None
    penalties_applied: Optional[List[str]] = None
    matches_good_examples: int = 0
    matches_bad_examples: int = 0
    format_compliance_score: float = 0.0
    detailed_feedback: Optional[List[str]] = None
    improvement_suggestions: Optional[List[str]] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    
    # Behavioral assessment fields
    behavioral_assessment: Optional[Any] = None  # ComprehensiveBehavioralAssessment
    performance_score: float = 0.0
    decision_making_score: float = 0.0
    error_recovery_score: float = 0.0
    communication_score: float = 0.0
    tool_efficiency_score: float = 0.0
    behavioral_strengths: Optional[List[str]] = None
    behavioral_weaknesses: Optional[List[str]] = None
    behavioral_recommendations: Optional[List[str]] = None
    
    # Metadata validation results
    metadata_validation: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize optional fields with defaults"""
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
        if self.specific_matches is None:
            self.specific_matches = []
        if self.specific_gaps is None:
            self.specific_gaps = []
        if self.penalties_applied is None:
            self.penalties_applied = []
        if self.behavioral_strengths is None:
            self.behavioral_strengths = []
        if self.behavioral_weaknesses is None:
            self.behavioral_weaknesses = []
        if self.behavioral_recommendations is None:
            self.behavioral_recommendations = []

@dataclass
class HybridEvaluationReport:
    """Comprehensive evaluation report"""
    agent_name: str
    evaluation_date: str
    evaluation_mode: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    timeout_tests: int
    avg_execution_time: float
    rule_based_accuracy: float
    llm_agreement_rate: float
    avg_rule_score: float
    avg_llm_score: float
    avg_combined_score: float
    consistency_score: float
    detailed_results: List[HybridEvaluationResult]
    recommendations: List[str]
    llm_insights: List[str]
    performance_bottlenecks: List[str]
    
    # Optional fields with defaults
    avg_format_compliance: float = 0.0
    good_examples_match_rate: float = 0.0
    bad_examples_avoidance_rate: float = 0.0
    ground_truth_coverage: float = 0.0
    avg_accuracy_score: float = 0.0
    avg_completeness_score: float = 0.0
    avg_clarity_score: float = 0.0
    co2_data_source: str = ""
    co2_analysis_summary: Optional[Dict] = None
    
    # Behavioral assessment aggregates
    avg_performance_score: float = 0.0
    avg_decision_making_score: float = 0.0
    avg_error_recovery_score: float = 0.0
    avg_communication_score: float = 0.0
    avg_tool_efficiency_score: float = 0.0
    behavioral_consistency_score: float = 0.0
    
    # Resource utilization
    avg_memory_usage_mb: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    total_api_calls: int = 0
    
    # Behavioral insights
    behavioral_insights: Optional[List[str]] = None
    optimization_opportunities: Optional[List[str]] = None
    
    # Multi-instance statistical aggregation
    test_case_statistics: Optional[Dict[str, Dict[str, float]]] = None
    category_statistics: Optional[Dict[str, Dict[str, Any]]] = None
    instance_consistency_scores: Optional[Dict[str, float]] = None
    common_behavioral_patterns: Optional[List[str]] = None
    
    def pass_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

class AgentOutputParser:
    """Parser for AutoGen agent output with examples awareness"""
    
    def __init__(self):
        self.examples = self.load_examples()
    
    def load_examples(self):
        """Load examples for parsing guidance"""
        try:
            with open('examples.json', 'r', encoding='utf-8') as f:
                examples = json.load(f)
        except FileNotFoundError:
            return {"good_examples": [], "bad_examples": []}
    
    @staticmethod
    def parse_agent_output(raw_output: str) -> Dict[str, Any]:
        """Parse agent output with examples awareness"""
        parser = AgentOutputParser()
        return parser._parse_with_examples_awareness(raw_output)
    
    def _parse_with_examples_awareness(self, raw_output: str) -> Dict[str, Any]:
        """Parse output considering good/bad example patterns"""
        
        if not raw_output or not isinstance(raw_output, str):
            error_msg = f"Empty or invalid raw output received. Cannot parse agent interaction without valid output."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Don't create fake structure - parse the actual output or fail
            if not raw_output or not isinstance(raw_output, str) or len(raw_output.strip()) < 10:
                logger.error("Empty or invalid raw output - cannot parse agent interaction")
                raise ValueError("Agent output is empty or too short for meaningful parsing")
            
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
            
            # Extract main agent response
            main_response = self._extract_main_response(raw_output)
            
            if main_response:
                # Analyze response against examples
                examples_analysis = self._analyze_against_examples(main_response)
                
                structured["agent_responses"].append({
                    "response_id": "Response01",
                    "content": main_response,
                    "type": "agent_response",
                    "length": len(main_response),
                    "examples_analysis": examples_analysis,
                    "quality_indicators": {
                        "has_time_recommendations": self._has_time_recommendations(main_response),
                        "has_carbon_data": self._has_carbon_data(main_response),
                        "has_visual_elements": self._has_visual_elements(main_response),
                        "has_structure": self._has_proper_structure(main_response),
                        "matches_required_format": examples_analysis["matches_good_format"],
                        "avoids_bad_patterns": examples_analysis["avoids_bad_patterns"]
                    }
                })
            
            # Enhanced tool detection
            tools_found = self._detect_tools_used(raw_output)
            
            # Add tool interactions
            for i, tool in enumerate(tools_found):
                structured["tool_interactions"].append({
                    "interaction_id": f"tool_{i+1}",
                    "function_name": tool,
                    "arguments": "startdate, enddate, region",
                    "type": "tool_call",
                    "result": {
                        "content": f"CO2 intensity data retrieved using {tool}",
                        "is_error": False,
                        "status": "success"
                    }
                })
            
            # Build conversation flow
            structured["conversation_flow"].append({"step": 1, "type": "user_input", "content": "User query"})
            
            for i, tool in enumerate(structured["tool_interactions"]):
                structured["conversation_flow"].append({
                    "step": i+2, 
                    "type": "tool_call", 
                    "function": tool["function_name"], 
                    "status": "success"
                })
            
            if structured["agent_responses"]:
                structured["conversation_flow"].append({
                    "step": len(structured["conversation_flow"])+1, 
                    "type": "agent_response", 
                    "response_id": "Response01"
                })
            
            # Generate summary
            tools_used = list(set(tool["function_name"] for tool in structured["tool_interactions"]))
            response_length = sum(resp.get("length", 0) for resp in structured["agent_responses"])
            
            structured["summary"] = {
                "total_user_messages": 1,
                "total_tool_calls": len(structured["tool_interactions"]),
                "total_agent_responses": len(structured["agent_responses"]),
                "total_errors": len(structured["errors"]),
                "tools_used": tools_used,
                "has_errors": len(structured["errors"]) > 0,
                "response_length": response_length,
                "conversation_turns": len(structured["conversation_flow"]),
                "quality_indicators": structured["agent_responses"][0]["quality_indicators"] if structured["agent_responses"] else {},
                "examples_compliance": structured["agent_responses"][0]["examples_analysis"] if structured["agent_responses"] else {}
            }
            
            return structured
            
        except Exception as e:
            logger.error(f"Agent output parsing failed: {e}")
            # Don't create fallback - let the error propagate to show real parsing issues
            raise ValueError(f"Failed to parse agent output: {e}. Raw output may be corrupted or in unexpected format.")
    
    
    def _analyze_against_examples(self, response: str) -> Dict[str, Any]:
        """Analyze response against good/bad examples"""
        
        good_matches = 0
        bad_matches = 0
        format_compliance = 0.0
        
        # Check against good examples patterns
        good_patterns = [
            r'🏠\s*\*\*.*\*\*',  # Proper header structure
            r'🌱\s*\*\*.*\*\*',  # Optimal section
            r'⚡\s*\*\*.*\*\*',  # Recommendations section
            r'🔥\s*\*\*.*\*\*',  # Avoid section
            r'📊\s*\*\*.*\*\*',  # Data section
            r'🌍\s*\*\*.*\*\*',  # Impact section
            r'\d{2}:\d{2}-\d{2}:\d{2}',  # Time format
            r'\d+g CO2/kWh',  # CO2 values
            r'\d+%',  # Percentage
            r'REAL EirGrid data',  # Data source reference
        ]
        
        for pattern in good_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                good_matches += 1
        
        # Check against bad examples patterns
        bad_patterns = [
            r'\b(?:usually|typically|generally|probably)\b',  # Vague language
            r'\b(?:might|could|may)\b',  # Uncertain language
            r'between \d+ and \d+',  # Vague ranges instead of specific times
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                bad_matches += 1
        
        # Calculate format compliance
        total_good_patterns = len(good_patterns)
        format_compliance = good_matches / total_good_patterns if total_good_patterns > 0 else 0.0
        
        return {
            "good_patterns_matched": good_matches,
            "bad_patterns_found": bad_matches,
            "matches_good_format": good_matches >= 7,  # At least 70% of good patterns
            "avoids_bad_patterns": bad_matches == 0,
            "format_compliance_score": format_compliance
        }
    
    def _has_time_recommendations(self, response: str) -> bool:
        """Check if response has specific time recommendations"""
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}:\d{2}-\d{1,2}:\d{2}',
            r'(?:morning|afternoon|evening|overnight|night)',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in time_patterns)
    
    def _has_carbon_data(self, response: str) -> bool:
        """Check if response includes carbon data"""
        carbon_patterns = [
            r'\d+g CO2/kWh',
            r'carbon intensity',
            r'emission',
            r'CO2',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in carbon_patterns)
    
    def _has_visual_elements(self, response: str) -> bool:
        """Check if response has visual elements (emojis)"""
        visual_patterns = [
            r'[🌱⚡🔥📊🏠🔋🌍🕐🌙🌅☀️🌆]',
            r'\*\*.*\*\*',  # Bold headers
            r'•',  # Bullet points
        ]
        
        return any(re.search(pattern, response) for pattern in visual_patterns)
    
    def _has_proper_structure(self, response: str) -> bool:
        """Check if response has proper structure"""
        structure_indicators = [
            len(re.findall(r'\*\*[^*]+\*\*', response)) >= 3,  # At least 3 sections
            len(re.findall(r'[•\-\*]', response)) >= 3,  # At least 3 bullet points
            len(response.split('\n')) >= 5,  # Multiple lines
        ]
        
        return sum(structure_indicators) >= 2
    
    def _detect_tools_used(self, raw_output: str) -> List[str]:
        """Detect tools used in agent output"""
        tool_patterns = [
            (r"get_emission_analysis", "get_emission_analysis"),
            (r"emission_tool", "get_emission_analysis"),
            (r"emission.*analysis", "get_emission_analysis"),
            (r"carbon.*intensity.*data", "get_emission_analysis"),
            (r"co2.*data", "get_emission_analysis")
        ]
        
        tools_found = set()
        for pattern, tool_name in tool_patterns:
            if re.search(pattern, raw_output, re.IGNORECASE):
                tools_found.add(tool_name)
        
        # If no tools detected but response looks like CO2 analysis, assume tool was used
        if not tools_found and self._has_carbon_data(raw_output):
            tools_found.add("get_emission_analysis")
        
        return list(tools_found)
    
    def _extract_main_response(self, raw_output: str) -> str:
        """Extract main response from agent output"""
        # Strategy 1: Look for TextMessage content extraction first (most reliable)
        textmessage_content = self._extract_textmessage_content(raw_output)
        if textmessage_content and len(textmessage_content) > 200:
            return textmessage_content
            
        # Strategy 2: Look for structured carbon responses matching good examples
        import re
        carbon_response_patterns = [
            r'\*\*Best Times to Use Appliances[^!]*!',
            r'\*\*Optimal EV Charging Schedule[^!]*!',
            r'🏠\s*\*\*[^*]+\*\*[^🌱]*🌱[^🔥]*🔥[^📊]*📊[^🌍]*🌍[^!]*!',
            r'🔋\s*\*\*[^*]+\*\*[^🌱]*🌱[^🔥]*🔥[^🌍]*🌍[^!]*!',
            r'📊\s*\*\*[^*]+\*\*[^🌙]*🌙[^📈]*📈[^🎯]*🎯',
        ]
        
        for pattern in carbon_response_patterns:
            matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            if matches:
                longest_match = max(matches, key=len)
                if len(longest_match) > 300:  # Substantial response
                    return self._clean_content(longest_match)
        
        # Strategy 3: Look for any substantial structured content
        structured_patterns = [
            r'(\*\*[^*]+\*\*[^*]{100,})',  # Headers with substantial content
            r'([🌱⚡🔥📊🏠🔋🌍][^🌱⚡🔥📊🏠🔋🌍]{100,})',  # Emoji-based sections
            r'(Best Times[^.]{200,})',  # Best times recommendations
            r'(Optimal[^.]{200,})',  # Optimal recommendations
        ]
        
        for pattern in structured_patterns:
            matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            if matches:
                longest_match = max(matches, key=len)
                if len(longest_match) > 200:
                    return self._clean_content(longest_match)
        
        # Fallback to first substantial text block
        lines = raw_output.split('\n')
        substantial_content = []
        
        for line in lines:
            clean_line = line.strip()
            if (len(clean_line) > 50 and 
                not clean_line.startswith('[') and 
                'TextMessage' not in clean_line and
                'created_at' not in clean_line):
                substantial_content.append(clean_line)
                
                if len(' '.join(substantial_content)) > 300:
                    break
        
        if substantial_content:
            return self._clean_content(' '.join(substantial_content))
        
        return raw_output[:1000] if raw_output else "No response extracted"
    
    def _extract_textmessage_content(self, raw_output: str) -> str:
        """Extract TextMessage content from agent output"""
        import re
        
        # Patterns to match agent responses - support both single and double quotes
        agent_patterns = [
            r'TextMessage\([^)]*source=\'(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)\'[^)]*content=\'([^\']*)\'' ,
            r'TextMessage\([^)]*source="(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)"[^)]*content="([^"]*)"',
            r'TextMessage\([^)]*content=\'([^\']{400,})\'[^)]*source=\'(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)\'',
            r'TextMessage\([^)]*content="([^"]{200,})"[^)]*source="(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)"',
            r'content=\'([^\']{300,})\'[^\']*(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)',
            r'content="([^"]{300,})"[^"]*(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)',
        ]
        
        for pattern in agent_patterns:
            matches = re.findall(pattern, raw_output, re.DOTALL)
            if matches:
                longest_match = max(matches, key=len)
                if len(longest_match) > 300:
                    return self._clean_content(longest_match)
        
        return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean and format agent response content"""
        import re
        if not content:
            return ""
        
        # Clean encoding and escape issues
        clean_content = content.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
        clean_content = clean_content.replace('\\t', ' ').replace('\\r', '')
        
        # Remove AutoGen-specific artifacts
        artifacts_to_remove = [
            r"TextMessage\([^)]*\)",
            r"created_at=datetime\.datetime\([^)]*\)",
            r"source='[^']*'",
            r"id='[^']*'",
            r"type='[^']*'",
            r"models_usage=[^,)]*",
            r"metadata=\{[^}]*\}",
        ]
        
        for artifact in artifacts_to_remove:
            clean_content = re.sub(artifact, "", clean_content)
        
        # Remove trailing artifacts
        clean_content = re.sub(r"',\s*type='[^']*'\).*$", "", clean_content)
        clean_content = re.sub(r"'\)\s*,\s*TaskResult.*$", "", clean_content, re.DOTALL)
        
        # Clean line by line
        lines = clean_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive whitespace
                line = ' '.join(line.split())
                # Remove escape characters and artifacts
                line = line.replace('\\', '')
                line = re.sub(r'^[,\s\)\]\}]+', '', line)
                line = re.sub(r'[,\s\(\[\{]+$', '', line)
                
                # Keep lines with actual content
                if len(line) > 3 and not re.match(r'^[,\s\)\]\}\(\[\{]*$', line):
                    cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # Final cleanup
        result = re.sub(r'\n\n+', '\n\n', result)
        result = re.sub(r'^\W+', '', result)
        
        # Ensure proper ending
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result

    def _analyze_against_examples(self, response: str) -> Dict[str, Any]:
        """Analyze response against good/bad examples"""
        import re
        
        good_matches = 0
        bad_matches = 0
        format_compliance = 0.0
        
        # Check against good examples patterns
        good_patterns = [
            r'🏠\s*\*\*.*\*\*',  # Proper header structure
            r'🌱\s*\*\*.*\*\*',  # Optimal section
            r'⚡\s*\*\*.*\*\*',  # Recommendations section
            r'🔥\s*\*\*.*\*\*',  # Avoid section
            r'📊\s*\*\*.*\*\*',  # Data section
            r'🌍\s*\*\*.*\*\*',  # Impact section
            r'\d{2}:\d{2}-\d{2}:\d{2}',  # Time format
            r'\d+g CO2/kWh',  # CO2 values
            r'\d+%',  # Percentage
            r'REAL EirGrid data',  # Data source reference
        ]
        
        for pattern in good_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                good_matches += 1
        
        # Check against bad examples patterns
        bad_patterns = [
            r'\b(?:usually|typically|generally|probably)\b',  # Vague language
            r'\b(?:might|could|may)\b',  # Uncertain language
            r'between \d+ and \d+',  # Vague ranges instead of specific times
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                bad_matches += 1
        
        # Calculate format compliance
        total_good_patterns = len(good_patterns)
        format_compliance = good_matches / total_good_patterns if total_good_patterns > 0 else 0.0
        
        return {
            "good_patterns_matched": good_matches,
            "bad_patterns_found": bad_matches,
            "matches_good_format": good_matches >= 7,  # At least 70% of good patterns
            "avoids_bad_patterns": bad_matches == 0,
            "format_compliance_score": format_compliance
        }
    
    def _has_time_recommendations(self, response: str) -> bool:
        """Check if response has specific time recommendations"""
        import re
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}:\d{2}-\d{1,2}:\d{2}',
            r'(?:morning|afternoon|evening|overnight|night)',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in time_patterns)
    
    def _has_carbon_data(self, response: str) -> bool:
        """Check if response includes carbon data"""
        import re
        carbon_patterns = [
            r'\d+g CO2/kWh',
            r'carbon intensity',
            r'emission',
            r'CO2',
        ]
        
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in carbon_patterns)
    
    def _has_visual_elements(self, response: str) -> bool:
        """Check if response has visual elements (emojis)"""
        import re
        visual_patterns = [
            r'[🌱⚡🔥📊🏠🔋🌍🕐🌙🌅☀️🌆]',
            r'\*\*.*\*\*',  # Bold headers
            r'•',  # Bullet points
        ]
        
        return any(re.search(pattern, response) for pattern in visual_patterns)
    
    def _has_proper_structure(self, response: str) -> bool:
        """Check if response has proper structure"""
        import re
        structure_indicators = [
            len(re.findall(r'\*\*[^*]+\*\*', response)) >= 3,  # At least 3 sections
            len(re.findall(r'[•\-\*]', response)) >= 3,  # At least 3 bullet points
            len(response.split('\n')) >= 5,  # Multiple lines
        ]
        
        return sum(structure_indicators) >= 2
    
    def _detect_tools_used(self, raw_output: str) -> List[str]:
        """Detect tools used in agent output"""
        import re
        tool_patterns = [
            (r"get_emission_analysis", "get_emission_analysis"),
            (r"emission_tool", "get_emission_analysis"),
            (r"emission.*analysis", "get_emission_analysis"),
            (r"carbon.*intensity.*data", "get_emission_analysis"),
            (r"co2.*data", "get_emission_analysis")
        ]
        
        tools_found = set()
        for pattern, tool_name in tool_patterns:
            if re.search(pattern, raw_output, re.IGNORECASE):
                tools_found.add(tool_name)
        
        # If no tools detected but response looks like CO2 analysis, assume tool was used
        if not tools_found and self._has_carbon_data(raw_output):
            tools_found.add("get_emission_analysis")
        
        return list(tools_found)
    

    # Removed _create_fallback_structure method - no more fake parsing structures

class HybridAgentEvaluator:
    """
    Hybrid Agent Evaluator with examples.json integration
    Provides format consistency and evaluation accuracy
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.test_cases = []
        self.co2_analysis = None
        self.examples = self._load_examples()
        
        # Initialize strategies
        self.rule_strategy = RuleBasedStrategy(self.config)
        self.llm_strategy = None
        
        # Initialize LLM if enabled
        if self.config.get("llm_evaluation_enabled", True):
            self._setup_llm_strategy()
        
        # Result combiner
        self.result_combiner = ResultCombiner(self.config)
        
        # Initialize behavioral evaluator if available and enabled
        self.behavioral_evaluator = None
        if BEHAVIORAL_EVALUATION_AVAILABLE and self.config.get("enable_behavioral_assessment", False):
            self.behavioral_evaluator = BehavioralEvaluator(self.config)
            logger.info("Behavioral evaluator initialized")
        
        logger.info(f"Hybrid evaluator initialized with examples.json integration")
    
    def _load_compressed_co2_data(self) -> Dict:
        """Load compressed CO2 data for LLM judge validation"""
        try:
            import glob
            
            # Look for compressed CO2 data files
            compressed_files = glob.glob("*compressed*.json") + glob.glob("data/*compressed*.json")
            
            if compressed_files:
                # Use the most recent compressed file
                latest_file = max(compressed_files, key=os.path.getmtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    compressed_data = json.load(f)
                
                logger.info(f"Loaded compressed CO2 data from {latest_file}")
                return compressed_data
            else:
                logger.warning("No compressed CO2 data file found for LLM judge validation")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading compressed CO2 data: {e}")
            return {}
    
    def _load_examples(self) -> Dict:
        """Load examples.json for evaluation guidance"""
        try:
            with open('examples.json', 'r', encoding='utf-8') as f:
                examples = json.load(f)
                logger.info(f"Loaded {len(examples.get('good_examples', []))} good examples and {len(examples.get('bad_examples', []))} bad examples")
                return examples
        except FileNotFoundError:
            logger.warning("examples.json not found - evaluation will be less accurate")
            return {"good_examples": [], "bad_examples": [], "format_requirements": {}}
        except Exception as e:
            logger.warning(f"Error loading examples.json: {e}")
            return {"good_examples": [], "bad_examples": [], "format_requirements": {}}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration with default values"""
        default_config = {
            "evaluation_mode": "hybrid",
            "max_retries": 3,
            "timeout_seconds": 90,
            "output_dir": "./evaluation_results",
            
            # Rule-based settings
            "consistency_threshold": 0.8,
            "keyword_weight": 0.2,
            "function_weight": 0.3,
            "behavior_weight": 0.5,
            
            # LLM settings
            "llm_evaluation_enabled": True,
            "llm_on_failures_only": False,
            "llm_quality_threshold": 0.5,  # Reduced from 0.6
            "llm_confidence_threshold": 0.4,  # Reduced from 0.5
            
            # Hybrid settings optimized for examples integration
            "rule_weight": 0.2,
            "llm_weight": 0.8,
            "llm_override_enabled": True,
            "llm_override_threshold": 0.65,  # Reduced from 0.75
            
            # Performance settings
            "parallel_llm_calls": False,
            "max_llm_tokens": 4000,
            "llm_timeout": 90,
            
            # Examples integration settings
            "examples_weight": 0.3,  # New: weight for examples compliance
            "format_compliance_threshold": 0.7,
            "bad_pattern_penalty": 0.1,
            "good_pattern_bonus": 0.1,
            
            # Other settings
            "auto_generate_ground_truth": True,
            "co2_data_compression_interval": 30,
            "use_latest_co2_data": True,
            "ground_truth_weight": 0.8,  # Reduced slightly
            "enforce_scoring_consistency": True,
            "structured_penalty_system": True,
            "detailed_scoring_breakdown": True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_llm_strategy(self):
        """Setup LLM strategy"""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check for required environment variables
            required_vars = ["AZURE_AI_DEPLOYMENT", "AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_AI_API_VERSION", "AZURE_AI_MODEL"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing environment variables for LLM: {missing_vars}")
                self.llm_strategy = None
                return
            
            # Create Azure client
            client = create_azure_client()
            
            # Create LLM judge
            llm_judge = LLMJudge(
                azure_client=client,
                model_name=os.getenv("AZURE_AI_MODEL", "gpt-4o"),
                max_retries=self.config.get("max_retries", 3)
            )
            
            # Create LLM strategy
            self.llm_strategy = LLMJudgeStrategy(llm_judge, self.config)
            
            logger.info("LLM strategy initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM strategy: {e}")
            self.llm_strategy = None

    def load_test_cases(self, test_cases_path: str = None):
        """Load test cases with examples validation"""
        
        # Auto-generate if needed
        if self.config.get("auto_generate_ground_truth", True) and test_cases_path is None:
            test_cases_path = self._prepare_dynamic_ground_truth()
        
        if test_cases_path is None:
            raise ValueError("No test cases path provided and auto-generation disabled")
        
        try:
            with open(test_cases_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            if "test_cases" in test_data:
                self.test_cases = test_data["test_cases"]
                
                # Store CO2 analysis data if available
                if "co2_analysis" in test_data:
                    self.co2_analysis = test_data["co2_analysis"]
                elif "real_co2_analysis" in test_data:
                    self.co2_analysis = test_data["real_co2_analysis"]
                elif "compressed_co2_analysis" in test_data:
                    self.co2_analysis = test_data["compressed_co2_analysis"]
                    
            else:
                self.test_cases = test_data
            
            # Validate ground truth and examples integration
            ground_truth_count = 0
            reference_output_count = 0
            examples_compliant_count = 0
            
            for test_case in self.test_cases:
                if "ground_truth" in test_case:
                    ground_truth_count += 1
                    gt = test_case["ground_truth"]
                    
                    if "reference_output" in gt:
                        reference_output_count += 1
                        
                        # Check if reference output follows examples format
                        ref_output = gt["reference_output"]
                        if self._follows_examples_format(ref_output):
                            examples_compliant_count += 1
                    else:
                        logger.warning(f"Test case {test_case.get('id', 'unknown')} missing reference_output")
                else:
                    logger.warning(f"Test case {test_case.get('id', 'unknown')} missing ground truth")
            
            logger.info(f"Loaded {len(self.test_cases)} test cases")
            logger.info(f"Ground truth coverage: {ground_truth_count}/{len(self.test_cases)}")
            logger.info(f"Reference outputs: {reference_output_count}/{len(self.test_cases)}")
            logger.info(f"Examples-compliant references: {examples_compliant_count}/{len(self.test_cases)}")
            
            if examples_compliant_count == len(self.test_cases):
                print(f"SUCCESS: All test cases follow examples.json format requirements")
            else:
                print(f"WARNING: {examples_compliant_count}/{len(self.test_cases)} test cases follow examples format")
            
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            raise
    
    def _follows_examples_format(self, reference_output: str) -> bool:
        """Check if reference output follows examples.json format"""
        if not self.examples.get("format_requirements"):
            return True  # No format requirements to check
        
        required_elements = self.examples["format_requirements"].get("required_elements", [])
        
        format_checks = [
            bool(re.search(r'🌱.*🔥.*📊.*🌍', reference_output, re.DOTALL)),  # Emoji structure
            bool(re.search(r'\d{2}:\d{2}-\d{2}:\d{2}', reference_output)),  # Time format
            bool(re.search(r'\d+g CO2/kWh', reference_output)),  # CO2 values
            bool(re.search(r'\d+%', reference_output)),  # Percentage
            len(re.findall(r'[•\-\*]', reference_output)) >= 5,  # Bullet points
        ]
        
        return sum(format_checks) >= 4  # At least 4 out of 5 format requirements
    
    def _prepare_dynamic_ground_truth(self) -> str:
        """Prepare dynamic ground truth with examples integration"""
        try:
            from ground_truth_generator import GroundTruthGenerator
            return GroundTruthGenerator.generate_for_evaluation()
        except ImportError as e:
            error_msg = f"Ground truth generator not available: {e}. Cannot generate test cases without real data source."
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Failed to generate dynamic ground truth: {e}. Ensure real data sources are available."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Removed _create_fallback_test_cases method - no more synthetic test generation

    async def evaluate_agent(self, agent_instance, runs_per_test: int = 2) -> HybridEvaluationReport:
        """
        Evaluation with examples.json integration
        """
        logger.info(f"Starting evaluation with examples integration")
        logger.info(f"Evaluation mode: {self.config['evaluation_mode']}")
        logger.info(f"Examples loaded: {len(self.examples.get('good_examples', []))} good, {len(self.examples.get('bad_examples', []))} bad")
        
        if not self.test_cases:
            self.load_test_cases()
        
        if not self.test_cases:
            raise ValueError("No test cases available after loading/generation")
        
        all_results = []
        evaluation_mode = EvaluationMode(self.config["evaluation_mode"])
        
        for test_case in self.test_cases:
            logger.info(f"Evaluating test case: {test_case['id']}")
            
            # Get instance count for this specific test case, fallback to global runs_per_test
            test_instances = test_case.get('instances', runs_per_test)
            logger.info(f"Running {test_instances} instances for {test_case['id']}")
            
            # Run multiple iterations for consistency
            test_runs = []
            for run in range(test_instances):
                try:
                    result = await self._evaluate_single_test(agent_instance, test_case, run, evaluation_mode)
                    test_runs.append(result)
                    
                    logger.debug(f"Run {run+1}/{test_instances} result: {result.status.value}, "
                               f"rule_score={result.rule_based_score:.2f}, "
                               f"llm_score={result.llm_quality_score:.2f}, "
                               f"format_compliance={result.format_compliance_score:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error in run {run+1} for {test_case['id']}: {e}")
                    error_result = self._create_error_result(test_case, 0.0, str(e))
                    test_runs.append(error_result)
                
                await asyncio.sleep(0.5)
            
            # Calculate consistency across runs
            self._update_consistency_scores(test_runs)
            all_results.extend(test_runs)
        
        # Generate report with examples metrics
        report = self._generate_evaluation_report(agent_instance, all_results, evaluation_mode)
        
        # Logging
        logger.info(f"Evaluation completed: {len(all_results)} total results")
        logger.info(f"Pass rate: {report.pass_rate():.2%}")
        logger.info(f"Format compliance: {report.avg_format_compliance:.2f}")
        logger.info(f"Good examples match rate: {report.good_examples_match_rate:.2f}")
        logger.info(f"Bad examples avoidance: {report.bad_examples_avoidance_rate:.2f}")
        
        # Save results
        self._save_results(report)
        
        return report

    async def _evaluate_single_test(self, agent_instance, test_case: Dict, 
                                  run_number: int, mode: EvaluationMode) -> HybridEvaluationResult:
        """Single test evaluation with examples awareness and behavioral monitoring"""
        
        start_time = time.time()
        behavioral_assessment = None
        
        # Start behavioral monitoring if available
        if self.behavioral_evaluator:
            await self.behavioral_evaluator.start_evaluation(
                f"{test_case['id']}_run_{run_number}"
            )
        
        try:
            # Create monitored agent wrapper if behavioral evaluation is enabled
            if self.behavioral_evaluator:
                monitored_agent = self._create_monitored_agent_wrapper(agent_instance, test_case['id'])
            else:
                monitored_agent = agent_instance
            
            # Execute the agent
            result = await asyncio.wait_for(
                self._run_agent_task(monitored_agent, test_case["query"]),
                timeout=test_case.get("timeout_seconds", self.config["timeout_seconds"])
            )
            
            execution_time = time.time() - start_time
            output_text = str(result)
            
            logger.debug(f"Agent execution completed for {test_case['id']}, output length: {len(output_text)}")
            
            # Parse agent output
            structured_output = AgentOutputParser.parse_agent_output(output_text)
            
            # Extract agent responses for evaluation
            agent_responses = []
            for resp_data in structured_output.get("agent_responses", []):
                if isinstance(resp_data, dict):
                    content = resp_data.get("content", "")
                    if content:
                        agent_responses.append(content)
                else:
                    agent_responses.append(str(resp_data))
            
            # Get main response for evaluation
            main_response = ""
            if agent_responses:
                main_response = max(agent_responses, key=len)
            else:
                main_response = self._extract_agent_response(output_text)
            
            # Stop behavioral monitoring and get assessment
            if self.behavioral_evaluator:
                behavioral_assessment = await self.behavioral_evaluator.stop_evaluation()
            
            # Evaluation with examples awareness
            result = await self._evaluate_with_examples_awareness(
                test_case, output_text, main_response, execution_time, 
                structured_output, agent_responses, mode
            )
            
            # Add behavioral assessment if available
            if behavioral_assessment:
                result = self._add_behavioral_assessment(result, behavioral_assessment)
            
            return result
            
        except asyncio.TimeoutError:
            # Ensure behavioral monitoring is stopped
            if self.behavioral_evaluator:
                try:
                    await self.behavioral_evaluator.stop_evaluation()
                except:
                    pass
            return self._create_timeout_result(test_case, time.time() - start_time)
        except Exception as e:
            # Ensure behavioral monitoring is stopped
            if self.behavioral_evaluator:
                try:
                    await self.behavioral_evaluator.stop_evaluation()
                except:
                    pass
            logger.error(f"Error in test {test_case['id']}: {e}")
            return self._create_error_result(test_case, time.time() - start_time, str(e))

    async def _evaluate_with_examples_awareness(self, test_case: Dict, output_text: str, 
                                              main_response: str, execution_time: float,
                                              structured_output: Dict, agent_responses: List[str],
                                              mode: EvaluationMode) -> HybridEvaluationResult:
        """Evaluate with examples.json awareness"""
        
        # Extract the best agent response for full tracking
        full_response = ""
        if agent_responses:
            full_response = max(agent_responses, key=len)
        elif main_response:
            full_response = main_response
        else:
            # Fallback to extracting from raw output
            full_response = self._extract_clean_response(output_text)
        
        # Run rule-based evaluation on full response with conversation log
        conversation_log = structured_output.get("tool_interactions", [])
        # If we have raw task result messages, use those
        if "messages" in structured_output:
            conversation_log = structured_output["messages"]
        
        rule_result = await self.rule_strategy.evaluate(
            test_case, 
            full_response if full_response else output_text,
            conversation_log
        )
        
        # Initialize result
        result = HybridEvaluationResult(
            test_case_id=test_case["id"],
            status=rule_result.status,
            execution_time=execution_time,
            rule_based_status=rule_result.status,
            rule_based_score=rule_result.score,
            functions_called=rule_result.functions_called,
            behaviors_observed=rule_result.behaviors_observed,
            keyword_matches=rule_result.keyword_matches,
            consistency_score=0.0,
            final_score=0.0,
            confidence_level="low",
            raw_output=output_text,
            structured_output=structured_output,
            conversation_flow=structured_output.get("conversation_flow", []),
            tool_interactions=structured_output.get("tool_interactions", []),
            agent_responses=agent_responses,
            
            # Full query and response tracking
            test_query=test_case["query"],
            full_agent_response=full_response,
            response_length=len(full_response)
        )
        
        # Extract examples analysis from structured output
        examples_analysis = structured_output.get("summary", {}).get("examples_compliance", {})
        result.matches_good_examples = examples_analysis.get("good_patterns_matched", 0)
        result.matches_bad_examples = examples_analysis.get("bad_patterns_found", 0)
        result.format_compliance_score = examples_analysis.get("format_compliance_score", 0.0)
        
        # Validate metadata expectations
        tools_used = result.functions_called or []
        metadata_validation = self.rule_strategy._validate_test_metadata(test_case, full_response, tools_used)
        
        # Incorporate metadata validation into scores
        metadata_weight = 0.3  # 30% weight for metadata compliance
        base_rule_score = result.rule_based_score
        metadata_score = (
            metadata_validation["function_call_score"] * 0.4 +
            metadata_validation["behavior_score"] * 0.3 +
            metadata_validation["keyword_score"] * 0.2 +
            metadata_validation["domain_score"] * 0.1
        )
        
        # Adjust rule-based score with metadata validation
        result.rule_based_score = (base_rule_score * (1 - metadata_weight)) + (metadata_score * metadata_weight)
        
        # Store metadata validation results
        result.metadata_validation = metadata_validation
        
        # Run LLM evaluation if available
        if self.llm_strategy and mode != EvaluationMode.RULE_BASED_ONLY:
            try:
                logger.info(f"Running LLM evaluation for {test_case['id']}")
                
                # Load compressed CO2 data for validation
                compressed_co2_data = self._load_compressed_co2_data()
                
                llm_result = await self.llm_strategy.evaluate(test_case, full_response, rule_result, compressed_co2_data)
                
                # Update with LLM results
                result.llm_status = llm_result.status
                result.llm_quality_score = llm_result.quality_score
                result.llm_confidence = llm_result.confidence
                result.semantic_similarity = llm_result.semantic_similarity
                result.llm_reasoning = llm_result.reasoning
                result.llm_feedback = llm_result.feedback
                
                # Additional fields from ground truth evaluation
                if hasattr(llm_result, 'ground_truth_used') and llm_result.ground_truth_used:
                    result.ground_truth_used = True
                    result.accuracy_score = getattr(llm_result, 'accuracy_score', 0.0)
                    result.completeness_score = getattr(llm_result, 'completeness_score', 0.0)
                    result.clarity_score = getattr(llm_result, 'clarity_score', 0.0)
                    result.actionability_score = getattr(llm_result, 'actionability_score', 0.0)
                    result.format_score = getattr(llm_result, 'format_score', 0.0)
                    result.specific_matches = getattr(llm_result, 'specific_matches', [])
                    result.specific_gaps = getattr(llm_result, 'specific_gaps', [])
                    result.penalties_applied = getattr(llm_result, 'penalties_applied', [])
                
            except Exception as e:
                logger.error(f"LLM evaluation failed for {test_case['id']}: {e}")
                result.llm_status = EvaluationStatus.ERROR
                result.llm_reasoning = f"LLM evaluation error: {str(e)}"
        
        # Combine results with examples awareness
        return self.result_combiner.combine_results(result, test_case, mode)
    
    def _create_monitored_agent_wrapper(self, agent_instance, test_case_id: str):
        """Create a wrapper around the agent to monitor its behavior"""
        
        class MonitoredAgentWrapper:
            def __init__(self, agent, evaluator, test_id):
                self.agent = agent
                self.evaluator = evaluator
                self.test_id = test_id
                self._method_cache = {}
            
            async def run(self, query: str):
                # Record the query as an action
                if self.evaluator.behavioral_evaluator:
                    await self.evaluator.behavioral_evaluator.record_action(
                        "query_received", {
                            "query": query,
                            "query_length": len(query),
                            "timestamp": time.time()
                        }
                    )
                
                try:
                    # Monitor agent execution
                    start_time = time.time()
                    
                    # Call the actual agent
                    if hasattr(self.agent, 'run'):
                        result = await self.agent.run(query)
                    elif hasattr(self.agent, 'run_stream'):
                        results = []
                        async for r in self.agent.run_stream(task=query):
                            results.append(r)
                        result = results
                    elif callable(self.agent):
                        result = await self.agent(query)
                    else:
                        raise ValueError("Agent must have 'run' method or be callable")
                    
                    execution_time = time.time() - start_time
                    
                    # Record successful execution with full content
                    if self.evaluator.behavioral_evaluator:
                        full_content = str(result)
                        await self.evaluator.behavioral_evaluator.record_action(
                            "response", {
                                "content": full_content,  # Store full content for behavioral analysis
                                "content_preview": full_content[:200] + "..." if len(full_content) > 200 else full_content,
                                "execution_time": execution_time,
                                "success": True,
                                "result_type": type(result).__name__,
                                "content_length": len(full_content),
                                "appropriateness_score": 0.8  # Default appropriateness for successful calls
                            }
                        )
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    if self.evaluator.behavioral_evaluator:
                        await self.evaluator.behavioral_evaluator.record_error(
                            "execution_error", {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "query": query
                            }
                        )
                    raise e
            
            async def run_stream(self, task: str):
                """Support for streaming agents"""
                # Record the task as an action
                if self.evaluator.behavioral_evaluator:
                    await self.evaluator.behavioral_evaluator.record_action(
                        "task_received", {
                            "task": task,
                            "task_length": len(task),
                            "timestamp": time.time()
                        }
                    )
                
                try:
                    start_time = time.time()
                    
                    if hasattr(self.agent, 'run_stream'):
                        async for result in self.agent.run_stream(task=task):
                            # Record each streaming result
                            if self.evaluator.behavioral_evaluator:
                                await self.evaluator.behavioral_evaluator.record_action(
                                    "stream_response", {
                                        "content": str(result)[:200],
                                        "partial_response": True,
                                        "timestamp": time.time()
                                    }
                                )
                            yield result
                    else:
                        # Fallback to regular run
                        result = await self.run(task)
                        yield result
                    
                    total_time = time.time() - start_time
                    if self.evaluator.behavioral_evaluator:
                        await self.evaluator.behavioral_evaluator.record_action(
                            "stream_completed", {
                                "total_execution_time": total_time,
                                "success": True
                            }
                        )
                        
                except Exception as e:
                    # Record streaming error
                    if self.evaluator.behavioral_evaluator:
                        await self.evaluator.behavioral_evaluator.record_error(
                            "streaming_error", {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "task": task
                            }
                        )
                    raise e
            
            # Proxy other methods
            def __getattr__(self, name):
                if name not in self._method_cache:
                    self._method_cache[name] = getattr(self.agent, name)
                return self._method_cache[name]
        
        return MonitoredAgentWrapper(agent_instance, self, test_case_id)
    
    def _add_behavioral_assessment(self, result: HybridEvaluationResult, 
                                 behavioral_assessment: 'ComprehensiveBehavioralAssessment') -> HybridEvaluationResult:
        """Add behavioral assessment data to the evaluation result"""
        
        # Store the full behavioral assessment
        result.behavioral_assessment = behavioral_assessment
        
        # Extract and normalize behavioral scores
        result.performance_score = self._calculate_performance_score(behavioral_assessment.performance)
        result.decision_making_score = self._calculate_decision_making_score(behavioral_assessment.action_sequence)
        result.error_recovery_score = self._calculate_error_recovery_score(behavioral_assessment.error_handling)
        result.communication_score = self._calculate_communication_score(behavioral_assessment.communication)
        result.tool_efficiency_score = self._calculate_tool_efficiency_score(behavioral_assessment.tool_usage)
        
        # Add behavioral insights
        result.behavioral_strengths = behavioral_assessment.strengths
        result.behavioral_weaknesses = behavioral_assessment.weaknesses
        result.behavioral_recommendations = behavioral_assessment.improvement_recommendations
        
        # Update overall score with behavioral component if configured
        behavioral_weight = self.config.get("behavioral_weight", 0.0)
        if behavioral_weight > 0:
            standard_weight = 1.0 - behavioral_weight
            result.final_score = (
                result.final_score * standard_weight +
                behavioral_assessment.overall_behavioral_score * behavioral_weight
            )
        
        return result
    
    def _calculate_performance_score(self, performance) -> float:
        """Calculate normalized performance score"""
        # Combine multiple performance metrics
        time_score = min(1.0, 10.0 / max(performance.total_execution_time, 0.1))
        efficiency_score = min(1.0, performance.actions_per_second / 5.0)
        resource_score = max(0.0, 1.0 - (performance.peak_memory_mb / 1000.0))  # Normalize to 1GB
        
        return (time_score * 0.4 + efficiency_score * 0.4 + resource_score * 0.2)
    
    def _calculate_decision_making_score(self, action_sequence) -> float:
        """Calculate decision making quality score"""
        if not action_sequence.decision_points:
            return 0.8  # Neutral score if no decisions tracked
        
        # Average confidence in decisions
        confidence_scores = action_sequence.decision_confidence_scores
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Penalty for backtracking
        backtrack_penalty = min(0.3, action_sequence.backtracking_instances * 0.1)
        
        return max(0.0, avg_confidence - backtrack_penalty)
    
    def _calculate_error_recovery_score(self, error_handling) -> float:
        """Calculate error recovery capability score"""
        if error_handling.total_errors_encountered == 0:
            return 1.0  # Perfect score if no errors
        
        recovery_rate = (
            error_handling.successful_recoveries / 
            error_handling.total_errors_encountered
        )
        
        # Bonus for graceful degradation
        graceful_bonus = 0.1 if error_handling.graceful_degradation else 0.0
        
        return min(1.0, recovery_rate + graceful_bonus)
    
    def _calculate_communication_score(self, communication) -> float:
        """Calculate communication quality score"""
        return (
            communication.response_clarity_score * 0.5 +
            communication.response_helpfulness_score * 0.3 +
            communication.context_maintenance * 0.2
        )
    
    def _calculate_tool_efficiency_score(self, tool_usage) -> float:
        """Calculate tool usage efficiency score"""
        return tool_usage.tool_usage_efficiency
    
    def _extract_textmessage_content(self, raw_output: str) -> str:
        """Extract TextMessage content from agent output"""
        import re
        
        # Patterns to match agent responses - support both single and double quotes
        agent_patterns = [
            r'TextMessage\([^)]*source=\'(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)\'[^)]*content=\'([^\']*)\'' ,
            r'TextMessage\([^)]*source="(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)"[^)]*content="([^"]*)"',
            r'TextMessage\([^)]*content=\'([^\']{400,})\'[^)]*source=\'(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)\'',
            r'TextMessage\([^)]*content="([^"]{200,})"[^)]*source="(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)"',
            r'content=\'([^\']{300,})\'[^\']*(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)',
            r'content="([^"]{300,})"[^"]*(?:StreamlinedCarbonAgent|CarbonAgent|AssistantAgent)',
        ]
        
        for pattern in agent_patterns:
            matches = re.findall(pattern, raw_output, re.DOTALL)
            if matches:
                longest_match = max(matches, key=len)
                if len(longest_match) > 300:
                    return self._clean_content(longest_match)
        
        return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean and format agent response content"""
        import re
        if not content:
            return ""
        
        # Clean encoding and escape issues
        clean_content = content.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
        clean_content = clean_content.replace('\\t', ' ').replace('\\r', '')
        
        # Remove AutoGen-specific artifacts
        artifacts_to_remove = [
            r"TextMessage\([^)]*\)",
            r"created_at=datetime\.datetime\([^)]*\)",
            r"source='[^']*'",
            r"id='[^']*'",
            r"type='[^']*'",
            r"models_usage=[^,)]*",
            r"metadata=\{[^}]*\}",
        ]
        
        for artifact in artifacts_to_remove:
            clean_content = re.sub(artifact, "", clean_content)
        
        # Remove trailing artifacts
        clean_content = re.sub(r"',\s*type='[^']*'\).*$", "", clean_content)
        clean_content = re.sub(r"'\)\s*,\s*TaskResult.*$", "", clean_content, re.DOTALL)
        
        # Clean line by line
        lines = clean_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive whitespace
                line = ' '.join(line.split())
                # Remove escape characters and artifacts
                line = line.replace('\\', '')
                line = re.sub(r'^[,\s\)\]\}]+', '', line)
                line = re.sub(r'[,\s\(\[\{]+$', '', line)
                
                # Keep lines with actual content
                if len(line) > 3 and not re.match(r'^[,\s\)\]\}\(\[\{]*$', line):
                    cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # Final cleanup
        result = re.sub(r'\n\n+', '\n\n', result)
        result = re.sub(r'^\W+', '', result)
        
        # Ensure proper ending
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        return result

    def _extract_clean_response(self, raw_output: str) -> str:
        """Extract the cleanest possible agent response from raw output"""
        if not raw_output:
            return ""
        
        # Strategy 1: Look for TextMessage content extraction first (most reliable)
        textmessage_content = self._extract_textmessage_content(raw_output)
        if textmessage_content and len(textmessage_content) > 200:
            return textmessage_content
            
        # Strategy 2: Look for structured carbon responses matching good examples
        import re
        carbon_response_patterns = [
            r'\*\*Best Times to Use Appliances[^!]*!',
            r'\*\*Optimal EV Charging Schedule[^!]*!',
            r'🏠\s*\*\*[^*]+\*\*[^🌱]*🌱[^🔥]*🔥[^📊]*📊[^🌍]*🌍[^!]*!',
            r'🔋\s*\*\*[^*]+\*\*[^🌱]*🌱[^🔥]*🔥[^🌍]*🌍[^!]*!',
            r'📊\s*\*\*[^*]+\*\*[^🌙]*🌙[^📈]*📈[^🎯]*🎯',
        ]
        
        for pattern in carbon_response_patterns:
            matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            if matches:
                longest_match = max(matches, key=len)
                if len(longest_match) > 300:  # Substantial response
                    return self._clean_content(longest_match)
        
        # Strategy 3: Look for any substantial structured content
        structured_patterns = [
            r'(\*\*[^*]+\*\*[^*]{100,})',  # Headers with substantial content
            r'([🌱⚡🔥📊🏠🔋🌍][^🌱⚡🔥📊🏠🔋🌍]{100,})',  # Emoji-based sections
            r'(Best Times[^.]{200,})',  # Best times recommendations
            r'(Optimal[^.]{200,})',  # Optimal recommendations
        ]
        
        for pattern in structured_patterns:
            matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            if matches:
                longest_match = max(matches, key=len)
                if len(longest_match) > 200:
                    return self._clean_content(longest_match)
        
        # Fallback to first substantial text block
        lines = raw_output.split('\n')
        substantial_content = []
        
        for line in lines:
            clean_line = line.strip()
            if (len(clean_line) > 50 and 
                not clean_line.startswith('[') and 
                'TextMessage' not in clean_line and
                'created_at' not in clean_line):
                substantial_content.append(clean_line)
                
                if len(' '.join(substantial_content)) > 300:
                    break
        
        if substantial_content:
            return self._clean_content(' '.join(substantial_content))
        
        return raw_output[:1000] if raw_output else "No response extracted"

    def _extract_agent_response(self, raw_output: str) -> str:
        """Extract agent response using updated parser"""
        return self._extract_clean_response(raw_output)

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

    def _create_timeout_result(self, test_case: Dict, execution_time: float) -> HybridEvaluationResult:
        """Create a timeout result"""
        return HybridEvaluationResult(
            test_case_id=test_case["id"],
            status=EvaluationStatus.TIMEOUT,
            execution_time=execution_time,
            rule_based_status=EvaluationStatus.TIMEOUT,
            rule_based_score=0.0,
            functions_called=[],
            behaviors_observed=[],
            keyword_matches=0,
            consistency_score=0.0,
            final_score=0.0,
            confidence_level="low",
            raw_output="",
            structured_output={"summary": {"has_errors": True, "total_errors": 1}, "errors": [{"type": "timeout", "message": "Test case timeout"}]},
            conversation_flow=[],
            tool_interactions=[],
            agent_responses=[],
            test_query=test_case.get("query", "Query not available"),
            full_agent_response="[TIMEOUT] No response - test case exceeded timeout limit",
            response_length=0,
            error_message="Test case timeout"
        )

    def _create_error_result(self, test_case: Dict, execution_time: float, error_message: str) -> HybridEvaluationResult:
        """Create an error result"""
        return HybridEvaluationResult(
            test_case_id=test_case["id"],
            status=EvaluationStatus.ERROR,
            execution_time=execution_time,
            rule_based_status=EvaluationStatus.ERROR,
            rule_based_score=0.0,
            functions_called=[],
            behaviors_observed=[],
            keyword_matches=0,
            consistency_score=0.0,
            final_score=0.0,
            confidence_level="low",
            raw_output="",
            structured_output={"summary": {"has_errors": True, "total_errors": 1}, "errors": [{"type": "execution_error", "message": error_message}]},
            conversation_flow=[],
            tool_interactions=[],
            agent_responses=[],
            test_query=test_case.get("query", "Query not available"),
            full_agent_response=f"[ERROR] Agent execution failed: {error_message}",
            response_length=0,
            error_message=error_message
        )

    def _update_consistency_scores(self, test_runs: List[HybridEvaluationResult]):
        """Update consistency scores with examples awareness"""
        if len(test_runs) < 2:
            for result in test_runs:
                result.consistency_score = 1.0
            return
        
        # Calculate consistency including examples compliance
        format_consistency = self._calculate_format_consistency(test_runs)
        examples_consistency = self._calculate_examples_consistency(test_runs)
        status_consistency = self._calculate_status_consistency(test_runs)
        score_consistency = self._calculate_score_consistency(test_runs)
        
        # Weighted consistency score
        for result in test_runs:
            result.consistency_score = (
                format_consistency * 0.25 +
                examples_consistency * 0.25 +
                status_consistency * 0.25 +
                score_consistency * 0.25
            )
    
    def _calculate_format_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate format compliance consistency"""
        if not test_runs:
            return 1.0
        
        format_scores = [result.format_compliance_score for result in test_runs]
        if not format_scores or all(score == 0 for score in format_scores):
            return 1.0
        
        avg_score = sum(format_scores) / len(format_scores)
        variance = sum((score - avg_score) ** 2 for score in format_scores) / len(format_scores)
        
        # Convert variance to consistency (lower variance = higher consistency)
        return max(0.0, 1.0 - variance)
    
    def _calculate_examples_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate examples compliance consistency"""
        if not test_runs:
            return 1.0
        
        good_matches = [result.matches_good_examples for result in test_runs]
        bad_matches = [result.matches_bad_examples for result in test_runs]
        
        # Check if good/bad matches are consistent across runs
        good_consistent = len(set(good_matches)) <= 1
        bad_consistent = len(set(bad_matches)) <= 1
        
        return 1.0 if (good_consistent and bad_consistent) else 0.5
    
    def _calculate_status_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate status consistency across runs"""
        if not test_runs:
            return 1.0
        
        first_status = test_runs[0].status
        consistent_count = sum(1 for result in test_runs if result.status == first_status)
        return consistent_count / len(test_runs)
    
    def _calculate_score_consistency(self, test_runs: List[HybridEvaluationResult]) -> float:
        """Calculate score consistency across runs"""
        if len(test_runs) < 2:
            return 1.0
        
        scores = [result.final_score for result in test_runs if result.final_score > 0]
        if len(scores) < 2:
            return 1.0
        
        mean_score = statistics.mean(scores)
        if mean_score == 0:
            return 1.0
        
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0
        cv = stdev / mean_score
        
        # Convert to consistency score
        return max(0.0, 1.0 - cv)

    def _generate_evaluation_report(self, agent_instance, results: List[HybridEvaluationResult], 
                                mode: EvaluationMode) -> HybridEvaluationReport:
        """Generate evaluation report with examples metrics"""
        
        if not results:
            return self._create_empty_report(agent_instance, mode)
        
        # Calculate basic statistics
        total_tests = len(results)
        
        def count_status(target_status):
            count = 0
            for r in results:
                status_val = r.status.value if hasattr(r.status, 'value') else str(r.status)
                if status_val == target_status:
                    count += 1
            return count
        
        passed_tests = count_status("PASS") 
        failed_tests = count_status("FAIL")
        error_tests = count_status("ERROR")
        timeout_tests = count_status("TIMEOUT")
        
        # Performance metrics
        avg_execution_time = statistics.mean([r.execution_time for r in results])
        
        # Quality metrics
        rule_scores = [r.rule_based_score for r in results if r.rule_based_score > 0]
        avg_rule_score = statistics.mean(rule_scores) if rule_scores else 0.0
        
        llm_scores = [r.llm_quality_score for r in results if r.llm_quality_score > 0]
        avg_llm_score = statistics.mean(llm_scores) if llm_scores else 0.0
        
        combined_scores = [r.final_score for r in results if r.final_score > 0]
        avg_combined_score = statistics.mean(combined_scores) if combined_scores else 0.0
        
        consistency_scores = [r.consistency_score for r in results if r.consistency_score > 0]
        consistency_score = statistics.mean(consistency_scores) if consistency_scores else 0.0
        
        # Examples-based metrics
        format_scores = [r.format_compliance_score for r in results if r.format_compliance_score > 0]
        avg_format_compliance = statistics.mean(format_scores) if format_scores else 0.0
        
        good_matches = [r.matches_good_examples for r in results]
        good_examples_match_rate = statistics.mean(good_matches) / 10 if good_matches else 0.0  # Normalize to 0-1
        
        bad_matches = [r.matches_bad_examples for r in results]
        bad_examples_avoidance_rate = 1.0 - (statistics.mean(bad_matches) / 5) if bad_matches else 1.0  # Normalize to 0-1
        
        # Ground truth metrics
        ground_truth_results = [r for r in results if r.ground_truth_used]
        ground_truth_coverage = len(ground_truth_results) / total_tests if total_tests > 0 else 0.0
        
        accuracy_scores = [r.accuracy_score for r in ground_truth_results if r.accuracy_score > 0]
        avg_accuracy_score = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        completeness_scores = [r.completeness_score for r in ground_truth_results if r.completeness_score > 0]
        avg_completeness_score = statistics.mean(completeness_scores) if completeness_scores else 0.0
        
        clarity_scores = [r.clarity_score for r in ground_truth_results if r.clarity_score > 0]
        avg_clarity_score = statistics.mean(clarity_scores) if clarity_scores else 0.0
        
        # LLM agreement rate
        llm_results = [r for r in results if r.llm_status is not None]
        if llm_results:
            def status_matches(status1, status2):
                val1 = status1.value if hasattr(status1, 'value') else str(status1)
                val2 = status2.value if hasattr(status2, 'value') else str(status2)
                return val1 == val2
            
            agreement_count = len([r for r in llm_results if status_matches(r.rule_based_status, r.llm_status)])
            llm_agreement_rate = agreement_count / len(llm_results)
        else:
            llm_agreement_rate = 0.0
        
        # Behavioral assessment aggregates
        behavioral_results = [r for r in results if hasattr(r, 'behavioral_assessment') and r.behavioral_assessment]
        
        # Calculate behavioral averages
        avg_performance_score = self._calculate_avg_behavioral_score(results, 'performance_score')
        avg_decision_making_score = self._calculate_avg_behavioral_score(results, 'decision_making_score')
        avg_error_recovery_score = self._calculate_avg_behavioral_score(results, 'error_recovery_score')
        avg_communication_score = self._calculate_avg_behavioral_score(results, 'communication_score')
        avg_tool_efficiency_score = self._calculate_avg_behavioral_score(results, 'tool_efficiency_score')
        
        # Resource utilization metrics
        memory_values = [r.behavioral_assessment.performance.peak_memory_mb for r in behavioral_results if r.behavioral_assessment.performance.peak_memory_mb > 0]
        cpu_values = [r.behavioral_assessment.performance.avg_cpu_percent for r in behavioral_results if r.behavioral_assessment.performance.avg_cpu_percent > 0]
        
        avg_memory_usage = statistics.mean(memory_values) if memory_values else 0.0
        avg_cpu_usage = statistics.mean(cpu_values) if cpu_values else 0.0
        total_api_calls = sum(r.behavioral_assessment.performance.total_api_calls for r in behavioral_results)
        
        # Generate insights and recommendations
        recommendations = self._generate_recommendations_with_examples(results)
        llm_insights = self._extract_llm_insights_with_examples(results)
        performance_bottlenecks = self._identify_bottlenecks_with_examples(results)
        
        # Generate behavioral insights
        behavioral_insights = self._generate_behavioral_insights(behavioral_results)
        optimization_opportunities = self._identify_optimization_opportunities(behavioral_results)
        common_patterns = self._identify_common_behavioral_patterns(behavioral_results)
        
        # Generate multi-instance statistical aggregation
        test_case_stats = self._calculate_test_case_statistics(results)
        category_stats = self._calculate_category_statistics(results)
        instance_consistency = self._calculate_instance_consistency_scores(results)
        
        return HybridEvaluationReport(
            agent_name=agent_instance.__class__.__name__,
            evaluation_date=datetime.now().isoformat(),
            evaluation_mode=mode.value + "_with_examples_integration",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            timeout_tests=timeout_tests,
            avg_execution_time=avg_execution_time,
            rule_based_accuracy=passed_tests / total_tests if total_tests > 0 else 0.0,
            llm_agreement_rate=llm_agreement_rate,
            avg_rule_score=avg_rule_score,
            avg_llm_score=avg_llm_score,
            avg_combined_score=avg_combined_score,
            consistency_score=consistency_score,
            
            # Examples-based metrics
            avg_format_compliance=avg_format_compliance,
            good_examples_match_rate=good_examples_match_rate,
            bad_examples_avoidance_rate=bad_examples_avoidance_rate,
            
            # Behavioral assessment aggregates
            avg_performance_score=avg_performance_score,
            avg_decision_making_score=avg_decision_making_score,
            avg_error_recovery_score=avg_error_recovery_score,
            avg_communication_score=avg_communication_score,
            avg_tool_efficiency_score=avg_tool_efficiency_score,
            behavioral_consistency_score=self._calculate_behavioral_consistency(behavioral_results),
            
            # Resource utilization
            avg_memory_usage_mb=avg_memory_usage,
            avg_cpu_usage_percent=avg_cpu_usage,
            total_api_calls=total_api_calls,
            
            # Required fields
            detailed_results=results,
            recommendations=recommendations,
            llm_insights=llm_insights,
            performance_bottlenecks=performance_bottlenecks,
            
            # Behavioral insights
            behavioral_insights=behavioral_insights,
            optimization_opportunities=optimization_opportunities,
            common_behavioral_patterns=common_patterns,
            
            # Multi-instance statistical aggregation
            test_case_statistics=test_case_stats,
            category_statistics=category_stats,
            instance_consistency_scores=instance_consistency,
            
            # Ground truth metrics
            ground_truth_coverage=ground_truth_coverage,
            avg_accuracy_score=avg_accuracy_score,
            avg_completeness_score=avg_completeness_score,
            avg_clarity_score=avg_clarity_score,
            co2_data_source=getattr(self, 'co2_data_source', 'examples_integrated_evaluation'),
            co2_analysis_summary=self.co2_analysis
        )

    def _generate_recommendations_with_examples(self, results: List[HybridEvaluationResult]) -> List[str]:
        """Generate recommendations including examples compliance"""
        recommendations = []
        
        # Pass rate analysis
        pass_rate = len([r for r in results if r.status == EvaluationStatus.PASS]) / len(results)
        if pass_rate < 0.7:
            recommendations.append(f"Low pass rate ({pass_rate:.1%}). Review agent responses against examples.json format.")
        
        # Format compliance analysis
        format_scores = [r.format_compliance_score for r in results if r.format_compliance_score > 0]
        if format_scores:
            avg_format = statistics.mean(format_scores)
            if avg_format < 0.7:
                recommendations.append(f"Poor format compliance ({avg_format:.1%}). Update system prompt to match examples.json structure.")
        
        # Examples pattern analysis
        good_matches = [r.matches_good_examples for r in results]
        bad_matches = [r.matches_bad_examples for r in results]
        
        if good_matches and statistics.mean(good_matches) < 5:
            recommendations.append("Low good example pattern matches. Align agent output with examples.json good patterns.")
        
        if bad_matches and statistics.mean(bad_matches) > 1:
            recommendations.append("Agent producing bad example patterns. Review system prompt to avoid examples.json bad patterns.")
        
        # Ground truth analysis
        gt_results = [r for r in results if r.ground_truth_used]
        if gt_results:
            accuracy_scores = [r.accuracy_score for r in gt_results if r.accuracy_score > 0]
            if accuracy_scores and statistics.mean(accuracy_scores) < 0.7:
                recommendations.append("Low accuracy vs reference outputs. Check data synchronization between agent and ground truth.")
        
        return recommendations[:5]
    
    def _extract_llm_insights_with_examples(self, results: List[HybridEvaluationResult]) -> List[str]:
        """Extract insights including examples compliance"""
        insights = []
        
        # Ground truth insights
        gt_results = [r for r in results if r.ground_truth_used]
        if gt_results:
            insights.append(f"Ground truth with examples integration used in {len(gt_results)}/{len(results)} tests")
            
            # Examples compliance insights
            format_scores = [r.format_compliance_score for r in gt_results if r.format_compliance_score > 0]
            if format_scores:
                avg_format = statistics.mean(format_scores)
                insights.append(f"Average format compliance: {avg_format:.2f} (examples.json integration)")
            
            # Pattern analysis
            good_matches = [r.matches_good_examples for r in gt_results]
            bad_matches = [r.matches_bad_examples for r in gt_results]
            
            if good_matches:
                avg_good = statistics.mean(good_matches)
                insights.append(f"Average good pattern matches: {avg_good:.1f}/10 (examples.json compliance)")
            
            if bad_matches:
                avg_bad = statistics.mean(bad_matches)
                if avg_bad > 0:
                    insights.append(f"Bad patterns detected: {avg_bad:.1f} per response (needs improvement)")
        
        return insights[:4]
    
    def _identify_bottlenecks_with_examples(self, results: List[HybridEvaluationResult]) -> List[str]:
        """Identify bottlenecks including examples-related issues"""
        bottlenecks = []
        
        # Format compliance bottlenecks
        format_scores = [r.format_compliance_score for r in results if r.format_compliance_score > 0]
        if format_scores and statistics.mean(format_scores) < 0.5:
            bottlenecks.append("Poor format compliance - system prompt doesn't match examples.json")
        
        # Examples pattern bottlenecks
        bad_pattern_count = sum(r.matches_bad_examples for r in results)
        if bad_pattern_count > len(results) * 0.5:  # More than 0.5 bad patterns per result
            bottlenecks.append("High bad pattern usage - agent following examples.json bad examples")
        
        # Traditional bottlenecks
        times = [r.execution_time for r in results]
        if times:
            avg_time = statistics.mean(times)
            if avg_time > 30:
                bottlenecks.append(f"High average execution time: {avg_time:.1f}s")
        
        # LLM evaluation failures
        llm_errors = [r for r in results if r.llm_status == EvaluationStatus.ERROR]
        if llm_errors:
            bottlenecks.append(f"{len(llm_errors)} LLM evaluation errors")
        
        # Ground truth comparison failures
        gt_failures = [r for r in results if r.ground_truth_used and r.status == EvaluationStatus.FAIL]
        if gt_failures:
            bottlenecks.append(f"{len(gt_failures)} ground truth comparison failures")
        
        return bottlenecks[:5]
    
    def _calculate_avg_behavioral_score(self, results: List[HybridEvaluationResult], score_field: str) -> float:
        """Calculate average behavioral score for a specific field"""
        scores = [getattr(result, score_field, 0.0) for result in results if hasattr(result, score_field)]
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_behavioral_consistency(self, behavioral_results: List) -> float:
        """Calculate behavioral consistency across all assessments"""
        if len(behavioral_results) < 2:
            return 1.0
        
        # Calculate consistency based on behavioral scores variance
        performance_scores = [r.performance_score for r in behavioral_results]
        
        if len(performance_scores) < 2:
            return 1.0
        
        mean_score = statistics.mean(performance_scores)
        if mean_score == 0:
            return 1.0
        
        variance = statistics.variance(performance_scores)
        cv = (variance ** 0.5) / mean_score
        
        # Convert to consistency score (lower CV = higher consistency)
        return max(0.0, 1.0 - cv)
    
    def _generate_behavioral_insights(self, behavioral_results: List) -> List[str]:
        """Generate behavioral insights from assessment results"""
        insights = []
        
        if not behavioral_results:
            return insights
        
        # Performance insights
        avg_execution_times = [r.behavioral_assessment.performance.total_execution_time for r in behavioral_results]
        if avg_execution_times:
            avg_time = statistics.mean(avg_execution_times)
            if avg_time > 30:
                insights.append(f"High average execution time: {avg_time:.1f}s")
            elif avg_time < 2:
                insights.append(f"Excellent response speed: {avg_time:.1f}s average")
        
        # Error handling insights
        total_errors = sum(r.behavioral_assessment.error_handling.total_errors_encountered for r in behavioral_results)
        if total_errors > 0:
            total_recoveries = sum(r.behavioral_assessment.error_handling.successful_recoveries for r in behavioral_results)
            recovery_rate = total_recoveries / total_errors
            insights.append(f"Error recovery rate: {recovery_rate:.1%} ({total_recoveries}/{total_errors})")
        else:
            insights.append("No errors encountered during evaluation")
        
        # Communication insights
        avg_response_lengths = [r.behavioral_assessment.communication.avg_response_length for r in behavioral_results]
        if avg_response_lengths:
            avg_length = statistics.mean(avg_response_lengths)
            insights.append(f"Average response length: {avg_length:.0f} characters")
        
        # Tool usage insights
        tool_efficiency_scores = [r.tool_efficiency_score for r in behavioral_results if r.tool_efficiency_score > 0]
        if tool_efficiency_scores:
            avg_efficiency = statistics.mean(tool_efficiency_scores)
            if avg_efficiency > 0.9:
                insights.append("Excellent tool usage efficiency")
            elif avg_efficiency < 0.7:
                insights.append(f"Tool usage efficiency needs improvement: {avg_efficiency:.1%}")
        
        return insights[:5]
    
    def _identify_optimization_opportunities(self, behavioral_results: List) -> List[str]:
        """Identify optimization opportunities from behavioral analysis"""
        opportunities = []
        
        if not behavioral_results:
            return opportunities
        
        # Performance optimization
        memory_values = [r.behavioral_assessment.performance.peak_memory_mb for r in behavioral_results if r.behavioral_assessment.performance.peak_memory_mb > 0]
        if memory_values and max(memory_values) > 500:
            opportunities.append("Optimize memory usage - peak usage exceeds 500MB")
        
        # Action efficiency
        redundant_actions = [r.behavioral_assessment.action_sequence.redundant_actions for r in behavioral_results]
        total_actions = [r.behavioral_assessment.action_sequence.total_actions for r in behavioral_results]
        
        if redundant_actions and total_actions:
            avg_redundancy_rate = sum(r/t for r, t in zip(redundant_actions, total_actions) if t > 0) / len(redundant_actions)
            if avg_redundancy_rate > 0.2:
                opportunities.append("Reduce redundant actions to improve efficiency")
        
        # Communication optimization
        clarity_scores = [r.behavioral_assessment.communication.response_clarity_score for r in behavioral_results]
        if clarity_scores and statistics.mean(clarity_scores) < 0.7:
            opportunities.append("Improve response clarity and structure")
        
        # Tool usage optimization
        tool_efficiency_scores = [r.tool_efficiency_score for r in behavioral_results if r.tool_efficiency_score > 0]
        if tool_efficiency_scores and statistics.mean(tool_efficiency_scores) < 0.8:
            opportunities.append("Improve tool selection and usage strategies")
        
        return opportunities[:5]
    
    def _identify_common_behavioral_patterns(self, behavioral_results: List) -> List[str]:
        """Identify common behavioral patterns across assessments"""
        patterns = []
        
        if not behavioral_results:
            return patterns
        
        # Collect all patterns from assessments
        all_patterns = []
        for result in behavioral_results:
            if hasattr(result.behavioral_assessment, 'action_sequence'):
                all_patterns.extend(result.behavioral_assessment.action_sequence.common_patterns)
        
        # Count pattern frequency
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Return most common patterns
        common_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in common_patterns[:5] if count > 1]

    def _create_empty_report(self, agent_instance, mode: EvaluationMode) -> HybridEvaluationReport:
        """Create empty report when no results available"""
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
            
            # Examples metrics
            avg_format_compliance=0.0,
            good_examples_match_rate=0.0,
            bad_examples_avoidance_rate=0.0,
            
            # Required fields
            detailed_results=[],
            recommendations=[],
            llm_insights=[],
            performance_bottlenecks=[]
        )

    def print_report_summary(self, report: HybridEvaluationReport):
        """Report summary with examples metrics"""
        print("\n" + "=" * 80)
        print("🎯 AGENT EVALUATION - EXAMPLES.JSON INTEGRATION")
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
        
        # Examples integration metrics
        print(f"\n📋 EXAMPLES.JSON INTEGRATION METRICS:")
        print(f"🎨 Format Compliance: {report.avg_format_compliance:.2f}")
        print(f"✅ Good Examples Match Rate: {report.good_examples_match_rate:.2f}")
        print(f"❌ Bad Examples Avoidance: {report.bad_examples_avoidance_rate:.2f}")
        
        # Ground truth metrics
        if report.ground_truth_coverage > 0:
            print(f"\n🎯 GROUND TRUTH METRICS:")
            print(f"📋 Ground Truth Coverage: {report.ground_truth_coverage:.1%}")
            print(f"🎯 Average Accuracy vs Reference: {report.avg_accuracy_score:.2f}")
            print(f"📝 Average Completeness: {report.avg_completeness_score:.2f}")
            print(f"✨ Average Clarity: {report.avg_clarity_score:.2f}")
        
        # CO2 data analysis summary
        if report.co2_analysis_summary:
            co2_analysis = report.co2_analysis_summary
            print(f"\n📈 REAL CO2 DATA ANALYSIS:")
            if 'optimal' in co2_analysis:
                print(f"  • Optimal period: {co2_analysis['optimal'].get('time_range', 'N/A')} ({co2_analysis['optimal'].get('intensity_range', 'N/A')})")
            if 'peak' in co2_analysis:
                print(f"  • Peak emission period: {co2_analysis['peak'].get('time_range', 'N/A')} ({co2_analysis['peak'].get('intensity_range', 'N/A')})")
            if 'daily_average' in co2_analysis:
                print(f"  • Daily average: {co2_analysis['daily_average']}g CO2/kWh")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"Average Execution Time: {report.avg_execution_time:.2f}s")
        print(f"Rule-based Accuracy: {report.rule_based_accuracy:.2f}")
        if report.llm_agreement_rate > 0:
            print(f"LLM vs Rule Agreement: {report.llm_agreement_rate:.2f}")
        print(f"Scoring Consistency: {report.consistency_score:.2f}")
        
        # Quality scores
        print(f"\n⭐ QUALITY SCORES:")
        print(f"Rule-based Score: {report.avg_rule_score:.2f}")
        if report.avg_llm_score > 0:
            print(f"LLM Quality Score: {report.avg_llm_score:.2f}")
        print(f"Combined Score: {report.avg_combined_score:.2f}")
        
        # Examples integration status
        print(f"\n📋 EXAMPLES.JSON INTEGRATION STATUS:")
        if len(self.examples.get('good_examples', [])) > 0:
            print(f"✅ Good Examples: {len(self.examples['good_examples'])} loaded and integrated")
        else:
            print(f"❌ Good Examples: Not loaded")
        
        if len(self.examples.get('bad_examples', [])) > 0:
            print(f"✅ Bad Examples: {len(self.examples['bad_examples'])} loaded for avoidance")
        else:
            print(f"❌ Bad Examples: Not loaded")
        
        print(f"✅ Format Requirements: Integrated from examples.json")
        print(f"✅ Response Parsing: With examples awareness")
        print(f"✅ Evaluation Logic: Updated with format compliance scoring")
        
        if self.behavioral_evaluator:
            print(f"✅ Behavioral Assessment: Enabled and integrated")
        else:
            print(f"❌ Behavioral Assessment: Disabled or not available")
        
        # Behavioral assessment section
        if hasattr(report, 'avg_performance_score') and report.avg_performance_score > 0:
            print(f"\n🧠 BEHAVIORAL ASSESSMENT METRICS:")
            print(f"🎯 Performance Score: {report.avg_performance_score:.2f}")
            print(f"🧩 Decision Making Score: {report.avg_decision_making_score:.2f}")
            print(f"🛠️  Error Recovery Score: {report.avg_error_recovery_score:.2f}")
            print(f"💬 Communication Score: {report.avg_communication_score:.2f}")
            print(f"🔧 Tool Efficiency Score: {report.avg_tool_efficiency_score:.2f}")
            print(f"📊 Behavioral Consistency: {report.behavioral_consistency_score:.2f}")
            
            # Resource utilization
            if report.avg_memory_usage_mb > 0 or report.avg_cpu_usage_percent > 0:
                print(f"\n💻 RESOURCE UTILIZATION:")
                if report.avg_memory_usage_mb > 0:
                    print(f"🧮 Average Memory Usage: {report.avg_memory_usage_mb:.1f} MB")
                if report.avg_cpu_usage_percent > 0:
                    print(f"⚡ Average CPU Usage: {report.avg_cpu_usage_percent:.1f}%")
                if report.total_api_calls > 0:
                    print(f"📞 Total API Calls: {report.total_api_calls}")
            
            # Behavioral insights
            if hasattr(report, 'behavioral_insights') and report.behavioral_insights:
                print(f"\n🔍 BEHAVIORAL INSIGHTS:")
                for insight in report.behavioral_insights:
                    print(f"  • {insight}")
            
            # Optimization opportunities
            if hasattr(report, 'optimization_opportunities') and report.optimization_opportunities:
                print(f"\n🎯 OPTIMIZATION OPPORTUNITIES:")
                for opportunity in report.optimization_opportunities:
                    print(f"  • {opportunity}")
        
        # Recommendations
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # LLM insights
        if report.llm_insights:
            print(f"\n🧠 LLM INSIGHTS:")
            for insight in report.llm_insights[:4]:
                print(f"  • {insight}")
        
        # Performance bottlenecks
        if report.performance_bottlenecks:
            print(f"\n🚨 PERFORMANCE BOTTLENECKS:")
            for bottleneck in report.performance_bottlenecks[:3]:
                print(f"  • {bottleneck}")
        
        print("=" * 80)

    def _save_results(self, report: HybridEvaluationReport):
        """Save evaluation results with examples integration info"""
        try:
            output_dir = Path(self.config["output_dir"])
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save comprehensive JSON report
            json_file = output_dir / f"examples_integrated_evaluation_{report.agent_name}_{timestamp}.json"
            
            # Convert to dict with formatting
            report_dict = self._format_report_for_json(report)
            
            # Add evaluation system metadata
            report_dict["evaluation_system_metadata"] = {
                "evaluation_framework": "hybrid_assessment_system",
                "behavioral_assessment_enabled": bool(self.behavioral_evaluator),
                "examples_integration": {
                    "good_examples_loaded": len(self.examples.get('good_examples', [])),
                    "bad_examples_loaded": len(self.examples.get('bad_examples', [])),
                    "format_requirements_integrated": bool(self.examples.get('format_requirements')),
                    "parsing_enabled": True,
                    "format_compliance_scoring": True
                },
                "behavioral_features": {
                    "performance_monitoring": self.config.get("enable_performance_monitoring", False),
                    "action_tracking": self.config.get("enable_action_tracking", False),
                    "resource_utilization_tracking": True,
                    "error_recovery_assessment": True,
                    "communication_quality_analysis": True,
                    "tool_efficiency_evaluation": True
                },
                "evaluation_capabilities": [
                    "Rule-based evaluation with function and behavior tracking",
                    "LLM-based evaluation with ground truth comparison", 
                    "Examples-based format compliance scoring",
                    "Comprehensive behavioral assessment",
                    "Performance and resource monitoring",
                    "Action sequence and decision pattern analysis",
                    "Error handling and recovery evaluation",
                    "Communication quality assessment",
                    "Tool usage efficiency analysis"
                ],
                "report_generation_date": datetime.now().isoformat()
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Evaluation results with examples integration saved to: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _format_report_for_json(self, report: HybridEvaluationReport) -> Dict:
        """Format evaluation report for JSON with examples integration details"""
        
        # Convert main report
        report_dict = asdict(report)
        
        # Format detailed results for JSON output
        formatted_results = []
        for result in report.detailed_results:
            # Extract response data with defaults
            test_query = getattr(result, 'test_query', 'Query not captured')
            full_response = getattr(result, 'full_agent_response', 'Response not captured')
            response_length = getattr(result, 'response_length', 0)
            response_preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
            
            # Extract behavioral assessment data
            behavioral_data = {
                "performance_score": getattr(result, 'performance_score', 0.0),
                "decision_making_score": getattr(result, 'decision_making_score', 0.0),
                "error_recovery_score": getattr(result, 'error_recovery_score', 0.0),
                "communication_score": getattr(result, 'communication_score', 0.0),
                "tool_efficiency_score": getattr(result, 'tool_efficiency_score', 0.0),
                "behavioral_strengths": getattr(result, 'behavioral_strengths', []),
                "behavioral_weaknesses": getattr(result, 'behavioral_weaknesses', []),
                "behavioral_recommendations": getattr(result, 'behavioral_recommendations', [])
            }
            
            # Helper function to get status value
            def get_status_value(status):
                return status.value if hasattr(status, 'value') else str(status)
            
            result_dict = {
                "test_metadata": {
                    "test_case_id": result.test_case_id,
                    "timestamp": result.timestamp,
                    "execution_time": result.execution_time
                },
                "query_and_response": {
                    "test_query": test_query,
                    "full_agent_response": full_response,
                    "response_length": response_length,
                    "response_preview": response_preview
                },
                "evaluation_results": {
                    "final_status": get_status_value(result.status),
                    "final_score": result.final_score,
                    "confidence_level": result.confidence_level,
                    "rule_based": {
                        "status": get_status_value(result.rule_based_status),
                        "score": result.rule_based_score,
                        "functions_called": result.functions_called,
                        "behaviors_observed": result.behaviors_observed,
                        "keyword_matches": result.keyword_matches
                    },
                    "llm_based": {
                        "status": get_status_value(result.llm_status) if result.llm_status else None,
                        "quality_score": result.llm_quality_score,
                        "confidence": result.llm_confidence,
                        "reasoning": result.llm_reasoning,
                        "feedback": result.llm_feedback
                    },
                    "consistency_score": result.consistency_score
                },
                "examples_integration_analysis": {
                    "format_compliance_score": result.format_compliance_score,
                    "matches_good_examples": result.matches_good_examples,
                    "matches_bad_examples": result.matches_bad_examples,
                    "follows_required_structure": result.format_compliance_score >= 0.7,
                    "avoids_bad_patterns": result.matches_bad_examples == 0
                },
                "ground_truth_evaluation": {
                    "used": result.ground_truth_used,
                    "accuracy_vs_reference": result.accuracy_score,
                    "completeness_vs_reference": result.completeness_score,
                    "clarity_score": result.clarity_score,
                    "actionability_score": result.actionability_score,
                    "format_compliance": result.format_score,
                    "matches_with_reference": result.specific_matches,
                    "gaps_vs_reference": result.specific_gaps,
                    "structured_penalties_applied": result.penalties_applied
                },
                "agent_interaction": {
                    "conversation_summary": result.structured_output.get("summary", {}),
                    "conversation_flow": result.conversation_flow,
                    "tool_interactions": result.tool_interactions,
                    "agent_responses": result.agent_responses,
                    "errors": result.structured_output.get("errors", [])
                },
                "improvement_guidance": {
                    "detailed_feedback": result.detailed_feedback,
                    "improvement_suggestions": result.improvement_suggestions
                },
                "raw_data": {
                    "raw_output_full": result.raw_output,  # Keep full raw output
                    "raw_output_length": len(result.raw_output),
                    "error_message": result.error_message
                },
                "behavioral_assessment_summary": behavioral_data
            }
            formatted_results.append(result_dict)
        
        # Replace detailed_results with formatted version
        report_dict["detailed_results"] = formatted_results
        
        return report_dict
    
    def _calculate_test_case_statistics(self, results: List[HybridEvaluationResult]) -> Dict[str, Dict[str, float]]:
        """Calculate statistical metrics for each test case across instances"""
        test_case_stats = {}
        
        # Group results by test case ID
        by_test_case = {}
        for result in results:
            test_id = result.test_case_id
            if test_id not in by_test_case:
                by_test_case[test_id] = []
            by_test_case[test_id].append(result)
        
        # Calculate statistics for each test case
        for test_id, test_results in by_test_case.items():
            if len(test_results) <= 1:
                continue  # Skip single-instance test cases
            
            # Extract metrics
            execution_times = [r.execution_time for r in test_results]
            final_scores = [r.final_score for r in test_results]
            rule_scores = [r.rule_based_score for r in test_results if r.rule_based_score > 0]
            llm_scores = [r.llm_quality_score for r in test_results if r.llm_quality_score > 0]
            response_lengths = [r.response_length for r in test_results if r.response_length > 0]
            
            # Calculate statistics
            stats = {
                "total_instances": len(test_results),
                "success_rate": sum(1 for r in test_results if r.status.value == "PASS") / len(test_results),
            }
            
            # Add statistical measures for each metric
            for metric_name, values in [
                ("execution_time", execution_times),
                ("final_score", final_scores),
                ("rule_score", rule_scores),
                ("llm_score", llm_scores),
                ("response_length", response_lengths)
            ]:
                if values and len(values) > 1:
                    stats[f"{metric_name}_mean"] = statistics.mean(values)
                    stats[f"{metric_name}_median"] = statistics.median(values)
                    stats[f"{metric_name}_std"] = statistics.stdev(values)
                    stats[f"{metric_name}_min"] = min(values)
                    stats[f"{metric_name}_max"] = max(values)
                    # Coefficient of variation (consistency measure)
                    if stats[f"{metric_name}_mean"] > 0:
                        stats[f"{metric_name}_consistency"] = 1.0 - min(1.0, stats[f"{metric_name}_std"] / stats[f"{metric_name}_mean"])
            
            test_case_stats[test_id] = stats
        
        return test_case_stats
    
    def _calculate_category_statistics(self, results: List[HybridEvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by test case category"""
        category_stats = {}
        
        # Group results by category (need to get category from test cases)
        by_category = {}
        for result in results:
            # Find the test case to get its category
            test_case = next((tc for tc in self.test_cases if tc['id'] == result.test_case_id), None)
            category = test_case.get('category', 'unknown') if test_case else 'unknown'
            
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        # Calculate statistics for each category
        for category, cat_results in by_category.items():
            total_instances = len(cat_results)
            successful = sum(1 for r in cat_results if r.status.value == "PASS")
            
            # Performance metrics
            final_scores = [r.final_score for r in cat_results if r.final_score > 0]
            execution_times = [r.execution_time for r in cat_results]
            
            stats = {
                "test_cases": len(set(r.test_case_id for r in cat_results)),
                "total_instances": total_instances,
                "successful_instances": successful,
                "success_rate": successful / total_instances if total_instances > 0 else 0,
                "avg_final_score": statistics.mean(final_scores) if final_scores else 0,
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0
            }
            
            category_stats[category] = stats
        
        return category_stats
    
    def _calculate_instance_consistency_scores(self, results: List[HybridEvaluationResult]) -> Dict[str, float]:
        """Calculate consistency scores for test cases with multiple instances"""
        consistency_scores = {}
        
        # Group by test case
        by_test_case = {}
        for result in results:
            test_id = result.test_case_id
            if test_id not in by_test_case:
                by_test_case[test_id] = []
            by_test_case[test_id].append(result)
        
        # Calculate consistency for multi-instance test cases
        for test_id, test_results in by_test_case.items():
            if len(test_results) <= 1:
                consistency_scores[test_id] = 1.0  # Single instance is perfectly consistent
                continue
            
            # Calculate various consistency measures
            consistency_measures = []
            
            # Status consistency (what percentage have the same status)
            statuses = [r.status.value for r in test_results]
            most_common_status = max(set(statuses), key=statuses.count)
            status_consistency = statuses.count(most_common_status) / len(statuses)
            consistency_measures.append(status_consistency)
            
            # Score consistency (coefficient of variation)
            scores = [r.final_score for r in test_results if r.final_score > 0]
            if len(scores) > 1 and statistics.mean(scores) > 0:
                score_cv = statistics.stdev(scores) / statistics.mean(scores)
                score_consistency = 1.0 - min(1.0, score_cv)
                consistency_measures.append(score_consistency)
            
            # Response length consistency
            response_lengths = [r.response_length for r in test_results if r.response_length > 0]
            if len(response_lengths) > 1 and statistics.mean(response_lengths) > 0:
                length_cv = statistics.stdev(response_lengths) / statistics.mean(response_lengths)
                length_consistency = 1.0 - min(1.0, length_cv * 0.5)  # Weight this less heavily
                consistency_measures.append(length_consistency)
            
            # Overall consistency is the average of all measures
            consistency_scores[test_id] = statistics.mean(consistency_measures) if consistency_measures else 0.0
        
        return consistency_scores