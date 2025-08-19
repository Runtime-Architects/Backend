"""
Behavioral Evaluator for comprehensive agent behavior assessment
Tracks performance metrics, action sequences, decision patterns, and more
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import threading

logger = logging.getLogger(__name__)

class BehaviorCategory(Enum):
    PERFORMANCE = "performance"
    DECISION_MAKING = "decision_making"
    ERROR_HANDLING = "error_handling"
    RESOURCE_USAGE = "resource_usage"
    COMMUNICATION = "communication"
    TOOL_USAGE = "tool_usage"
    LEARNING = "learning"

@dataclass
class PerformanceMetrics:
    """Detailed performance metrics tracking"""
    # Timing metrics
    total_execution_time: float = 0.0
    response_generation_time: float = 0.0
    tool_execution_time: float = 0.0
    decision_time: float = 0.0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    total_api_calls: int = 0
    total_tokens_used: int = 0
    
    # Efficiency metrics
    actions_per_second: float = 0.0
    successful_actions_ratio: float = 0.0
    
    # Latency metrics
    first_response_latency: float = 0.0
    avg_response_latency: float = 0.0
    max_response_latency: float = 0.0

@dataclass
class ActionSequenceAnalysis:
    """Analysis of agent's action sequences and decision patterns"""
    total_actions: int = 0
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    
    # Efficiency metrics
    redundant_actions: int = 0
    optimal_path_deviation: float = 0.0
    backtracking_instances: int = 0
    
    # Pattern analysis
    common_patterns: List[str] = field(default_factory=list)
    unique_strategies: List[str] = field(default_factory=list)
    
    # Decision quality
    correct_decisions: int = 0
    questionable_decisions: int = 0
    decision_confidence_scores: List[float] = field(default_factory=list)

@dataclass
class ErrorHandlingBehavior:
    """Assessment of how agent handles errors and failures"""
    total_errors_encountered: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Recovery behavior
    successful_recoveries: int = 0
    recovery_strategies: List[str] = field(default_factory=list)
    recovery_times: List[float] = field(default_factory=list)
    
    # Error patterns
    repeated_errors: int = 0
    escalation_behavior: List[str] = field(default_factory=list)
    graceful_degradation: bool = False

@dataclass
class CommunicationBehavior:
    """Assessment of agent's communication patterns and quality"""
    # Response characteristics
    total_responses: int = 0
    avg_response_length: float = 0.0
    response_clarity_score: float = 0.0
    response_helpfulness_score: float = 0.0
    
    # Conversation flow
    topic_consistency: float = 0.0
    context_maintenance: float = 0.0
    follow_up_quality: float = 0.0
    
    # User interaction patterns
    clarification_requests: int = 0
    proactive_suggestions: int = 0
    empathy_indicators: List[str] = field(default_factory=list)

@dataclass
class ToolUsageBehavior:
    """Assessment of how agent uses available tools and functions"""
    # Tool utilization
    tools_available: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_usage_efficiency: float = 0.0
    
    # Usage patterns
    tool_call_sequence: List[Dict[str, Any]] = field(default_factory=list)
    parallel_tool_usage: int = 0
    sequential_tool_usage: int = 0
    
    # Tool selection analysis
    appropriate_tool_selections: int = 0
    suboptimal_tool_selections: int = 0
    missing_tool_opportunities: int = 0

@dataclass
class ComprehensiveBehavioralAssessment:
    """Complete behavioral assessment combining all aspects"""
    test_case_id: str
    timestamp: str
    
    # Core assessments
    performance: PerformanceMetrics
    action_sequence: ActionSequenceAnalysis
    error_handling: ErrorHandlingBehavior
    communication: CommunicationBehavior
    tool_usage: ToolUsageBehavior
    
    # Overall scores
    overall_behavioral_score: float = 0.0
    behavioral_consistency_score: float = 0.0
    behavioral_efficiency_score: float = 0.0
    
    # Behavioral insights
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)

class BehavioralEvaluator:
    """Behavioral evaluator for comprehensive agent assessment"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitoring_active = False
        self.performance_monitor = None
        self.action_history = []
        self.error_history = []
        self.start_time = None
        
        # Monitoring settings
        self.sample_interval = config.get("performance_sample_interval", 0.1)  # 100ms
        self.max_history_size = config.get("max_history_size", 1000)
        
        logger.info("Behavioral evaluator initialized")
    
    async def start_evaluation(self, test_case_id: str) -> None:
        """Start behavioral monitoring for a test case"""
        self.test_case_id = test_case_id
        self.start_time = time.time()
        self.monitoring_active = True
        self.action_history = []
        self.error_history = []
        
        # Start performance monitoring in background
        self.performance_monitor = asyncio.create_task(self._monitor_performance())
        logger.info(f"Started behavioral evaluation for {test_case_id}")
    
    async def stop_evaluation(self) -> ComprehensiveBehavioralAssessment:
        """Stop monitoring and generate comprehensive assessment"""
        self.monitoring_active = False
        
        if self.performance_monitor:
            self.performance_monitor.cancel()
            try:
                await self.performance_monitor
            except asyncio.CancelledError:
                pass
        
        # Generate comprehensive assessment
        assessment = await self._generate_assessment()
        logger.info(f"Completed behavioral evaluation for {self.test_case_id}")
        return assessment
    
    async def record_action(self, action_type: str, action_data: Dict[str, Any]) -> None:
        """Record an agent action for analysis"""
        action_record = {
            "timestamp": time.time(),
            "type": action_type,
            "data": action_data,
            "execution_time": action_data.get("execution_time", 0.0)
        }
        
        self.action_history.append(action_record)
        
        # Maintain history size limit
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]
    
    async def record_error(self, error_type: str, error_details: Dict[str, Any]) -> None:
        """Record an error for behavioral analysis"""
        error_record = {
            "timestamp": time.time(),
            "type": error_type,
            "details": error_details,
            "recovery_attempted": False,
            "recovery_successful": False
        }
        
        self.error_history.append(error_record)
        logger.debug(f"Recorded error: {error_type}")
    
    async def record_recovery_attempt(self, strategy: str, success: bool) -> None:
        """Record error recovery attempt"""
        if self.error_history:
            latest_error = self.error_history[-1]
            latest_error["recovery_attempted"] = True
            latest_error["recovery_successful"] = success
            latest_error["recovery_strategy"] = strategy
            latest_error["recovery_time"] = time.time() - latest_error["timestamp"]
    
    async def _monitor_performance(self) -> None:
        """Background task to monitor system performance"""
        performance_samples = []
        
        try:
            process = psutil.Process()
            
            while self.monitoring_active:
                try:
                    # Collect performance sample
                    sample = {
                        "timestamp": time.time(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "cpu_percent": process.cpu_percent(interval=None)
                    }
                    performance_samples.append(sample)
                    
                    # Maintain sample size limit
                    if len(performance_samples) > 1000:
                        performance_samples = performance_samples[-800:]
                    
                    await asyncio.sleep(self.sample_interval)
                    
                except Exception as e:
                    logger.warning(f"Performance monitoring error: {e}")
                    await asyncio.sleep(self.sample_interval)
                    
        except asyncio.CancelledError:
            pass
        
        # Store final samples for analysis
        self.performance_samples = performance_samples
    
    async def _generate_assessment(self) -> ComprehensiveBehavioralAssessment:
        """Generate comprehensive behavioral assessment"""
        
        # Calculate performance metrics
        performance = await self._analyze_performance()
        
        # Analyze action sequences
        action_sequence = await self._analyze_action_sequences()
        
        # Assess error handling
        error_handling = await self._analyze_error_handling()
        
        # Evaluate communication behavior
        communication = await self._analyze_communication()
        
        # Assess tool usage
        tool_usage = await self._analyze_tool_usage()
        
        # Calculate overall scores
        overall_score = self._calculate_overall_behavioral_score(
            performance, action_sequence, error_handling, communication, tool_usage
        )
        
        # Generate insights
        strengths, weaknesses, recommendations = self._generate_behavioral_insights(
            performance, action_sequence, error_handling, communication, tool_usage
        )
        
        return ComprehensiveBehavioralAssessment(
            test_case_id=self.test_case_id,
            timestamp=datetime.now().isoformat(),
            performance=performance,
            action_sequence=action_sequence,
            error_handling=error_handling,
            communication=communication,
            tool_usage=tool_usage,
            overall_behavioral_score=overall_score,
            behavioral_consistency_score=self._calculate_consistency_score(),
            behavioral_efficiency_score=self._calculate_efficiency_score(),
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_recommendations=recommendations
        )
    
    async def _analyze_performance(self) -> PerformanceMetrics:
        """Analyze performance metrics from collected data"""
        
        if not hasattr(self, 'performance_samples') or not self.performance_samples:
            return PerformanceMetrics()
        
        samples = self.performance_samples
        
        # Calculate timing metrics
        total_time = time.time() - self.start_time if self.start_time else 0.0
        
        # Tool execution time from action history
        tool_actions = [a for a in self.action_history if a["type"] == "tool_call"]
        tool_execution_time = sum(a.get("execution_time", 0.0) for a in tool_actions)
        
        # Resource metrics
        memory_values = [s["memory_mb"] for s in samples if "memory_mb" in s]
        cpu_values = [s["cpu_percent"] for s in samples if "cpu_percent" in s]
        
        peak_memory = max(memory_values) if memory_values else 0.0
        avg_cpu = statistics.mean(cpu_values) if cpu_values else 0.0
        
        # Response latency from action history
        response_actions = [a for a in self.action_history if a["type"] == "response"]
        response_times = [a.get("execution_time", 0.0) for a in response_actions]
        
        return PerformanceMetrics(
            total_execution_time=total_time,
            tool_execution_time=tool_execution_time,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            total_api_calls=len([a for a in self.action_history if a["type"] == "api_call"]),
            actions_per_second=len(self.action_history) / max(total_time, 0.001),
            first_response_latency=response_times[0] if response_times else 0.0,
            avg_response_latency=statistics.mean(response_times) if response_times else 0.0,
            max_response_latency=max(response_times) if response_times else 0.0
        )
    
    async def _analyze_action_sequences(self) -> ActionSequenceAnalysis:
        """Analyze agent's action sequences and decision patterns"""
        
        if not self.action_history:
            return ActionSequenceAnalysis()
        
        # Basic metrics
        total_actions = len(self.action_history)
        
        # Identify decision points (where agent had multiple options)
        decision_points = []
        for i, action in enumerate(self.action_history):
            if action["type"] in ["tool_selection", "strategy_choice", "response_planning"]:
                decision_points.append({
                    "index": i,
                    "type": action["type"],
                    "options_considered": action.get("data", {}).get("options", []),
                    "choice_made": action.get("data", {}).get("choice"),
                    "confidence": action.get("data", {}).get("confidence", 0.0)
                })
        
        # Detect patterns
        action_types = [a["type"] for a in self.action_history]
        common_patterns = self._identify_action_patterns(action_types)
        
        # Analyze efficiency
        redundant_actions = self._count_redundant_actions()
        backtracking = self._detect_backtracking()
        
        return ActionSequenceAnalysis(
            total_actions=total_actions,
            action_sequence=self.action_history,
            decision_points=decision_points,
            redundant_actions=redundant_actions,
            backtracking_instances=backtracking,
            common_patterns=common_patterns,
            decision_confidence_scores=[dp["confidence"] for dp in decision_points]
        )
    
    async def _analyze_error_handling(self) -> ErrorHandlingBehavior:
        """Analyze how agent handles errors and failures"""
        
        if not self.error_history:
            return ErrorHandlingBehavior()
        
        total_errors = len(self.error_history)
        error_types = {}
        
        successful_recoveries = 0
        recovery_strategies = []
        recovery_times = []
        
        for error in self.error_history:
            error_type = error["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error.get("recovery_attempted"):
                if error.get("recovery_successful"):
                    successful_recoveries += 1
                    recovery_strategies.append(error.get("recovery_strategy", "unknown"))
                    recovery_times.append(error.get("recovery_time", 0.0))
        
        # Detect repeated errors
        repeated_errors = sum(1 for count in error_types.values() if count > 1)
        
        return ErrorHandlingBehavior(
            total_errors_encountered=total_errors,
            error_types=error_types,
            successful_recoveries=successful_recoveries,
            recovery_strategies=list(set(recovery_strategies)),
            recovery_times=recovery_times,
            repeated_errors=repeated_errors,
            graceful_degradation=self._assess_graceful_degradation()
        )
    
    async def _analyze_communication(self) -> CommunicationBehavior:
        """Analyze agent's communication patterns"""
        
        # Extract communication actions
        communication_actions = [
            a for a in self.action_history 
            if a["type"] in ["response", "message", "clarification"]
        ]
        
        if not communication_actions:
            return CommunicationBehavior()
        
        # Calculate response characteristics
        response_lengths = []
        for action in communication_actions:
            content = action.get("data", {}).get("content", "")
            response_lengths.append(len(content))
        
        avg_response_length = statistics.mean(response_lengths) if response_lengths else 0.0
        
        # Count specific interaction types
        clarification_requests = len([
            a for a in communication_actions 
            if "clarification" in a.get("data", {}).get("content", "").lower()
        ])
        
        return CommunicationBehavior(
            total_responses=len(communication_actions),
            avg_response_length=avg_response_length,
            clarification_requests=clarification_requests,
            response_clarity_score=self._calculate_clarity_score(communication_actions),
            response_helpfulness_score=self._calculate_helpfulness_score(communication_actions)
        )
    
    async def _analyze_tool_usage(self) -> ToolUsageBehavior:
        """Analyze how agent uses available tools"""
        
        tool_actions = [a for a in self.action_history if a["type"] == "tool_call"]
        
        if not tool_actions:
            return ToolUsageBehavior()
        
        # Extract tool usage patterns
        tools_used = list(set(
            action.get("data", {}).get("tool_name", "unknown") 
            for action in tool_actions
        ))
        
        # Analyze tool selection appropriateness
        appropriate_selections = 0
        suboptimal_selections = 0
        
        for action in tool_actions:
            appropriateness = action.get("data", {}).get("appropriateness_score", 0.5)
            if appropriateness >= 0.7:
                appropriate_selections += 1
            elif appropriateness < 0.4:
                suboptimal_selections += 1
        
        return ToolUsageBehavior(
            tools_used=tools_used,
            tool_call_sequence=tool_actions,
            appropriate_tool_selections=appropriate_selections,
            suboptimal_tool_selections=suboptimal_selections,
            tool_usage_efficiency=self._calculate_tool_efficiency(tool_actions)
        )
    
    def _calculate_overall_behavioral_score(self, performance: PerformanceMetrics,
                                          action_sequence: ActionSequenceAnalysis,
                                          error_handling: ErrorHandlingBehavior,
                                          communication: CommunicationBehavior,
                                          tool_usage: ToolUsageBehavior) -> float:
        """Calculate overall behavioral score"""
        
        # Component scores (0.0 to 1.0)
        performance_score = min(1.0, performance.actions_per_second / 10.0)  # Normalize to reasonable range
        
        efficiency_score = 1.0 - (action_sequence.redundant_actions / max(action_sequence.total_actions, 1))
        
        error_recovery_score = (
            error_handling.successful_recoveries / max(error_handling.total_errors_encountered, 1)
            if error_handling.total_errors_encountered > 0 else 1.0
        )
        
        communication_score = (communication.response_clarity_score + communication.response_helpfulness_score) / 2
        
        tool_efficiency_score = tool_usage.tool_usage_efficiency
        
        # Weighted combination
        weights = {
            "performance": 0.2,
            "efficiency": 0.25,
            "error_recovery": 0.2,
            "communication": 0.2,
            "tool_usage": 0.15
        }
        
        overall_score = (
            performance_score * weights["performance"] +
            efficiency_score * weights["efficiency"] +
            error_recovery_score * weights["error_recovery"] +
            communication_score * weights["communication"] +
            tool_efficiency_score * weights["tool_usage"]
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _calculate_consistency_score(self) -> float:
        """Calculate behavioral consistency across actions"""
        if len(self.action_history) < 3:
            return 1.0
        
        # Analyze response time consistency
        response_times = [
            a.get("execution_time", 0.0) for a in self.action_history 
            if a.get("execution_time", 0.0) > 0
        ]
        
        if len(response_times) < 2:
            return 1.0
        
        # Lower coefficient of variation indicates higher consistency
        mean_time = statistics.mean(response_times)
        if mean_time == 0:
            return 1.0
        
        std_time = statistics.stdev(response_times)
        cv = std_time / mean_time
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = max(0.0, 1.0 - cv)
        return consistency
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate behavioral efficiency score"""
        if not self.action_history:
            return 0.0
        
        total_actions = len(self.action_history)
        redundant_actions = self._count_redundant_actions()
        
        efficiency = 1.0 - (redundant_actions / total_actions)
        return max(0.0, efficiency)
    
    # Helper methods for pattern analysis
    def _identify_action_patterns(self, action_types: List[str]) -> List[str]:
        """Identify common action patterns"""
        patterns = []
        
        # Look for common sequences
        for i in range(len(action_types) - 2):
            pattern = " -> ".join(action_types[i:i+3])
            if action_types.count(action_types[i]) > 1:
                patterns.append(pattern)
        
        return list(set(patterns))[:5]  # Return top 5 unique patterns
    
    def _count_redundant_actions(self) -> int:
        """Count redundant actions in sequence"""
        redundant = 0
        
        for i in range(1, len(self.action_history)):
            current = self.action_history[i]
            previous = self.action_history[i-1]
            
            # Same action type with very similar data
            if (current["type"] == previous["type"] and
                current.get("data", {}).get("tool_name") == previous.get("data", {}).get("tool_name")):
                redundant += 1
        
        return redundant
    
    def _detect_backtracking(self) -> int:
        """Detect instances where agent backtracks or reverses decisions"""
        backtracking = 0
        
        # Look for patterns where agent reverses a previous action
        action_types = [a["type"] for a in self.action_history]
        
        for i in range(2, len(action_types)):
            if action_types[i] == action_types[i-2] and action_types[i] != action_types[i-1]:
                backtracking += 1
        
        return backtracking
    
    def _assess_graceful_degradation(self) -> bool:
        """Assess if agent degrades gracefully when facing errors"""
        if not self.error_history:
            return True
        
        # Check if agent continues to function after errors
        error_timestamps = [e["timestamp"] for e in self.error_history]
        
        for error_time in error_timestamps:
            # Check if agent continued to perform actions after error
            post_error_actions = [
                a for a in self.action_history 
                if a["timestamp"] > error_time
            ]
            
            if not post_error_actions:
                return False  # Agent stopped functioning after error
        
        return True
    
    def _calculate_clarity_score(self, communication_actions: List[Dict]) -> float:
        """Calculate response clarity score"""
        if not communication_actions:
            return 0.0
        
        clarity_indicators = 0
        total_responses = len(communication_actions)
        
        for action in communication_actions:
            content = action.get("data", {}).get("content", "").lower()
            
            # Check for clarity indicators
            if any(indicator in content for indicator in ["specific", "clear", "exactly", "precisely"]):
                clarity_indicators += 1
            
            # Check for structured format
            if any(format_indicator in content for format_indicator in ["•", "-", "1.", "step"]):
                clarity_indicators += 0.5
        
        return min(1.0, clarity_indicators / total_responses)
    
    def _calculate_helpfulness_score(self, communication_actions: List[Dict]) -> float:
        """Calculate response helpfulness score"""
        if not communication_actions:
            return 0.0
        
        helpfulness_indicators = 0
        total_responses = len(communication_actions)
        
        for action in communication_actions:
            content = action.get("data", {}).get("content", "").lower()
            
            # Check for helpfulness indicators
            helpful_phrases = ["recommend", "suggest", "help", "best", "optimal", "should"]
            if any(phrase in content for phrase in helpful_phrases):
                helpfulness_indicators += 1
        
        return min(1.0, helpfulness_indicators / total_responses)
    
    def _calculate_tool_efficiency(self, tool_actions: List[Dict]) -> float:
        """Calculate tool usage efficiency"""
        if not tool_actions:
            return 0.0
        
        successful_calls = len([
            a for a in tool_actions 
            if a.get("data", {}).get("success", False)
        ])
        
        return successful_calls / len(tool_actions)
    
    def _generate_behavioral_insights(self, performance: PerformanceMetrics,
                                    action_sequence: ActionSequenceAnalysis,
                                    error_handling: ErrorHandlingBehavior,
                                    communication: CommunicationBehavior,
                                    tool_usage: ToolUsageBehavior) -> Tuple[List[str], List[str], List[str]]:
        """Generate behavioral insights and recommendations"""
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Performance analysis
        if performance.actions_per_second > 5:
            strengths.append("High action execution rate")
        elif performance.actions_per_second < 1:
            weaknesses.append("Low action execution rate")
            recommendations.append("Optimize response generation speed")
        
        # Error handling analysis
        if error_handling.total_errors_encountered == 0:
            strengths.append("No errors encountered")
        elif error_handling.successful_recoveries > 0:
            success_rate = error_handling.successful_recoveries / error_handling.total_errors_encountered
            if success_rate > 0.8:
                strengths.append("Excellent error recovery capabilities")
            else:
                weaknesses.append("Poor error recovery performance")
                recommendations.append("Improve error handling and recovery strategies")
        
        # Communication analysis
        if communication.response_clarity_score > 0.8:
            strengths.append("Clear and well-structured responses")
        elif communication.response_clarity_score < 0.5:
            weaknesses.append("Unclear or poorly structured responses")
            recommendations.append("Improve response clarity and structure")
        
        # Tool usage analysis
        if tool_usage.tool_usage_efficiency > 0.9:
            strengths.append("Highly efficient tool usage")
        elif tool_usage.tool_usage_efficiency < 0.6:
            weaknesses.append("Inefficient tool usage")
            recommendations.append("Review tool selection strategies")
        
        # Action sequence analysis
        if action_sequence.redundant_actions > action_sequence.total_actions * 0.2:
            weaknesses.append("High number of redundant actions")
            recommendations.append("Optimize action planning to reduce redundancy")
        
        if action_sequence.backtracking_instances > 2:
            weaknesses.append("Frequent backtracking in decision making")
            recommendations.append("Improve initial decision making to reduce backtracking")
        
        return strengths[:5], weaknesses[:5], recommendations[:5]