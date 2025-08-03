"""
Result combiner for our hybrid agent evaluation
Intelligently combines rule-based and LLM evaluation results
FIXED: Status handling inconsistencies and disagreement resolution
"""

import logging
from typing import Dict, List, Any, Optional
from evaluation_strategies import EvaluationMode, EvaluationStatus

logger = logging.getLogger(__name__)

class ResultCombiner:
    """Combines rule-based and LLM evaluation results intelligently"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rule_weight = config.get("rule_weight", 0.4)
        self.llm_weight = config.get("llm_weight", 0.6)
        self.behavioral_weight = config.get("behavioral_weight", 0.2)
        self.llm_override_enabled = config.get("llm_override_enabled", True)
        self.llm_override_threshold = config.get("llm_override_threshold", 0.8)
        self.enable_behavioral_assessment = config.get("enable_behavioral_assessment", False)
    
    def combine_results(self, hybrid_result, test_case: Dict, mode: EvaluationMode):
        """
        Combine rule-based and LLM results into final evaluation
        Modifies the hybrid_result in place
        """
        
        if mode == EvaluationMode.RULE_BASED_ONLY:
            return self._apply_rule_based_only(hybrid_result)
        elif mode == EvaluationMode.LLM_ONLY:
            return self._apply_llm_only(hybrid_result)
        else:  # HYBRID mode
            return self._apply_hybrid_logic(hybrid_result, test_case)
    
    def _apply_rule_based_only(self, result):
        """Apply rule-based only logic"""
        result.status = result.rule_based_status
        result.final_score = result.rule_based_score
        result.confidence_level = self._determine_confidence(result.rule_based_score, 0.0)
        
        # Add rule-based feedback
        result.detailed_feedback = [
            f"Rule-based evaluation: {result.rule_based_status.value}",
            f"Functions called: {len(result.functions_called)}",
            f"Behaviors observed: {len(result.behaviors_observed)}",
            f"Keyword matches: {result.keyword_matches}"
        ]
        
        return result
    
    def _apply_llm_only(self, result):
        """Apply balanced LLM only logic with behavioral consideration"""
        if result.llm_status is not None:
            # Calculate comprehensive score even for LLM-only mode
            # This ensures behavioral metrics are considered if available
            if (self.enable_behavioral_assessment and 
                hasattr(result, 'behavioral_assessment') and 
                result.behavioral_assessment is not None):
                # Include behavioral component in LLM-only mode for fairness
                behavioral_score = self._calculate_behavioral_component_score(result)
                
                # Adjusted weights for LLM-only mode
                llm_weight = 0.8  # Primary weight to LLM
                behavioral_weight = 0.2  # Small behavioral consideration
                
                result.final_score = (
                    result.llm_quality_score * llm_weight +
                    behavioral_score * behavioral_weight
                )
            else:
                result.final_score = result.llm_quality_score
            
            # Re-evaluate status based on calibrated score
            if result.final_score >= self.config.get("llm_quality_threshold", 0.6):
                result.status = EvaluationStatus.PASS
            else:
                result.status = result.llm_status  # Keep original LLM decision
            
            result.confidence_level = self._determine_confidence(result.final_score, result.llm_confidence)
            
            # Generate comprehensive feedback
            result.detailed_feedback = [
                f"LLM evaluation: {result.llm_status.value}",
                f"Quality score: {result.llm_quality_score:.2f}",
                f"Final score (with adjustments): {result.final_score:.2f}",
                f"LLM confidence: {result.llm_confidence:.2f}",
                result.llm_reasoning
            ]
            
            # Add behavioral insights if available
            if (self.enable_behavioral_assessment and 
                hasattr(result, 'behavioral_assessment') and 
                result.behavioral_assessment is not None):
                if hasattr(result, 'behavioral_strengths') and result.behavioral_strengths:
                    result.detailed_feedback.append(f"Behavioral strengths: {', '.join(result.behavioral_strengths[:2])}")
            
            if result.llm_feedback:
                result.detailed_feedback.extend(result.llm_feedback)
                
            # Add semantic equivalence note if it helped
            if hasattr(result, 'accuracy_score') and result.accuracy_score > result.llm_quality_score:
                result.detailed_feedback.append("Note: Semantic equivalence bonus applied for valid alternative responses")
        else:
            # LLM evaluation failed - this is an error condition, not a fallback scenario
            result.status = EvaluationStatus.ERROR
            result.final_score = 0.0
            result.confidence_level = "error"
            result.detailed_feedback = ["LLM evaluation failed - cannot provide LLM-only evaluation without valid LLM results"]
        
        return result
    
    def _apply_hybrid_logic(self, result, test_case: Dict):
        """Apply robust hybrid logic"""
        
        # If LLM evaluation is missing, fall back to rule-based
        if result.llm_status is None:
            logger.warning(f"LLM evaluation missing for {result.test_case_id}, using rule-based")
            return self._apply_rule_based_only(result)
        
        # Calculate combined score with behavioral metrics
        combined_score = self._calculate_comprehensive_score(result)
        
        # Determine final status using multiple strategies
        final_status = self._determine_hybrid_status(result, test_case)
        
        # Set confidence level
        confidence_level = self._determine_confidence(combined_score, result.llm_confidence)
        
        # Generate comprehensive feedback
        detailed_feedback = self._generate_hybrid_feedback(result)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(result)
        
        # Update result
        result.status = final_status
        result.final_score = combined_score
        result.confidence_level = confidence_level
        result.detailed_feedback = detailed_feedback
        result.improvement_suggestions = improvement_suggestions
        
        # Log the final decision
        final_status_val = final_status.value if hasattr(final_status, 'value') else str(final_status)
        logger.debug(f"Final result for {result.test_case_id}: {final_status_val} (score: {combined_score:.2f})")
        
        return result
    
    def _determine_hybrid_status(self, result, test_case: Dict) -> EvaluationStatus:
        """FIXED: Determine final status using hybrid logic with improved decision making"""
        
        rule_status = result.rule_based_status
        llm_status = result.llm_status
        
        # FIXED: Ensure we're working with enum values consistently
        def normalize_status(status):
            if isinstance(status, str):
                try:
                    return EvaluationStatus(status)
                except ValueError:
                    # Handle case where string doesn't match enum
                    if status.upper() == "PASS":
                        return EvaluationStatus.PASS
                    elif status.upper() == "FAIL":
                        return EvaluationStatus.FAIL
                    elif status.upper() == "ERROR":
                        return EvaluationStatus.ERROR
                    elif status.upper() == "TIMEOUT":
                        return EvaluationStatus.TIMEOUT
                    else:
                        return EvaluationStatus.FAIL  # Default fallback
            return status
        
        rule_status = normalize_status(rule_status)
        llm_status = normalize_status(llm_status)
        
        # Handle error cases first
        if rule_status == EvaluationStatus.ERROR or llm_status == EvaluationStatus.ERROR:
            return EvaluationStatus.ERROR
        
        if rule_status == EvaluationStatus.TIMEOUT or llm_status == EvaluationStatus.TIMEOUT:
            return EvaluationStatus.TIMEOUT
        
        # Agreement case - both agree
        if rule_status == llm_status:
            logger.debug(f"Agreement: {rule_status.value} for {result.test_case_id}")
            return rule_status
        
        # Disagreement case - need to resolve
        logger.info(f"Disagreement for {result.test_case_id}: Rule={rule_status.value}, LLM={llm_status.value}")
        return self._resolve_disagreement_fixed(result, test_case, rule_status, llm_status)
    
    def _resolve_disagreement_fixed(self, result, test_case: Dict, rule_status: EvaluationStatus, llm_status: EvaluationStatus) -> EvaluationStatus:
        """FIXED: Resolve disagreement with better logic for LLM-only mode"""
        
        llm_confidence = result.llm_confidence
        rule_score = result.rule_based_score
        llm_quality = result.llm_quality_score
        
        logger.info(f"Resolving disagreement for {result.test_case_id}: "
                   f"Rule={rule_status.value}({rule_score:.2f}), LLM={llm_status.value}({llm_quality:.2f}, conf:{llm_confidence:.2f})")
        
        # FIXED: For LLM-only evaluation, be more lenient with LLM decisions
        # If LLM has reasonable confidence and quality, trust it more
        if (llm_confidence >= 0.6 and llm_quality >= 0.3):  # Lowered thresholds
            logger.info(f"LLM decision accepted (confidence: {llm_confidence:.2f}, quality: {llm_quality:.2f})")
            return llm_status
        
        # FIXED: Don't be too harsh on rule-based when LLM has low confidence
        if llm_confidence < 0.5 and rule_score >= 0.4:
            logger.info(f"Rule-based decision accepted due to low LLM confidence")
            return rule_status
        
        # FIXED: More balanced resolution
        # Weight both evaluations more evenly
        rule_weight = 0.4
        llm_weight = 0.6
        
        # Adjust weights based on confidence and quality
        if llm_confidence < 0.6:
            rule_weight += 0.2
            llm_weight -= 0.2
        
        if llm_quality >= 0.6:
            llm_weight += 0.1
            rule_weight -= 0.1
        
        # Score-based decision with adjusted weights
        rule_evidence = rule_score * rule_weight
        llm_evidence = llm_quality * llm_weight * max(0.5, llm_confidence)  # Don't let very low confidence destroy LLM evidence
        
        if rule_evidence > llm_evidence:
            logger.info(f"Weighted decision: Rule wins ({rule_evidence:.3f} > {llm_evidence:.3f})")
            return rule_status
        else:
            logger.info(f"Weighted decision: LLM wins ({llm_evidence:.3f} > {rule_evidence:.3f})")
            return llm_status
    
    def _determine_confidence(self, combined_score: float, llm_confidence: float) -> str:
        """Determine confidence level for the evaluation"""
        
        # Factor in both score and LLM confidence
        if llm_confidence > 0:
            avg_confidence = (combined_score + llm_confidence) / 2
        else:
            avg_confidence = combined_score
        
        if avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_comprehensive_score(self, result) -> float:
        """Calculate comprehensive score including behavioral and performance metrics"""
        # Base hybrid score
        base_score = (
            result.rule_based_score * self.rule_weight +
            result.llm_quality_score * self.llm_weight
        )
        
        # Add behavioral assessment if available
        if (self.enable_behavioral_assessment and 
            hasattr(result, 'behavioral_assessment') and 
            result.behavioral_assessment is not None):
            
            # Calculate behavioral component score
            behavioral_score = self._calculate_behavioral_component_score(result)
            
            # Normalize weights to include behavioral component
            total_weight = self.rule_weight + self.llm_weight + self.behavioral_weight
            normalized_rule_weight = self.rule_weight / total_weight
            normalized_llm_weight = self.llm_weight / total_weight
            normalized_behavioral_weight = self.behavioral_weight / total_weight
            
            # Recalculate with behavioral component
            comprehensive_score = (
                result.rule_based_score * normalized_rule_weight +
                result.llm_quality_score * normalized_llm_weight +
                behavioral_score * normalized_behavioral_weight
            )
            
            logger.debug(f"Comprehensive score for {result.test_case_id}: "
                        f"base={base_score:.3f}, behavioral={behavioral_score:.3f}, "
                        f"final={comprehensive_score:.3f}")
            
            return comprehensive_score
        
        return base_score
    
    def _calculate_behavioral_component_score(self, result) -> float:
        """Calculate overall behavioral component score from individual metrics"""
        scores = []
        
        # Include all behavioral scores with equal weighting
        if hasattr(result, 'performance_score') and result.performance_score > 0:
            scores.append(result.performance_score)
        
        if hasattr(result, 'decision_making_score') and result.decision_making_score > 0:
            scores.append(result.decision_making_score)
            
        if hasattr(result, 'error_recovery_score') and result.error_recovery_score > 0:
            scores.append(result.error_recovery_score)
            
        if hasattr(result, 'communication_score') and result.communication_score > 0:
            scores.append(result.communication_score)
            
        if hasattr(result, 'tool_efficiency_score') and result.tool_efficiency_score > 0:
            scores.append(result.tool_efficiency_score)
        
        # Return average of available behavioral scores
        if scores:
            return sum(scores) / len(scores)
        else:
            # Fallback to consistency score if no behavioral metrics available
            return result.consistency_score
    
    def _generate_hybrid_feedback(self, result) -> List[str]:
        """Generate comprehensive feedback combining both evaluations"""
        
        feedback = []
        
        # Status comparison
        if result.rule_based_status == result.llm_status:
            feedback.append(f"✅ Rule-based and LLM evaluations agree: {result.status.value}")
        else:
            feedback.append(f"⚠️ Disagreement: Rule-based={result.rule_based_status.value}, "
                          f"LLM={result.llm_status.value}, Final={result.status.value}")
        
        # Score information
        score_info = f"📊 Scores: Rule={result.rule_based_score:.2f}, LLM={result.llm_quality_score:.2f}"
        
        # Add behavioral scores if available
        if (self.enable_behavioral_assessment and 
            hasattr(result, 'behavioral_assessment') and 
            result.behavioral_assessment is not None):
            behavioral_scores = []
            if hasattr(result, 'performance_score') and result.performance_score > 0:
                behavioral_scores.append(f"Performance={result.performance_score:.2f}")
            if hasattr(result, 'decision_making_score') and result.decision_making_score > 0:
                behavioral_scores.append(f"Decision={result.decision_making_score:.2f}")
            if hasattr(result, 'error_recovery_score') and result.error_recovery_score > 0:
                behavioral_scores.append(f"Recovery={result.error_recovery_score:.2f}")
            if hasattr(result, 'communication_score') and result.communication_score > 0:
                behavioral_scores.append(f"Communication={result.communication_score:.2f}")
            if hasattr(result, 'tool_efficiency_score') and result.tool_efficiency_score > 0:
                behavioral_scores.append(f"ToolEfficiency={result.tool_efficiency_score:.2f}")
            
            if behavioral_scores:
                score_info += f", Behavioral=({', '.join(behavioral_scores)})"
        
        score_info += f", Combined={result.final_score:.2f}"
        feedback.append(score_info)
        
        # Function analysis
        if result.functions_called:
            feedback.append(f"🔧 Functions called: {', '.join(result.functions_called)}")
        else:
            feedback.append("🔧 No functions called")
        
        # Behavior analysis
        if result.behaviors_observed:
            feedback.append(f"🎭 Behaviors observed: {', '.join(result.behaviors_observed)}")
        
        # Behavioral assessment insights
        if (self.enable_behavioral_assessment and 
            hasattr(result, 'behavioral_assessment') and 
            result.behavioral_assessment is not None):
            
            # Performance insights
            if hasattr(result.behavioral_assessment, 'performance'):
                perf = result.behavioral_assessment.performance
                if perf.total_execution_time > 0:
                    feedback.append(f"⏱️ Performance: {perf.total_execution_time:.2f}s execution, "
                                  f"{perf.total_api_calls} API calls")
            
            # Behavioral strengths and weaknesses
            if hasattr(result, 'behavioral_strengths') and result.behavioral_strengths:
                feedback.append(f"💪 Strengths: {', '.join(result.behavioral_strengths[:3])}")
            
            if hasattr(result, 'behavioral_weaknesses') and result.behavioral_weaknesses:
                feedback.append(f"⚠️ Areas for improvement: {', '.join(result.behavioral_weaknesses[:3])}")
        
        # Keyword matching
        if hasattr(result, 'keyword_matches'):
            feedback.append(f"🔍 Keyword matches: {result.keyword_matches}")
        
        # LLM reasoning
        if result.llm_reasoning:
            feedback.append(f"🧠 LLM reasoning: {result.llm_reasoning[:200]}...")
        
        # LLM feedback
        if result.llm_feedback:
            feedback.append("💬 LLM feedback:")
            feedback.extend([f"  • {fb}" for fb in result.llm_feedback[:3]])
        
        return feedback
    
    def _generate_improvement_suggestions(self, result) -> List[str]:
        """Generate improvement suggestions based on combined analysis"""
        
        suggestions = []
        
        # Low score suggestions
        if result.final_score < 0.5:
            suggestions.append("Overall score is low - review agent capabilities and test requirements")
        
        # Function-related suggestions
        if not result.functions_called and result.rule_based_score < 0.6:
            suggestions.append("No functions called - ensure agent knows when and how to use available tools")
        
        # Quality-related suggestions
        if result.llm_quality_score < 0.6:
            suggestions.append("LLM indicates low response quality - improve response completeness and clarity")
        
        # Consistency suggestions
        if result.rule_based_status != result.llm_status:
            suggestions.append("Rule-based and LLM evaluations disagree - review evaluation criteria alignment")
        
        # Confidence suggestions
        if result.llm_confidence < 0.6:
            suggestions.append("LLM has low confidence - response may be ambiguous or incomplete")
        
        # Behavioral-based suggestions
        if (self.enable_behavioral_assessment and 
            hasattr(result, 'behavioral_assessment') and 
            result.behavioral_assessment is not None):
            
            # Performance suggestions
            if hasattr(result, 'performance_score') and result.performance_score < 0.6:
                suggestions.append("Performance score is low - optimize execution time and resource usage")
            
            # Decision making suggestions
            if hasattr(result, 'decision_making_score') and result.decision_making_score < 0.6:
                suggestions.append("Decision making needs improvement - review action selection logic")
            
            # Error recovery suggestions
            if hasattr(result, 'error_recovery_score') and result.error_recovery_score < 0.6:
                suggestions.append("Error recovery is weak - implement better error handling strategies")
            
            # Communication suggestions
            if hasattr(result, 'communication_score') and result.communication_score < 0.6:
                suggestions.append("Communication quality is low - improve response clarity and helpfulness")
            
            # Tool efficiency suggestions
            if hasattr(result, 'tool_efficiency_score') and result.tool_efficiency_score < 0.6:
                suggestions.append("Tool usage is inefficient - optimize function call patterns")
            
            # Add behavioral recommendations if available
            if hasattr(result, 'behavioral_recommendations') and result.behavioral_recommendations:
                suggestions.extend(result.behavioral_recommendations[:2])
        
        # Add LLM-specific suggestions
        if hasattr(result, 'improvement_suggestions') and result.improvement_suggestions:
            suggestions.extend(result.improvement_suggestions[:2])  # Add top 2 LLM suggestions
        
        return suggestions[:7]  # Limit to top 7 suggestions (increased to accommodate behavioral)
    
    def calculate_agreement_metrics(self, results: List) -> Dict[str, float]:
        """Calculate agreement metrics between rule-based and LLM evaluations"""
        
        if not results:
            return {}
        
        # Filter results where both evaluations ran
        both_ran = [r for r in results if r.llm_status is not None]
        
        if not both_ran:
            return {"agreement_rate": 0.0, "samples": 0}
        
        # Status agreement
        status_agreements = [r for r in both_ran if r.rule_based_status == r.llm_status]
        status_agreement_rate = len(status_agreements) / len(both_ran)
        
        # Score correlation (simplified)
        score_differences = [abs(r.rule_based_score - r.llm_quality_score) for r in both_ran]
        avg_score_difference = sum(score_differences) / len(score_differences)
        score_agreement = max(0, 1 - avg_score_difference)  # Convert difference to agreement
        
        # High confidence LLM overrides
        overrides = [r for r in both_ran if 
                    r.rule_based_status != r.llm_status and 
                    r.status == r.llm_status and 
                    r.llm_confidence >= self.llm_override_threshold]
        override_rate = len(overrides) / len(both_ran)
        
        return {
            "agreement_rate": status_agreement_rate,
            "score_agreement": score_agreement,
            "override_rate": override_rate,
            "samples": len(both_ran),
            "avg_score_difference": avg_score_difference
        }