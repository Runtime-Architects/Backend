"""
Result combiner for our hybrid agent evaluation
Intelligently combines rule-based and LLM evaluation results
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
        self.llm_override_enabled = config.get("llm_override_enabled", True)
        self.llm_override_threshold = config.get("llm_override_threshold", 0.8)
    
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
        """Apply LLM only logic"""
        if result.llm_status is not None:
            result.status = result.llm_status
            result.final_score = result.llm_quality_score
            result.confidence_level = self._determine_confidence(result.llm_quality_score, result.llm_confidence)
            
            result.detailed_feedback = [
                f"LLM evaluation: {result.llm_status.value}",
                f"Quality score: {result.llm_quality_score:.2f}",
                f"LLM confidence: {result.llm_confidence:.2f}",
                result.llm_reasoning
            ]
            
            if result.llm_feedback:
                result.detailed_feedback.extend(result.llm_feedback)
        else:
            # Fallback to rule-based if LLM failed
            result.status = result.rule_based_status
            result.final_score = result.rule_based_score
            result.confidence_level = "low"
            result.detailed_feedback = ["LLM evaluation failed, using rule-based fallback"]
        
        return result
    
    def _apply_hybrid_logic(self, result, test_case: Dict):
        """Apply robust hybrid logic"""
        
        # If LLM evaluation is missing, fall back to rule-based
        if result.llm_status is None:
            logger.warning(f"LLM evaluation missing for {result.test_case_id}, using rule-based")
            return self._apply_rule_based_only(result)
        
        # Calculate combined score
        combined_score = (
            result.rule_based_score * self.rule_weight +
            result.llm_quality_score * self.llm_weight
        )
        
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
        """Determine final status using hybrid logic with improved decision making"""
        
        rule_status = result.rule_based_status
        llm_status = result.llm_status
        
        # Ensure we're working with enum values
        if isinstance(rule_status, str):
            rule_status = EvaluationStatus(rule_status)
        if isinstance(llm_status, str):
            llm_status = EvaluationStatus(llm_status)
        
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
        return self._resolve_disagreement(result, test_case)
    
    def _resolve_disagreement(self, result, test_case: Dict) -> EvaluationStatus:
        """Resolve disagreement between rule-based and LLM evaluation with improved logic"""
        
        rule_status = result.rule_based_status
        llm_status = result.llm_status
        llm_confidence = result.llm_confidence
        rule_score = result.rule_based_score
        llm_quality = result.llm_quality_score
        
        # Ensure enum types
        if isinstance(rule_status, str):
            rule_status = EvaluationStatus(rule_status)
        if isinstance(llm_status, str):
            llm_status = EvaluationStatus(llm_status)
        
        logger.info(f"Resolving disagreement for {result.test_case_id}: "
                   f"Rule={rule_status.value}({rule_score:.2f}), LLM={llm_status.value}({llm_quality:.2f}, conf:{llm_confidence:.2f})")
        
        # High confidence LLM override (more restrictive)
        if (self.llm_override_enabled and 
            llm_confidence >= self.llm_override_threshold and
            llm_quality >= 0.8 and
            abs(llm_quality - rule_score) >= 0.3):  # Significant quality difference
            
            logger.info(f"LLM override applied (confidence: {llm_confidence:.2f}, quality gap: {abs(llm_quality - rule_score):.2f})")
            return llm_status
        
        # Strong rule-based evidence (both score and categorical check)
        if rule_score >= 0.8 and llm_quality < 0.6:
            logger.info(f"Rule-based override (strong rule score: {rule_score:.2f})")
            return rule_status
        
        # Category-specific logic with more balanced approach
        category = test_case.get("category", "")
        
        if category == "irrelevant_query":
            # For irrelevant queries, prefer LLM's contextual understanding
            if llm_confidence >= 0.6:
                return llm_status
        elif category in ["function_call", "tool_usage"]:
            # For function-focused tests, trust rule-based more
            if rule_score >= 0.6:
                return rule_status
        elif category in ["quality", "user_experience", "domain_expertise"]:
            # For quality tests, prefer LLM if confident
            if llm_confidence >= 0.7:
                return llm_status
        
        # Balanced resolution based on combined evidence
        rule_weight_adjusted = self.rule_weight
        llm_weight_adjusted = self.llm_weight
        
        # Adjust weights based on confidence and scores
        if llm_confidence < 0.6:
            rule_weight_adjusted += 0.2
            llm_weight_adjusted -= 0.2
        
        if rule_score >= 0.8:
            rule_weight_adjusted += 0.1
            llm_weight_adjusted -= 0.1
        
        # Weighted decision
        rule_evidence = rule_score * rule_weight_adjusted
        llm_evidence = llm_quality * llm_weight_adjusted * llm_confidence
        
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
        feedback.append(f"📊 Scores: Rule={result.rule_based_score:.2f}, "
                       f"LLM={result.llm_quality_score:.2f}, "
                       f"Combined={result.final_score:.2f}")
        
        # Function analysis
        if result.functions_called:
            feedback.append(f"🔧 Functions called: {', '.join(result.functions_called)}")
        else:
            feedback.append("🔧 No functions called")
        
        # Behavior analysis
        if result.behaviors_observed:
            feedback.append(f"🎭 Behaviors observed: {', '.join(result.behaviors_observed)}")
        
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
        
        # Add LLM-specific suggestions
        if hasattr(result, 'improvement_suggestions') and result.improvement_suggestions:
            suggestions.extend(result.improvement_suggestions[:2])  # Add top 2 LLM suggestions
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
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