# Hybrid Agent Evaluation System

A evaluation framework for testing AI agents that provide carbon emission recommendations based on real EirGrid data. The system uses a **hybrid approach** combining rule-based validation with LLM-as-Judge evaluation to ensure accurate, helpful, and well-formatted agent responses.

## Table of Contents

- [Overview](#overview)
- [Evaluation Modes](#evaluation-modes)
- [System Architecture](#system-architecture)
- [Evaluation Flow](#evaluation-flow)
- [Scoring System](#scoring-system)
- [Configuration](#configuration)
- [Data Sources](#data-sources)
- [Quick Start](#quick-start)
- [Understanding Results](#understanding-results)
- [Advanced Usage](#advanced-usage)
- [Recent Updates](#recent-updates)

## Overview

This evaluation system tests AI agents that help users optimize their electricity usage in Ireland by providing carbon emission recommendations. The **calibrated evaluation system** ensures fair, balanced, and consistent assessment across multiple dimensions:

- **Functional Correctness**: Does the agent call the right tools? (Fixed function detection)
- **Data Accuracy**: Are the CO2 values and time recommendations correct? (Ground truth comparison)
- **Response Quality**: Is the response helpful, clear, and well-structured? (LLM-based assessment)
- **Format Compliance**: Does the response follow expected formatting guidelines? (Semantic equivalence support)
- **Behavioral Assessment**: Does the agent demonstrate proper decision-making and tool usage? (20% weight in all modes)

## Evaluation Modes

The system supports **three calibrated evaluation modes** with consistent thresholds and balanced scoring:

### 🔄 Hybrid Mode
- **Weights**: Rule-based (30%) + LLM-based (50%) + Behavioral (20%)
- **Threshold**: 0.6 score to pass
- **Best for**: Comprehensive balanced evaluation

### 🤖 LLM-Only Mode
- **Weights**: LLM-based (80%) + Behavioral (20%)
- **Features**: Semantic equivalence bonus, reduced penalties
- **Best for**: Content quality and accuracy focus

### 📋 Rule-Based Only Mode
- **Weights**: Rule-based (80%) + Behavioral (20%)
- **Features**: Fixed function detection, strict checking
- **Best for**: Tool usage and format validation

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Agent Evaluation System                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐  │
│  │   Test Cases    │    │  Compressed      │    │   Examples  │  │
│  │   (.json)       │    │  CO2 Data        │    │   (.json)   │  │
│  │                 │    │  (EirGrid)       │    │             │  │
│  └─────────────────┘    └──────────────────┘    └─────────────┘  │
│           │                       │                      │       │
│           ▼                       ▼                      ▼       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Hybrid Agent Evaluator                         │ │
│  │                                                             │ │
│  │  ┌─────────────────┐           ┌─────────────────────────┐  │ │
│  │  │  Rule-Based     │           │    LLM-as-Judge         │  │ │
│  │  │  Evaluation     │           │    Evaluation           │  │ │
│  │  │                 │           │                         │  │ │
│  │  │ • Function calls│           │ • Content analysis      │  │ │
│  │  │ • Keyword match │           │ • Ground truth compare  │  │ │
│  │  │ • Behavior check│           │ • Quality assessment    │  │ │
│  │  └─────────────────┘           └─────────────────────────┘  │ │
│  │           │                               │                 │ │
│  │           └───────────────┬───────────────┘                 │ │
│  │                           ▼                                 │ │
│  │              ┌─────────────────────────┐                    │ │
│  │              │    Score Combination    │                    │ │
│  │              │    & Final Result       │                    │ │
│  │              └─────────────────────────┘                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Evaluation Report                           │ │
│  │          (JSON with detailed metrics & feedback)            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Evaluation Flow

### 1. Data Preparation Phase

```python
# 1. Load compressed CO2 data (EirGrid sources)
compressed_data = CO2DataCompressor.prepare_for_evaluation("./data")

# 2. Generate ground truth test cases
test_cases = GroundTruthGenerator.generate_for_evaluation(compressed_data)

# 3. Load configuration and examples
config = load_config("streamlined_config.json")
examples = load_examples("examples.json")
```

### 2. Agent Execution Phase

```python
# For each test case:
for test_case in test_cases:
    # Execute agent with query
    agent_response = await agent.run(test_case.query)
    
    # Parse and structure response
    structured_output = AgentOutputParser.parse_agent_output(agent_response)
```

### 3. Evaluation Phase

#### Rule-Based Evaluation
```python
rule_result = await rule_strategy.evaluate(test_case, full_response)
# Checks:
# - Expected functions called? ✓
# - Required keywords present? ✓  
# - Expected behaviors observed? ✓
```

#### LLM-as-Judge Evaluation
```python
llm_result = await llm_strategy.evaluate(
    test_case, 
    full_response, 
    rule_result, 
    compressed_co2_data  # ← Critical for data validation
)
# Evaluates:
# - Content accuracy vs ground truth
# - Response completeness
# - Format compliance
# - Actionability and clarity
```

### 4. Score Combination

```python
final_score = (rule_weight * rule_score) + (llm_weight * llm_score)
# Default weights: rule_weight=0.3, llm_weight=0.7
```

## Key Files

### Core Evaluation System
| File | Purpose |
|------|---------|
| `hybrid_agent_evaluator.py` | Main evaluation orchestrator - coordinates rule-based and LLM evaluation strategies |
| `hybrid_evaluation_runner.py` | Interactive runner script with multi-instance support - entry point for running evaluations |
| `evaluation_strategies.py` | Base classes and implementations for rule-based and LLM-based evaluation methods |
| `llm_judge.py` | LLM-as-Judge implementation for quality assessment and ground truth comparison |
| `behavioral_evaluator.py` | Evaluates agent decision-making patterns and tool usage behaviors |
| `result_combiner.py` | Combines and weights scores from different evaluation strategies |

### Data Management
| File | Purpose |
|------|---------|
| `co2_data_compressor.py` | Compresses raw EirGrid CO2 data into evaluation-ready format |
| `ground_truth_generator.py` | Generates reference answers and scoring criteria for test cases with date extraction |
| `examples.json` | Template responses showing good/bad answer formats for evaluation |

### Agent Integration  
| File | Purpose |
|------|---------|
| `kamal_agent_integration.py` | Integration wrapper for external agent implementations |

### CO2 Analysis Tools
| Directory/File | Purpose |
|----------------|---------|
| `co2_analysis_tool/` | Package for CO2 data analysis and visualization |
| `co2_analysis_tool/co2_analysis.py` | Core CO2 emission analysis functions with dynamic data loading |
| `co2_analysis_tool/co2_analysis_util.py` | Utility functions for CO2 data processing |
| `co2_analysis_tool/co2_plot.py` | Visualization tools for CO2 emission patterns |

### Web Scraping Tools
| Directory/File | Purpose |
|----------------|---------|
| `scraper_tools/` | Package for EirGrid data collection |
| `scraper_tools/csv_downloader.py` | Downloads CO2 data from EirGrid in CSV format |
| `scraper_tools/unified_downloader.py` | Unified interface for downloading various data formats |
| `scraper_tools/run_eirgrid_downloader.py` | Script to execute EirGrid data collection |
| `scraper_tools/comprehensive_test.py` | Comprehensive testing of scraper functionality |

### Configuration Files
| File | Purpose |
|------|---------|
| `test_configs/compressed_data_test_cases.json` | Test cases with ground truth data and scoring criteria |
| `test_configs/streamlined_config.json` | Main configuration for evaluation system |
| `test_configs/generated_test_cases_with_ground_truth_*.json` | Dynamically generated test cases with real data |

### Core Dependencies
| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies for the evaluation system |

### Data Directories
| Directory | Purpose |
|-----------|---------|
| `data/` | Stores compressed CO2 data files and raw EirGrid downloads |
| `evaluation_results/` | Contains detailed JSON reports from evaluation runs |


## Scoring System

### Score Components (Calibrated)

| Component | Hybrid Mode | LLM-Only Mode | Rule-Only Mode | Range |
|-----------|-------------|---------------|----------------|-------|
| **Rule Score** | 30% | 0% | 80% | 0.0 - 1.0 |
| **LLM Score** | 50% | 80% | 0% | 0.0 - 1.0 |
| **Behavioral Score** | 20% | 20% | 20% | 0.0 - 1.0 |
| **Pass Threshold** | 0.6 | 0.6 | 0.6 | Consistent |

### Detailed Scoring Breakdown

#### Rule-Based Scoring
```python
rule_score = (
    function_weight * function_score +    # Did agent call get_emission_analysis?
    keyword_weight * keyword_score +     # Contains required keywords?
    behavior_weight * behavior_score     # Shows expected behaviors?
)
```

#### LLM-Based Scoring
```python
llm_score = (
    accuracy_weight * accuracy_score +        # Factually correct vs real data?
    completeness_weight * completeness_score + # Covers required content?
    clarity_weight * clarity_score +          # Clear and well-structured?
    actionability_weight * actionability_score + # Provides actionable advice?
    format_weight * format_score              # Follows formatting guidelines?
)
```

### Pass/Fail Criteria (Calibrated)

```python
# Test passes if:
final_score >= 0.6  # Consistent across all modes

# Additional criteria:
- No critical errors (timeouts, exceptions)
- Agent response contains actual content (not just tool output)
- Required functions were called (enforced in rule-based)
- Semantic equivalence bonus for valid alternatives (LLM-only)
- One minor issue allowed if score >= 0.54 (rule-based flexibility)
```

## Configuration

### Available Configurations

#### Main Configuration (`streamlined_config.json`)
```json
{
  "evaluation_mode": "hybrid",           // Default mode: hybrid, llm_only, rule_based_only
  "auto_generate_ground_truth": true,    // Automatic ground truth generation
  "llm_evaluation_enabled": true,        // Enable LLM-as-Judge evaluation
  "llm_quality_threshold": 0.5,          // Minimum LLM quality threshold
  "llm_confidence_threshold": 0.4,       // Minimum confidence threshold
  "timeout_seconds": 60,                 // Per-test timeout
  
  // Score weighting
  "rule_weight": 0.3,                    // Rule-based evaluation weight
  "llm_weight": 0.7,                     // LLM evaluation weight
  "function_weight": 0.4,                // Function call validation weight
  "keyword_weight": 0.2,                 // Keyword matching weight
  "behavior_weight": 0.4,                // Behavioral assessment weight
  
  // Multi-instance evaluation
  "auto_run_scraper": true,              // Automatic scraper integration
  "use_compressed_data": true,           // Use compressed CO2 data
  "relaxed_validation": true,            // Allow minor validation issues
  
  // Performance settings
  "consistency_threshold": 0.8,          // Score consistency requirement
  "max_llm_tokens": 4000,               // Maximum LLM response tokens
  "llm_timeout": 90,                     // LLM evaluation timeout
  "parallel_llm_calls": false,           // Parallel processing (disabled for stability)
  
  // Ground truth settings
  "ground_truth_weight": 0.8,            // Ground truth comparison weight
  "enforce_scoring_consistency": true,    // Strict scoring validation
  "structured_penalty_system": true      // Use structured penalty system
}
```

### Test Cases (`test_configs/compressed_data_test_cases.json`)

```json
{
  "metadata": {
    "co2_data_source": "compressed_eirgrid_data",
    "compressed_data_points": 46,
    "absolute_min": 199,                 // Actual minimum CO2 value
    "absolute_max": 260,                 // Actual maximum CO2 value
    "daily_average": 221                 // Actual daily average
  },
  "test_cases": [
    {
      "id": "carbon_001_real_appliances",
      "query": "What is the best time to use my appliances today in Ireland?",
      "expected_functions": ["get_emission_analysis"],
      "ground_truth": {
        "reference_output": "🏠 **Best Times to Use Appliances...",
        "scoring_criteria": {
          "must_have": [
            "specific optimal time period 10:00-13:00 mentioned",
            "real carbon intensity values (must include 199g CO2/kWh range)",
            "at least 3 specific appliances mentioned"
          ]
        }
      }
    }
  ]
}
```

## Data Sources

### EirGrid CO2 Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   EirGrid API   │───▶│  Raw CO2 Data   │──▶│ Compressed Data  │
│   (Real-time)   │    │     (.json)     │    │    (.json)       │
└─────────────────┘    └─────────────────┘    └──────────────────┘
                                ┬───────────────────────│
                                ▼                       ▼
                       ┌─────────────────┐    ┌──────────────────┐
                       │  Ground Truth   │    │   Agent Tool     │
                       │  Generation     │    │  (Evaluation)    │
                       └─────────────────┘    └──────────────────┘
```

### Data Consistency Guarantee

**Critical**: Both the agent and the LLM judge use the **same compressed CO2 data** to ensure:
- Agent gets: `min_intensity: 199, optimal_time: 11:30`
- LLM Judge validates: Agent values within 199-260g CO2/kWh range ✓

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AZURE_DEPLOYMENT="your-deployment"
export AZURE_ENDPOINT="your-endpoint"  
export API_KEY="your-api-key"
export API_VERSION="2024-02-01"
```

### 2. Run Evaluation

```bash
# Interactive mode
python hybrid_evaluation_runner.py

# Select evaluation mode (1-3):
# 1. Rule-based evaluation
# 2. LLM-only evaluation  
# 3. Hybrid evaluation (recommended)

# Choose number of instances per test case (1-5):
# Multiple instances provide statistical confidence
# 1 = Single run per test
# 3 = Three runs per test (recommended for consistency)
# 5 = Maximum statistical confidence
```

### 3. Multi-Instance Evaluation

The system now supports running multiple instances of each test case for statistical analysis:

```bash
Select option (1-5): 3
How many times should each test case be run? (1-5): 3

Starting Hybrid Evaluation (running each test 3 times)
# Each test case will be executed 3 times
# Results are automatically averaged and aggregated
# Statistical metrics include mean, standard deviation, consistency scores
```

### 4. Check Results

Results are saved to `evaluation_results/` with detailed JSON reports:

```json
{
  "agent_name": "AssistantAgent",
  "evaluation_date": "2025-07-30T06:13:37.502843",
  "evaluation_mode": "hybrid",
  "total_tests": 6,
  "passed_tests": 4,                    // Calibrated: ~50-67% pass rate
  "failed_tests": 2,
  "avg_final_score": 0.72,              // Balanced scoring
  "avg_rule_score": 0.68,               // Fixed function detection
  "avg_llm_score": 0.78,                // Semantic equivalence support
  "calibration_notes": "v7.0 - Balanced evaluation",
  "detailed_results": [...]
}
```

## Understanding Results

### Report Structure

```json
{
  "test_metadata": {...},
  "query_and_response": {
    "test_query": "What is the best time to use appliances?",
    "full_agent_response": "🏠 **Best Times to Use Appliances...",
    "response_length": 1247
  },
  "evaluation_results": {
    "final_status": "PASS",
    "final_score": 0.85,
    "rule_based": {
      "status": "PASS",
      "score": 1.0,
      "functions_called": ["get_emission_analysis"],
      "behaviors_observed": ["correct_function_call", "high_quality_response"]
    },
    "llm_based": {
      "status": "PASS", 
      "quality_score": 0.78,
      "confidence": 0.85,
      "reasoning": "Agent provides accurate CO2 data (199-260g CO2/kWh)...",
      "feedback": [
        "Matches found: Optimal time window 10:00-13:00",
        "Missing elements: Could include more specific appliance schedules"
      ]
    }
  },
  "ground_truth_evaluation": {
    "accuracy_vs_reference": 0.85,
    "completeness_vs_reference": 0.80,
    "matches_with_reference": ["Time recommendations", "CO2 values"],
    "gaps_vs_reference": ["Could include percentage calculations"]
  }
}
```

### Key Metrics Explained

#### Overall Test Results
| Metric | Meaning | Calculation | Example |
|--------|---------|-------------|---------|
| **Passed** | Tests meeting pass threshold | `final_score >= 0.6` | `1 (16.7%)` means 1 out of 6 tests passed |
| **Failed** | Tests below pass threshold | `final_score < 0.6` | `4 (66.7%)` means 4 out of 6 tests failed |
| **Errors** | Tests with execution errors | System exceptions/crashes | `0` means no technical errors occurred |
| **Timeouts** | Tests exceeding time limit | Agent response took too long | `1` means 1 test timed out |

#### Examples.json Integration Metrics
| Metric | Meaning | Range | Calculation |
|--------|---------|-------|-------------|
| **Format Compliance** | How well responses match expected format | 0.0-1.0 | Compares agent output structure to examples.json templates |
| **Good Examples Match Rate** | Similarity to positive examples | 0.0-1.0 | Measures alignment with "good" response patterns |
| **Bad Examples Avoidance** | Avoidance of negative patterns | 0.0-1.0 | Checks agent didn't repeat "bad" example mistakes |

#### Ground Truth Metrics
| Metric | Meaning | Range | Calculation |
|--------|---------|-------|-------------|
| **Ground Truth Coverage** | Percentage of tests with reference data | 0-100% | `(tests_with_ground_truth / total_tests) × 100` |
| **Average Accuracy vs Reference** | Factual correctness against reference | 0.0-1.0 | LLM judge compares agent output to ground truth data |
| **Average Completeness** | How much required content is covered | 0.0-1.0 | Checks if agent included all expected elements |
| **Average Clarity** | Response readability and structure | 0.0-1.0 | LLM judge assesses clarity and organization |

#### Real CO2 Data Analysis
These metrics show the actual carbon emission data used for evaluation:
- **Optimal period**: Time range with lowest CO2 emissions (best for appliance use)
- **Peak emission period**: Time range with highest CO2 emissions (avoid if possible)
- **Daily average**: Overall CO2 intensity across 24 hours

#### Performance Metrics
| Metric | Meaning | Calculation |
|--------|---------|-------------|
| **Average Execution Time** | Mean time per test case | `total_execution_time / number_of_tests` |
| **Rule-based Accuracy** | Success rate of rule-based validation | `rule_based_passed / total_tests` |
| **Scoring Consistency** | Agreement between evaluation methods | Standard deviation of scores across methods |

#### Quality Scores
| Metric | Meaning | Range | Weight in Final Score |
|--------|---------|-------|----------------------|
| **Rule-based Score** | Function calls, keywords, behavior | 0.0-1.0 | 30% (Hybrid), 80% (Rule-only) |
| **LLM Quality Score** | Content accuracy, completeness, clarity | 0.0-1.0 | 50% (Hybrid), 80% (LLM-only) |
| **Combined Score** | Weighted average of all components | 0.0-1.0 | Final test result (pass if ≥ 0.6) |

#### Detailed Score Components
| Component | Meaning | Good Score | Calibration Notes |
|-----------|---------|------------|-------------------|
| **final_score** | Overall test performance | ≥ 0.6 | Aligned across all modes |
| **accuracy_vs_reference** | How factually correct vs ground truth | ≥ 0.7 | Semantic equivalence bonus |
| **completeness_vs_reference** | How much required content is covered | ≥ 0.6 | Reduced penalties |
| **quality_score** | LLM judge's overall quality assessment | ≥ 0.6 | Consistent threshold |
| **confidence** | LLM judge's confidence in evaluation | ≥ 0.5 | Tiebreaker for borderline cases |
| **functions_called** | Actual function detection from logs | Expected | Fixed detection bug |
| **behavioral_score** | Performance & decision-making | ≥ 0.6 | 20% weight in all modes |

## Advanced Usage

### Custom Test Cases

Create custom test cases in the compressed_data_test_cases.json format:

```json
{
  "id": "custom_test_001",
  "query": "Your custom query here",
  "expected_functions": ["get_emission_analysis"],
  "expected_behavior": ["correct_function_call", "user_friendly"],
  "ground_truth": {
    "reference_output": "Expected agent response format...",
    "scoring_criteria": {
      "must_have": ["Required elements"],
      "should_have": ["Nice to have elements"],
      "penalty_conditions": ["Penalty if missing X: -0.1"]
    }
  }
}
```

### Custom Evaluation Strategies

Extend the base evaluation strategy:

```python
from evaluation_strategies import BaseEvaluationStrategy

class CustomEvaluationStrategy(BaseEvaluationStrategy):
    async def evaluate(self, test_case: Dict, output_text: str) -> Dict:
        # Your custom evaluation logic
        return {
            "status": "PASS" | "FAIL",
            "score": 0.85,
            "reasoning": "Why this score was given"
        }
```

### Performance Tuning

```json
{
  "parallel_llm_calls": false,          // Enable for faster evaluation
  "max_llm_tokens": 4000,               // Increase for longer responses
  "llm_timeout": 90,                    // Adjust timeout for complex queries
  "consistency_threshold": 0.8,         // Score consistency requirement
  "enforce_scoring_consistency": true   // Strict scoring validation
}
```

## License

This evaluation system is part of the Runtime Architect's UCD x Microsoft Summer Mentorship project.