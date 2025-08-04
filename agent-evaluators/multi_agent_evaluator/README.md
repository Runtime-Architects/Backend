# Multi-Agent Prompt Engineering Tool

This framework provides a way to evaluate the performance of agents in a multi-agent system (MAS) by analyzing interaction logs, critiquing each agent's behavior, and offering actionable suggestions for prompt engineering improvements. The goal is to enhance collaboration, efficiency, and task success in multi-agent workflows.

## Overview
This toolset provides an interactive console interface for analyzing and improving AI agent performance through prompt engineering. It helps you:

- Test custom prompts with multi-agent workflow

- Evaluate and critique agent performance using existing conversation logs

-  Receive suggestions for prompt improvements


## Usage

Run the main script:
```
python sp_evaluator_flow.py
```

The main menu will be displayed
```
===== Prompt Engineer =====
1. Run a custom test
2. Evaluate existing log
3. Prompt Engineer
4. Exit
```

### Option 1: Run a Custom Test
1. Enter a prompt to test the multi-agent workflow

2. The system will execute the workflow and display all agent interactions

3. Results are automatically saved to agent_logs/ with timestamp

4. An AI critique of the interaction is generated and saved with the log

```
> 1
> Enter the prompt to test: What's the best time to run my dishwasher in Ireland?
(Observe the agent interactions and critique)
```

### Option 2: Evaluate Existing Log
1. Enter the path to an existing log file

2. The system will analyze the log and generate a performance critique

3. The critique is added to the log file

```
> 2
> Enter the log file to evaluate (or type 'exit' to quit) <file_path>
(Observe critique for existing log)
```

### Option 3: Prompt Engineer
1. The system scans the agent_logs/ directory

2. Shows the 10 most recent logs

3. Analyzes the 10 most recent critiques to suggest which agent's prompt needs improvement

4. Provides specific recommendations for prompt engineering

```
> 3
> There are 1 logs.
> The most recent are:
> agent_logs\agent_logs_20250802_172201.json

> Press Enter to generate suggestions.
(Get improvement suggestions)
```

## Log File Structure
Logs are saved as JSON files with this structure:
```
{
  "planner_system_prompt": "...",
  "carbon_system_prompt": "...",
  "policy_system_prompt": "...",
  "report_system_prompt": "...",
  "time_taken": 12.34,
  "query": "user's original query",
  "log": [
    {"agent1": "response content"},
    {"agent2": "response content"}
  ],
  "critic": "AI-generated performance analysis"
}
```