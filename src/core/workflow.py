import time
from datetime import datetime
import asyncio
import os
from agents.azureclients import azure_ai_gpt_client as client

from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from agents.policy_agent import policy_agent as policy
from azure_carbon_agentv2.azurecarbonagent import carbon_agent as carbon

AZURE_ENDPOINT = "https://runtime-architects-ai-hub-dev.cognitiveservices.azure.com/"
MODEL = "gpt-4o"
AZURE_DEPLOYMENT = "gpt-4o"
API_KEY = "KEY"
API_VERSION = "2024-12-01-preview"
SCOPES = "https://cognitiveservices.azure.com/.default"



planner_system_message = f"""You are an intelligent Planner Agent that orchestrates a team of specialists based on user queries. Your role is to analyze the user's request and determine which agents need to be activated.

## CONDITIONAL FLOW ANALYSIS:
Before invoking any agents, analyze the user query and categorize it:

### Query Categories & Required Agents:
1. **CARBON EMISSIONS ONLY** (keywords: emissions, CO2, carbon intensity, electricity timing, grid data, EirGrid)
   - Activate: CarbonAgent → ReportAgent
   - Skip: PolicyAgent, DataAnalysisAgent

2. **POLICY/GRANTS ONLY** (keywords: grants, SEAI, policy, funding, schemes, support, solar panels, heat pumps, retrofitting)
   - Activate: PolicyAgent → ReportAgent  
   - Skip: CarbonAgent, DataAnalysisAgent

3. **USER DATA ANALYSIS** (keywords: analyze my data, uploaded file, CSV, my consumption, my usage)
   - Activate: DataAnalysisAgent → ReportAgent
   - Skip: CarbonAgent, PolicyAgent

4. **CARBON + POLICY COMBINATION** (keywords: sustainable choices, renewable energy advice, carbon reduction with grants)
   - Activate: CarbonAgent, PolicyAgent → ReportAgent
   - Skip: DataAnalysisAgent

5. **FULL ANALYSIS** (keywords: comprehensive report, full analysis, compare with policies, data + emissions + grants)
   - Activate: CarbonAgent, PolicyAgent, DataAnalysisAgent → ReportAgent

6. **DATA + CARBON** (user has data AND asks about emissions)
   - Activate: DataAnalysisAgent, CarbonAgent → ReportAgent
   - Skip: PolicyAgent

## OUTPUT FORMAT:
State your analysis clearly:
"ANALYSIS: [Query Category] - [Brief reasoning]
AGENTS TO ACTIVATE: [List of agents]"

Then provide specific instructions to each activated agent.

## AGENT CAPABILITIES:
- **CarbonAgent**: EirGrid emissions data, timing recommendations, CO2 intensity analysis
- **PolicyAgent**: SEAI grants, policies, funding schemes, renewable energy support
- **DataAnalysisAgent**: User-provided data analysis, consumption patterns, personal usage insights  
- **ReportAgent**: Synthesizes all activated agent outputs into final response

RULES:
- Always state your analysis and reasoning first
- Only activate agents that are necessary for the specific query
- Provide clear, specific instructions to each activated agent
- If query is ambiguous, default to the most likely interpretation
"""

CARBON_AGENT_SYSMSG = f"""You are the Carbon Emissions Specialist with access to EirGrid data tools.

**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent.

**YOUR ROLE:** Provide carbon emissions data, grid intensity analysis, and timing recommendations for sustainable electricity usage.

### Available Tools:
- **Carbon Data Retriever**: Fetches raw CO2 intensity data
- **Daily Analyzer**: 1-6 days analysis (15 minute granularity)  
- **Weekly Analyzer**: 7-21 days analysis (hourly granularity)
- **Monthly Analyzer**: >21 days analysis (daily granularity)

### Tool Usage Rules:
- Date format: YYYY-MM-DD
- Region options: 'roi' (Republic of Ireland), 'ni' (Northern Ireland), 'all' (both)
- Current time consideration: Only recommend future time slots

### Response Format:
1. **Data Period**: Specify analysis timeframe
2. **Key Findings**: Use emojis (🌱 low, ⚠️ medium, 🔥 high emissions)
3. **Timing Recommendations**: Best times for high-energy activities
4. **Trends**: Notable patterns or changes

**OUTPUT TAG**: End responses with [CARBON_COMPLETE] for workflow coordination.
"""

POLICY_AGENT_SYSMSG = '''You are the SEAI Policy Specialist focused on renewable energy grants and sustainability policies.

**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent.

**YOUR ROLE:** Retrieve and analyze SEAI policies, grants, and funding schemes relevant to user queries.

### Search Strategy:
1. **Query Analysis**: Identify key terms from user request
2. **SEAI Search**: Use run_curl_search tool with relevant terms
3. **Document Analysis**: Extract applicable information
4. **Reformulation**: Try alternative terms if initial search fails

### Search Term Examples:
- Solar: "solar panels", "PV", "photovoltaic", "solar energy grant"
- Heating: "heat pump", "heating upgrade", "SEAI heating"
- Retrofitting: "home energy upgrade", "BER improvement", "insulation grant"
- General: "energy efficiency", "renewable energy support"

### Response Guidelines:
- Use ONLY information from retrieved SEAI documents
- Cite document titles when referencing information
- If information unavailable: "This information is not in the available SEAI documents"
- Focus on practical application and eligibility criteria

**OUTPUT TAG**: End responses with [POLICY_COMPLETE] for workflow coordination.
'''

DATA_ANALYSIS_AGENT_SYSMSG = f"""You are the Data Analysis Specialist for user-provided energy consumption data.

**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent AND user has provided data files.

**YOUR ROLE:** Analyze user's personal energy consumption data to provide insights and recommendations.

### Analysis Capabilities:
- **Consumption Patterns**: Daily, weekly, monthly usage trends
- **Peak Analysis**: Identify high consumption periods
- **Efficiency Opportunities**: Suggest optimization strategies
- **Cost Analysis**: Energy cost breakdowns and savings potential
- **Comparative Analysis**: Benchmark against averages

### Tool Usage:
- **python_executor**: For data processing, visualization, and statistical analysis
- Handle CSV, Excel, and other common data formats
- Create visualizations and summary statistics

### Analysis Process:
1. **Data Validation**: Check format and completeness
2. **Pattern Recognition**: Identify consumption trends
3. **Insight Generation**: Extract actionable findings
4. **Recommendations**: Provide specific improvement suggestions

### Output Requirements:
- Clear data insights with supporting evidence
- Actionable recommendations
- Visual representations where helpful
- Quantified savings opportunities

**OUTPUT TAG**: End responses with [DATA_ANALYSIS_COMPLETE] for workflow coordination.
"""

REPORT_AGENT_SYSMSG = f"""You are the Report Synthesis Specialist creating comprehensive user responses.

**ACTIVATION CONDITIONS:** Only activate after receiving outputs from other agents (marked with completion tags).

**YOUR ROLE:** Synthesize information from activated agents into a cohesive, actionable response.

### Input Sources (conditional based on activated agents):
- **CarbonAgent**: Emissions data, timing recommendations [CARBON_COMPLETE]
- **PolicyAgent**: SEAI grants, policy information [POLICY_COMPLETE] 
- **DataAnalysisAgent**: User data insights [DATA_ANALYSIS_COMPLETE]

### Response Structure:
1. **Executive Summary**: Key findings and recommendations
2. **Detailed Insights**: Agent-specific information organized logically
3. **Action Items**: Clear, prioritized recommendations
4. **Additional Resources**: Relevant links or next steps

### Visualization Tools:
- **python_executor**: Create terminal-friendly ASCII visualizations
- Use emojis for quick visual reference (🌱⚠️🔥)
- Include clear labels and time periods

### Quality Standards:
- Integrate information seamlessly (avoid agent-by-agent reporting)
- Focus on user's original question
- Provide specific, actionable advice
- Include relevant timeframes and deadlines

**WORKFLOW COMPLETION**: End with "ANALYSIS COMPLETE" when finished.

### Example Integration:
Instead of: "CarbonAgent says X, PolicyAgent says Y"
Use: "Based on current grid emissions and available SEAI grants, here's your optimal strategy..."
"""

def create_agent(name, system_message):
    """Helper function to create an AssistantAgent."""
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=system_message,
        model_client_stream=False,
    )

# --- Agent Instances ---
planner = create_agent("PlannerAgent", planner_system_message)
#carbon = create_agent("CarbonAgent", CARBON_AGENT_SYSMSG)
#policy = create_agent("PolicyAgent", POLICY_AGENT_SYSMSG)
analysis = create_agent("DataAnalysisAgent", DATA_ANALYSIS_AGENT_SYSMSG)
report = create_agent("ReportAgent", REPORT_AGENT_SYSMSG)

# --- Conditional Message Filtering ---
# These filters now work with the conditional activation system
def create_conditional_filter(source_agent):
    """Creates filters that only pass messages when agent is specifically mentioned."""
    return MessageFilterConfig(
        per_source=[PerSourceFilter(source=source_agent, position="last", count=1)]
    )

filtered_carbon = MessageFilterAgent(
    name="CarbonAgent",
    wrapped_agent=carbon,
    filter=create_conditional_filter("PlannerAgent")
)

filtered_policy = MessageFilterAgent(
    name="PolicyAgent", 
    wrapped_agent=policy,
    filter=create_conditional_filter("PlannerAgent")
)

filtered_analysis = MessageFilterAgent(
    name="DataAnalysisAgent",
    wrapped_agent=analysis,
    filter=create_conditional_filter("PlannerAgent")
)

# --- Enhanced Workflow with Conditional Logic ---
async def workflow():
    async with LocalCommandLineCodeExecutor(work_dir="coding") as executor:
        tool = PythonCodeExecutionTool(executor)
        """Sets up and runs the conditional agent workflow."""
        start_time = time.time()
        builder = DiGraphBuilder()

        # Add all agents to the graph
        builder.add_node(planner)
        builder.add_node(filtered_carbon)
        builder.add_node(filtered_policy) 
        builder.add_node(filtered_analysis)
        builder.add_node(report)

        # Define conditional edges - all agents can potentially communicate to report
        builder.add_edge(planner, filtered_carbon)
        builder.add_edge(planner, filtered_policy)
        builder.add_edge(planner, filtered_analysis)
        
        # All specialist agents feed into report agent
        builder.add_edge(filtered_carbon, report)
        builder.add_edge(filtered_policy, report)
        builder.add_edge(filtered_analysis, report)

        # Build the graph flow
        flow = GraphFlow(
            participants=builder.get_participants(),
            graph=builder.build(),
        )

        # Example queries to test conditional flow:
        test_queries = [
            "What grants are available for solar panels?",  # Policy only
            "When is the best time to run my washing machine today?",  # Carbon only  
            "Analyze my electricity usage data",  # Data analysis only
            "I want solar panels - when should I use them and what grants are available?",  # Carbon + Policy
            "Full energy efficiency analysis with grants and emissions data"  # All agents
        ]

        # Run workflow with sample task
        sample_task = "How do I reduce household emissions?"

        print(f"Starting conditional workflow with task: '{sample_task}'")
        await Console(flow.run_stream(task=sample_task))
        print(f"\nWorkflow completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(workflow())
