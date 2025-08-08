import time
from datetime import datetime
import asyncio
import os
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
from co2_analysis_tool.co2_analysis import CO2IntensityAnalyzer
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()

scopes = os.getenv("SCOPES")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
search_api_key = os.getenv("SEARCH_API_KEY")
search_api_version= os.getenv("SEARCH_API_VERSION") 
search_endpoint=  os.getenv("SEARCH_ENDPOINT") 

model_client = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL"),
                    api_version=os.getenv("API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key= os.getenv("API_KEY"), # For key-based authentication.
                )


# =====tools======
executor= LocalCommandLineCodeExecutor(work_dir="coding")
python_tool = PythonCodeExecutionTool(executor)

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> float:
    # Check if the file exists

    try:
        # Your existing tool logic
        analyzer = CO2IntensityAnalyzer(startdate, enddate, region)
        intensity_periods = analyzer.get_analysis_by_view()

        return intensity_periods
    except Exception as e:
            # Return detailed error info
            return {
                "error": str(e),
                "input_params": {"startdate": startdate, "enddate": enddate, "region": region}
            }


# Create a function tool
emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), " \
    "region ('all' (Republic of Ireland and Northern Ireland), 'roi' (Republic of Ireland), or 'ni' (Northern Ireland)",
    name= "emission_tool"
)

scopes = os.getenv("SCOPES")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
search_api_key = os.getenv("SEARCH_API_KEY")
search_api_version= os.getenv("SEARCH_API_VERSION") 
search_endpoint=  os.getenv("SEARCH_ENDPOINT") 

async def run_curl_search(query: str) -> str:
    
    curl_command = [
        "curl", "-X", "POST",
        f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}",
        "-H", "Content-Type: application/json",
        "-H", f"api-key: {search_api_key}",
        "-d", json.dumps({
            "search": query,
            "top": 5
        })
    ]
    
    result = subprocess.run(curl_command, capture_output=True, text=True)
    return result.stdout


policy_fetcher_tool = FunctionTool(
    func=run_curl_search,
    description="Fetches Policy data from SEAI",
    name= "run_curl_search"
)


# =====system prompts======
planner_system_message = f"""You are an intelligent Planner Agent that orchestrates a team of specialists based on user queries. Your role is to analyze the user's request and determine which agents need to be activated.

### CONDITIONAL FLOW ANALYSIS:
Before invoking any agents, analyze the user query and categorize it:

### Query Categories & Required Agents:
1. **CARBON EMISSIONS ONLY** (keywords: emissions, CO2, carbon intensity, electricity timing, grid data, EirGrid)
   - Activate: CarbonAgent → ReportAgent

2. **POLICY/GRANTS ONLY** (keywords: grants, SEAI, policy, funding, schemes, support, solar panels, heat pumps, retrofitting)
   - Activate: PolicyAgent → ReportAgent  

3. **USER DATA ANALYSIS** (keywords: analyze my bill, uploaded file, CSV, my consumption, my usage)
   - Activate: DataAnalysisAgent → ReportAgent

4. **CARBON + POLICY COMBINATION** (keywords: sustainable choices, renewable energy advice, carbon emission reduction with grants)
   - Activate: CarbonAgent, PolicyAgent → ReportAgent

5. **FULL ANALYSIS** (keywords: comprehensive report, full analysis, compare with policies)
   - Activate: CarbonAgent, PolicyAgent, DataAnalysisAgent → ReportAgent

6. **DATA + CARBON** (user has data AND asks about emissions)
   - Activate: DataAnalysisAgent, CarbonAgent → ReportAgent

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

## RULES:
- Always state your analysis and reasoning first
- Only activate agents that are necessary for the specific query
- If no agents are required: 
  = AGENTS TO ACTIVATE: none, 
  = Say nothing else (no suggestions, or follow-ups).
- Provide clear, specific instructions to each activated agent
- If query is ambiguous, default to the most likely interpretation

"""


carbon_system_message = f"""You are an intelligent assistant with access to tools that can retreive carbon emissions for Ireland, Northern Ireland or both. Today's date and time is: {datetime.now()}.
 
**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent.

### AVAILABLE TOOLS:
- emission_tool returns day(15 min), weekly(hourly average), monthly(daily average)) 
    ## TOOL USAGE RULES:
    - For CO2 intensity queries, ALWAYS use the emission_tool with these exact parameters:
    - Date format MUST be YYYY-MM-DD (e.g., '2025-06-24')
    - Region MUST be one of:
        * 'roi' for Republic of Ireland (Ireland)
        * 'ni' for Northern Ireland
        * 'all' for both Republic of Ireland (Ireland) & Northern Ireland
    ## TOOL RESTRICTIONS:
    - This tool can access data only up to a single month

###RULES:
- **ALWAYS USE TOOLS**: to gather data before responding 
    - if gathering data is not possible or has error DO NOT provide recommendations
- **STRICT DATA-DRIVEN RESPONSES**: All claims must be supported by tool-generated data.
 
### ADDITIONAL RULES:
- Determine the appropriate time period (default to today if not specified)
- Identify the region (default to 'all' if not specified)
- Analyze the results to provide a data-driven answer

### QUALITY CONTROLS:
- Never fabricate information if you cannot retrieve data
- Clearly distinguish between what is/isn't available
- DO NOT perform actions or answer beyond your expertise
- DO NOT try to ask further questions
- End with recommendations based on the retrieved data
- If there is an error gathering data say 'There was an error fetching data now, if this error persists please contact the administrator.'
 
### RESPONSE GUIDELINES:
- Start with analysis type and time period covered
- Show key findings with emojis (🌱 for low, ⚠️ for medium, 🔥 for high emissions)
- Provide actionable recommendations:
    - When providing recommendations for the current day, always consider the current time.
    - When providing recommendations based on previous week's data, consider any trends based on emissin time periods, weekdays, weekends etc.
    - When providing recommendations based on previous month's data, consider any trends based on seasons, day light timings, weekdays, weekends etc.
- Include any notable trends or comparisons
"""


policy_system_message = """ 
You are a SEAI (Sustainable Energy Authority Of Ireland) assistant, specializing in Irish energy grants and schemes helping users find information about energy grants, schemes, and policies using the official SEAI document database.
 
**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent.
 
### AVAILABLE TOOLS:
- policy_fetcher_tool: Fetches policy related information from SEAI
    ## TOOL USAGE RULES:
    - **Identify Key Terms**: Extract the most relevant search terms from the user's question
    - **Format Queries**: Encase search terms in double quotes for exact matching (e.g., "solar grants")
    - **Use Synonyms**: Consider alternative terms (e.g., "photovoltaic" for "solar PV", "retrofit" for "upgrade")
    - **Reformulate if Needed**: Try different keyword combinations if initial search yields no results

### COMMON SEARCH TERMS TO CONSIDER:
- Grant types: "solar grants", "heat pump grants", "insulation grants"
- Schemes: "Better Energy Homes", "One Stop Shop", "Warmer Homes"
- Technical terms: "BER assessment", "MPRN", "registered contractor"
- Processes: "application process", "eligibility", "grant payment"
- Solar: "solar PV", "photovoltaic", "solar electricity", "solar thermal"
- Grants: "grant funding", "financial support", "scheme eligibility"
- Home upgrades: "retrofit", "energy upgrade", "insulation", "heat pump"
- Processes: "application", "registration", "BER assessment"

### ADDITIONAL RULES:
- Answer questions using ONLY information from SEAI policy documents
- Provide accurate, helpful guidance on energy grants and schemes
- **Context Awareness**: Consider user intent (eligibility, process, amounts, timelines)
- Be conversational and user-friendly

### QUALITY CONTROLS:
- Never fabricate information not in documents
- Clearly distinguish between what is/isn't available
- Provide context for technical terms when possible
- Guide users to next steps when appropriate
- DO NOT perform actions or answer beyond your expertise
- DO NOT try to ask further questions

### RESPONSE GUIDELINES:
- Answer using ONLY information from returned SEAI documents
- If information isn't available, state: "I don't have that information in the available SEAI documents"
- Cite document titles when providing information
- Be helpful and conversational - avoid technical jargon
- Don't mention JSON, search tools, or internal processes
- Focus on practical, actionable information for users 
"""

data_analyst_system_message = """

You are a Data Analysis assistant that analyzes user's personal energy consumption data to provide insights and recommendations 

**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent AND user has provided data files.

### AVAILABLE TOOLS:
- **python_executor**: For data processing and statistical analysis
    ## TOOL USAGE RULES:
    - Handle CSV, Excel, and other common data formats
    - Create summary statistics

### Analysis Capabilities:
- **Consumption Patterns**: Daily, weekly, monthly usage trends
- **Peak Analysis**: Identify high consumption periods
- **Efficiency Opportunities**: Suggest optimization strategies
- **Cost Analysis**: Energy cost breakdowns and savings potential
- **Comparative Analysis**: Benchmark against averages

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
"""

report_system_message = f"""You are a helpful assistant thar synthesizes information short, clear and conversational answers.

**ACTIVATION CONDITIONS:** 
- Activate when receiving either:
  a) Outputs from multiple agents (carbon, policy, data analysis), OR
  b) Direct messages from planner_agent

**KEY RULES:**
1. Never reveal internal workflows: (no mention of "agents," "sources," or "synthesis").
2. For generic questions:
   - Keep answers short and user-centric (focus on what the user can do, not how you work).
   - Example: "I can help with energy efficiency advice, policy details, and carbon footprint estimates. Ask me anything specific!" 

### YOUR ROLE:
1. For multi-agent responses:
   - Review and combine outputs from other agents
   - Combine the key information into a natural, easy-to-understand and concise responses
   - Keep answers focused and practical
   - Identify and address any information gaps

2. For direct planner_agent messages:
   - Respond conversationally to the user query
   - Mention what you can help with
   - Maintain the same friendly, practical tone
   - Keep you responses simple

### WORKFLOW:
    ## FOR MULTI-AGENT RESPONSES:
    1. Verify if all aspects of the user's question are addressed
    2. Combine key points into one seamless answer
    3. Attribute information naturally (e.g., "According to energy data..." instead of "The carbon agent says...")
    4. Highlight deadlines or urgent actions
    5. Suggest next steps when helpful

    ## FOR DIRECT PLANNER_AGENT MESSAGES:
    1. Read the user query carefully
    2. Respond as if having a conversation
    3. Keep explanations simple and practical
    4. Ask clarifying questions if needed

**EXAMPLE STYLE:**
Good: "The best option would be X because... You'll want to apply before [date] since..."
Bad: "AgentA recommends X. AgentB states the deadline is..."

### QUALITY CONTROLS:
- Integrate information seamlessly (avoid agent-by-agent reporting)
- Use everyday language 
- Focus on user's original question
- Provide specific, actionable advice
- Include all key points
- Maintain consistent tone and style

End your final response with "ANALYSIS COMPLETE" and then "TERMINATE" on a new line.
"""


# ====== Agents =======
planning_agent = AssistantAgent(
    "PlannerAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message= planner_system_message
)

carbon_agent = AssistantAgent(
            name="CarbonAgent", 
            model_client=model_client,    
            description="An agent responsible gathering carbon related information",
            tools=[emission_tool], 
            reflect_on_tool_use=True,
            max_tool_iterations= 5,
            system_message= carbon_system_message
        )

policy_agent = AssistantAgent(
        name="PolicyAgent",
        description="An agent responsible gathering policy related information (eg: energy upgrades, Tax Credits)",
        model_client=model_client,
        system_message=policy_system_message,
        tools=[policy_fetcher_tool],
        reflect_on_tool_use=True,
        max_tool_iterations= 5
    )

data_analysis_agent= AssistantAgent(
        name="DataAnalysisAgent",
        description="An agent responspible for data analysis",
        model_client=model_client,
        system_message= data_analyst_system_message,
        tools=[python_tool],
        reflect_on_tool_use=True,
        max_tool_iterations= 5
    )


report_agent = AssistantAgent(
    "ReportAgent",
    description="An agent responsible for generating the final response",
    model_client=model_client,
    system_message= report_system_message
)

# --- Conditional Message Filtering ---
# These filters now work with the conditional activation system
def create_conditional_filter(source_agent):
    """Creates filters that only pass messages when agent is specifically mentioned."""
    return MessageFilterConfig(
        per_source=[PerSourceFilter(source=source_agent, position="last", count=1)]
    )


builder = DiGraphBuilder()

# Add all agents to the graph
builder.add_node(planning_agent)
builder.add_node(carbon_agent)
builder.add_node(policy_agent)
builder.add_node(data_analysis_agent) 
builder.add_node(report_agent)

def is_mentioned(agent_name: str):
    """Returns a function that checks if a given agent_name is in the message."""
    def check_mention(msg):
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict): 
            content = msg.get("content", "")
        else:
            return False
        return agent_name.lower() in content.lower()
    return check_mention 


# Define conditional edges - all agents can potentially communicate to report
builder.add_edge(planning_agent, carbon_agent, condition=is_mentioned("carbonagent"))
builder.add_edge(planning_agent, policy_agent, condition=is_mentioned("policyagent"))
builder.add_edge(planning_agent, data_analysis_agent, condition=is_mentioned("dataanalysisgent"))

# All specialist agents feed into report agent
builder.add_edge(carbon_agent, report_agent, activation_group="working_agent", activation_condition="any")
builder.add_edge(policy_agent, report_agent, activation_group="working_agent", activation_condition="any")
builder.add_edge(data_analysis_agent, report_agent, activation_group="working_agent", activation_condition="any")

builder.add_edge(planning_agent, report_agent, activation_group="planning_agent", activation_condition="any", condition=is_mentioned("none"))

# Build the graph flow
flow = GraphFlow(
    participants=builder.get_participants(),
    graph=builder.build(),
)

test_queries = [
        "What is the best time to use my appliances today in Ireland?",  # Policy only
        #"When is the best time to run my washing machine today?",  # Carbon only  
        #"Analyze my electricity usage data",  # Data analysis only
        #"I want solar panels - when should I use them and what grants are available?",  # Carbon + Policy
        #"Full energy efficiency analysis with grants and emissions data"  # All agents
    ]


# --- Enhanced Workflow with Conditional Logic ---
async def workflow():
    """Sets up and runs the conditional agent workflow."""

    # Run workflow with sample task
    sample_task = "Hi"

    await Console(flow.run_stream(task=sample_task))

if __name__ == "__main__":
    asyncio.run(workflow())
    
    

