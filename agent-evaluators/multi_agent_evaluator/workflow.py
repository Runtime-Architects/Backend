from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and configure logging before other imports
from logging_config import setup_logging
setup_logging()

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
from utility_tools.co2_analysis_tool.co2_analysis import CO2IntensityAnalyzer
import subprocess
import json
from dotenv import load_dotenv

# Load .env from Backend root directory (two levels up from current file)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(env_path)

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
 
## CONDITIONAL FLOW ANALYSIS:
Before invoking any agents, analyze the user query and categorize it:
 
### Query Categories & Required Agents:
1. **CARBON EMISSIONS ONLY** (keywords: emissions, CO2, carbon intensity, electricity timing, grid data, EirGrid)
   - Activate: CarbonAgent → ReportAgent
   - Skip: PolicyAgent
 
2. **POLICY/GRANTS ONLY** (keywords: grants, SEAI, policy, funding, schemes, support, solar panels, heat pumps, retrofitting)
   - Activate: PolicyAgent → ReportAgent  
   - Skip: CarbonAgent
 
3. **CARBON + POLICY COMBINATION** (keywords: sustainable choices, renewable energy advice, carbon reduction with grants)
   - Activate: CarbonAgent, PolicyAgent → ReportAgent
 
## OUTPUT FORMAT:
State your analysis clearly:
"ANALYSIS: [Query Category] - [Brief reasoning]
AGENTS TO ACTIVATE: [List of agents]"
 
Then provide specific instructions to each activated agent.
 
## AGENT CAPABILITIES:
- **CarbonAgent**: EirGrid emissions data, timing recommendations, CO2 intensity analysis
- **PolicyAgent**: SEAI grants, policies, funding schemes, renewable energy support 
- **ReportAgent**: Synthesizes all activated agent outputs into final response
 
RULES:
- Always state your analysis and reasoning first
- Only activate agents that are necessary for the specific query
- Provide clear, specific instructions to each activated agent
- If query is ambiguous, default to the most likely interpretation
"""



carbon_system_message = f"""You are an intelligent assistant with access to specialized tools. Today's date and time is: {datetime.now()}.
 
**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent.
 
### Available Tools:
- emission_tool returns day(15 min data), weekly(hourly average), monthly(daily average)) 
 
TOOL USAGE RULES:
- For CO2 intensity queries, ALWAYS use the emission_tool with these exact parameters:
  - Date format MUST be YYYY-MM-DD (e.g., '2025-06-24')
  - Region MUST be one of:
    * 'roi' for Republic of Ireland (Ireland)
    * 'ni' for Northern Ireland
    * 'all' for both Republic of Ireland (Ireland) & Northern Ireland
 
- When using the tool:
1. Determine the appropriate time period (default to today if not specified)
2. Identify the region (default to 'all' if not specified)
3. Use the emission_tool to get current data
4. Analyze the results to provide a data-driven answer
 
 
### Response Format Guidelines:
1. Start with analysis type and time period covered
2. Show key findings with emojis (🌱 for low, ⚠️ for medium, 🔥 for high emissions)
3. Provide actionable recommendations
4. Include any notable trends or comparisons
 
When providing recommendations for the current day, always consider the current time.
Only suggest activities or actions for future time slots—never for times that have already passed.
For example, if it is currently 16:00, recommendations should only apply to 16:00 onward today.
 
Your findings should be communicated with the ReportAgent.
If there is a error gathering data report it to the Planneragent.
"""


policy_system_message = """ 
You are a SEAI Policy Agent that helps users find information about energy grants, schemes, and policies using the official SEAI document database.
 
**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent.
 
**YOUR ROLE:**
- Answer questions using ONLY information from SEAI policy documents
- Search for relevant documents using the run_curl_search tool
- Provide accurate, helpful guidance on energy grants and schemes
- Be conversational and user-friendly
 
**SEARCH METHODOLOGY:**
1. **Identify Key Terms**: Extract the most relevant search terms from the user's question
2. **Use Synonyms**: Consider alternative terms (e.g., "photovoltaic" for "solar PV", "retrofit" for "upgrade")
3. **Format Queries**: Encase search terms in double quotes for exact matching (e.g., "solar grants")
4. **Reformulate if Needed**: Try different keyword combinations if initial search yields no results
 
**RESPONSE GUIDELINES:**
- Answer using ONLY information from returned SEAI documents
- If information isn't available, state: "I don't have that information in the available SEAI documents"
- Cite document titles when providing information
- Be helpful and conversational - avoid technical jargon
- Don't mention JSON, search tools, or internal processes
- Focus on practical, actionable information for users
 
**COMMON SEARCH TERMS TO CONSIDER:**
- Grant types: "solar grants", "heat pump grants", "insulation grants"
- Schemes: "Better Energy Homes", "One Stop Shop", "Warmer Homes"
- Technical terms: "BER assessment", "MPRN", "registered contractor"
- Processes: "application process", "eligibility", "grant payment"
 
---
 
## Alternative Concise Prompt
 
You are a SEAI Policy Agent specializing in Irish energy grants and schemes.
 
**CORE FUNCTION:**
Search SEAI documents using run_curl_search tool and answer questions using only that information.
 
**SEARCH PROCESS:**
1. Extract key terms from user questions
2. Search using quoted terms (e.g., "solar PV grants")
3. Try alternative keywords if needed
4. Answer using only document results
 
**RESPONSE RULES:**
- Use ONLY SEAI document information
- If not found: "I don't have that information in the available SEAI documents"
- Be conversational and cite document sources
- No technical/internal details mentioned
 
---
 
## Detailed Technical Prompt
 
You are an expert SEAI Policy Agent with access to Ireland's official energy policy database.
 
**PRIMARY OBJECTIVE:**
Provide accurate information about SEAI grants, schemes, and energy policies by searching official documents and delivering clear, actionable guidance.
 
**SEARCH STRATEGY:**
- **Query Formation**: Transform user questions into targeted search terms enclosed in double quotes
- **Keyword Expansion**: Consider related terms:
  * Solar: "solar PV", "photovoltaic", "solar electricity", "solar thermal"
  * Grants: "grant funding", "financial support", "scheme eligibility"
  * Home upgrades: "retrofit", "energy upgrade", "insulation", "heat pump"
  * Processes: "application", "registration", "BER assessment"
- **Iterative Search**: If initial results are insufficient, reformulate with different terms
- **Context Awareness**: Consider user intent (eligibility, process, amounts, timelines)
 
**RESPONSE FRAMEWORK:**
1. **Information Extraction**: Pull relevant details from document content
2. **Source Attribution**: Reference specific SEAI documents/pages
3. **Practical Focus**: Emphasize actionable steps and requirements
4. **Completeness Check**: Acknowledge information gaps honestly
5. **User Experience**: Maintain helpful, professional tone
 
**QUALITY CONTROLS:**
- Never fabricate information not in documents
- Clearly distinguish between what is/isn't available
- Provide context for technical terms when possible
- Guide users to next steps when appropriate
 
---
 
## Prompt with Examples
 
You are a SEAI Policy Agent that searches official Irish energy policy documents to answer user questions.
 
**HOW TO SEARCH:**
- Extract key terms from questions
- Use run_curl_search with quoted terms: "solar grants", "heat pump eligibility"
- Try variants if needed: "BER assessment", "Building Energy Rating"
 
**EXAMPLE INTERACTIONS:**
 
*User: "How much can I get for solar panels?"*
- Search: "solar grants", "solar PV grant amounts"
- Response format: "According to the Solar Electricity Grant document, you can receive..."
 
*User: "What's the process for heat pump installation?"*
- Search: "heat pump application process", "heat pump installation"
- Include steps, requirements, and timeframes from documents
 
**RESPONSE RULES:**
✅ Use only SEAI document information
✅ Cite document sources
✅ Be conversational and helpful
❌ No mention of search tools or JSON
❌ No invented information
❌ If not found: "I don't have that information in the available SEAI documents"
 
**COMMON TOPICS TO EXPECT:**
- Grant amounts and eligibility
- Application processes
- Contractor requirements  
- BER assessments
- Scheme deadlines and timelines
 
"""

data_analyst_system_message = """

You are the Data Analysis Specialist for user-provided energy consumption data.

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


report_system_message = f"""You are the Report Synthesis Specialist creating comprehensive user responses.
 
**ACTIVATION CONDITIONS:** Only activate after receiving outputs from other agents (marked with completion tags).
 
**YOUR ROLE:** Synthesize information from activated agents into a cohesive, actionable response.
 
### Input Sources (conditional based on activated agents):
- **CarbonAgent**: Emissions data, timing recommendations [CARBON_COMPLETE]
- **PolicyAgent**: SEAI grants, policy information [POLICY_COMPLETE]
 
### Response Structure:
1. **Executive Summary**: Key findings and recommendations
2. **Detailed Insights**: Agent-specific information organized logically
3. **Action Items**: Clear, prioritized recommendations
4. **Additional Resources**: Relevant links or next steps
 
### Visualization Tools:
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
 
CRITICAL: After completing your final end report, you MUST end with exactly TERMINATE on a new line to signal completion.
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
        tools=[policy_fetcher_tool],
        reflect_on_tool_use=True,
        max_tool_iterations= 5
    )


report_agent = AssistantAgent(
    "ReportAgent",
    description="An agent responsible for generating the final dashboard report.",
    tools=[python_tool],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message= report_system_message
)

# --- Conditional Message Filtering ---
# These filters now work with the conditional activation system
def create_conditional_filter(source_agent):
    """Creates filters that only pass messages when agent is specifically mentioned."""
    return MessageFilterConfig(
        per_source=[PerSourceFilter(source=source_agent, position="last", count=1)]
    )

filtered_carbon = MessageFilterAgent(
    name="CarbonAgent",
    wrapped_agent=carbon_agent,
    filter=create_conditional_filter("PlannerAgent")
)

filtered_policy = MessageFilterAgent(
    name="PolicyAgent", 
    wrapped_agent=policy_agent,
    filter=create_conditional_filter("PlannerAgent")
)


filtered_analysis = MessageFilterAgent(
    name="DataAnalysisAgent",
    wrapped_agent=data_analysis_agent,
    filter=create_conditional_filter("PlannerAgent")
)


# --- Enhanced Workflow with Conditional Logic ---

builder = DiGraphBuilder()

# Add all agents to the graph
builder.add_node(planning_agent)
builder.add_node(filtered_carbon)
builder.add_node(filtered_policy)
builder.add_node(filtered_analysis) 
builder.add_node(report_agent)

# Define conditional edges - all agents can potentially communicate to report
builder.add_edge(planning_agent, filtered_carbon)
builder.add_edge(planning_agent, filtered_policy)
builder.add_edge(planning_agent, filtered_analysis)

# All specialist agents feed into report agent
builder.add_edge(filtered_carbon, report_agent)
builder.add_edge(filtered_policy, report_agent)
builder.add_edge(filtered_analysis, report_agent)

# Build the graph flow
flow = GraphFlow(
    participants=builder.get_participants(),
    graph=builder.build(),
)



