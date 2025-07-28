import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_core.tools import FunctionTool
from co2_analysis_tool.co2_analysis import CO2IntensityAnalyzer
from datetime import datetime
import json
import subprocess
import os
import time
from dotenv import load_dotenv
load_dotenv()

#client for agents and selector_chat
model_client = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL"),
                    api_version=os.getenv("API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key= os.getenv("API_KEY"), # For key-based authentication.
                )

model_client2 = AzureOpenAIChatCompletionClient(
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                    model=os.getenv("MODEL2"),
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
 
4. **CARBON + POLICY COMBINATION** (keywords: sustainable choices, renewable energy advice, carbon reduction with grants)
   - Activate: CarbonAgent, PolicyAgent → ReportAgent
 
5. **FULL ANALYSIS** (keywords: comprehensive report, full analysis, compare with policies, data + emissions + grants)
   - Activate: CarbonAgent, PolicyAgent → ReportAgent
 
6. **DATA + CARBON** (user has data AND asks about emissions)
   - Activate: CarbonAgent → ReportAgent
   - Skip: PolicyAgent
 
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
 
CRITICAL: After completing your final end report, you MUST end with exactly TERMINATE on a new line to signal completion.
"""


# ====== Agents =======

planning_agent = AssistantAgent(
    "planning_agent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message= planner_system_message
)

user_proxy = UserProxyAgent(
    name="User",
)


carbon_agent = AssistantAgent(
            name="carbon_agent", 
            model_client=model_client,    
            description="An agent responsible gathering carbon related information",
            tools=[emission_tool], 
            reflect_on_tool_use=True,
            max_tool_iterations= 5,
            system_message= carbon_system_message
        )

policy_agent = AssistantAgent(
        name="policy_agent",
        description="An agent responsible gathering policy related information (eg: energy upgrades, Tax Credits)",
        model_client=model_client,
        system_message=policy_system_message,
        tools=[policy_fetcher_tool],
        reflect_on_tool_use=True,
        max_tool_iterations= 5
    )


report_agent = AssistantAgent(
    "report_agent",
    description="An agent responsible for generating the final dashboard report.",
    tools=[python_tool],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message= report_system_message
)




# Define a termination condition that stops the task if the planner accepts the final report and says TERMINATE.
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=40) # Increased max messages for more complex flow

selector_prompt = """Select the most appropriate agent to handle this request based on the following rules:

1. If the user's request is simple and can be handled by a single specialist agent, select that agent directly
2. If the request is complex and requires task decomposition, select the planning_agent first
3. If multiple agents have contributed to the conversation and synthesis is needed, select the report_agent
4. Use only the required agents.

Available agents:
{roles}

Current conversation context:
{history}

Based on the above, select the next agent from {participants} to continue the conversation.
Only select one agent.
"""


# Create a team to collaborate and accomplish the task
# Update your team with the filtered agents
team = SelectorGroupChat(
    [user_proxy, 
     planning_agent, 
     carbon_agent, 
     policy_agent, 
     report_agent],
    model_client=model_client2,
    selector_prompt=selector_prompt,
    termination_condition=termination,
    allow_repeated_speaker=True
)


async def main():

    # task = "What was the average daily carbon emission in Ireland for the first two weeks of this month versus the last two weeks of the last month?"
    # task= "What is the carbon emission last week?"
    #task= "Analyze Ireland's emissions trends over the past month and suggest 2 policies to reduce peak emissions"    
    task="What is the ber related policy for homes?"
    # task="If Ireland doubled its wind energy capacity, how would emissions change? Refer to SEAI's wind energy grants"
    #task= "How do I reduce household emissions?"
    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    start= time.time()
    asyncio.run(main())
    print(f'Time taken: {time.time() - start}')


