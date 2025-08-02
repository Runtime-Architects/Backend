from agents.tools.emission_tool import get_emission_analysis
from agents.tools.seai_policy_search import run_curl_search

from autogen_core.tools import FunctionTool

emission_tool = FunctionTool(
    func=get_emission_analysis,
    description="Gets the CO2 intensity levels. Parameters: startdate (YYYY-MM-DD), enddate (YYYY-MM-DD), region ('all', 'roi', or 'ni')",
    name="emission_tool",
)

policy_search_tool = FunctionTool(
    func=run_curl_search,
    description="Searches SEAI policy documents for information about energy grants, schemes, and policies. Use quoted search terms for exact matching (e.g., 'solar grants', 'BER assessment', 'heat pump eligibility')",
    name="run_curl_search",
)
