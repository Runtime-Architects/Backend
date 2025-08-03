import os
from agents import agent_builder, agent_sysmsgs
from agents.agent_tools import emission_tool, policy_search_tool
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.agents import (
    AssistantAgent,
    MessageFilterAgent,
    MessageFilterConfig,
    PerSourceFilter,
)
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
import logging
import asyncio

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from agents.client import AzureClientFactory

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

azure_factory = AzureClientFactory(
    azure_deployment=os.environ["AZURE_AI_DEPLOYMENT"],
    model=os.environ["AZURE_AI_MODEL"],
    api_version=os.environ["AZURE_AI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_AI_ENDPOINT"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    max_completion_tokens=1024,
)

client = azure_factory.get_client()

async def initialize_agents():
    """Initialize AutoGen agents and workflow."""
    async with LocalCommandLineCodeExecutor(work_dir="coding") as executor:
        tool = PythonCodeExecutionTool(executor)

        try:
            # --- Agent setup context ---
            context = {
                "model_client": client,
                "emission_tool": emission_tool,
                "policy_search_tool": policy_search_tool,
                "CARBON_AGENT_SYSMSG": agent_sysmsgs.CARBON_AGENT_SYSMSG,
                "POLICY_AGENT_SYSMSG": agent_sysmsgs.POLICY_AGENT_SYSMSG,
                "DATA_ANALYSIS_AGENT_SYSMSG": agent_sysmsgs.DATA_ANALYSIS_AGENT_SYSMSG,
                "PLANNER_AGENT_SYSMSG": agent_sysmsgs.PLANNER_AGENT_SYSMSG,
                "REPORT_AGENT_SYSMSG": agent_sysmsgs.REPORT_AGENT_SYSMSG,
            }

            # --- Config path ---
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

            # --- Create factory and build agents ---
            factory = agent_builder.AgentFactory(config_path=config_path, context=context)
            agents = factory.build_agents(agent_class=AssistantAgent)

            # --- Extract agents by name ---
            agent_map = {agent.name: agent for agent in agents}
            logger.info(f"Agents created: {[agent.name for agent in agents]}")

            planner = agent_map["PlannerAgent"]
            report = agent_map["ReportAgent"]
            carbon = agent_map["CarbonAgent"]
            policy = agent_map["PolicyAgent"]
            analysis = agent_map["DataAnalysisAgent"]

            # --- Message filtering ---
            def create_conditional_filter(source_agent):
                return MessageFilterConfig(
                    per_source=[PerSourceFilter(source=source_agent, position="last", count=1)]
                )

            filtered_carbon = MessageFilterAgent(
                name="CarbonAgent", wrapped_agent=carbon,
                filter=create_conditional_filter("PlannerAgent")
            )
            filtered_policy = MessageFilterAgent(
                name="PolicyAgent", wrapped_agent=policy,
                filter=create_conditional_filter("PlannerAgent")
            )
            filtered_analysis = MessageFilterAgent(
                name="DataAnalysisAgent", wrapped_agent=analysis,
                filter=create_conditional_filter("PlannerAgent")
            )

            # --- Build workflow graph ---
            logger.info("Building workflow graph...")
            builder = DiGraphBuilder()
            builder.add_node(planner)
            builder.add_node(filtered_carbon)
            builder.add_node(filtered_policy)
            builder.add_node(filtered_analysis)
            builder.add_node(report)

            builder.add_edge(planner, filtered_carbon)
            builder.add_edge(planner, filtered_policy)
            builder.add_edge(planner, filtered_analysis)

            builder.add_edge(filtered_carbon, report)
            builder.add_edge(filtered_policy, report)
            builder.add_edge(filtered_analysis, report)

            team_flow = GraphFlow(
                participants=builder.get_participants(),
                graph=builder.build(),
            )

            logger.info("AutoGen agents initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

        return team_flow



if __name__ == "__main__":
    asyncio.run(initialize_agents())
