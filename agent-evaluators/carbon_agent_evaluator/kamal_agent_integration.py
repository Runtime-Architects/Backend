"""
Helper to integrate Kamal's azurecarbonagent.py with the evaluation system
Handles import issues and provides fallback implementations
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from autogen_agentchat.agents import AssistantAgent
from azure_client_factory import create_azure_client
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv

# Setup logger
logger = logging.getLogger(__name__)

load_dotenv()

class KamalAgentIntegrator:
    """
    Integrates Kamal's carbon agent with the evaluation system
    Handles various import scenarios and provides compatible implementations
    """
    
    def __init__(self):
        self.client = self._setup_azure_client()
        self.agent = None
        self.integration_method = None
    
    def _setup_azure_client(self):
        """Setup Azure OpenAI client"""
        return create_azure_client()
    
    async def get_carbon_agent(self) -> AssistantAgent:
        """
        Get Kamal's carbon agent using multiple integration approaches
        Returns a working carbon agent instance
        """
        
        # Method 1: Try direct import of carbon_agent instance
        try:
            from azurecarbonagent import carbon_agent
            self.agent = carbon_agent
            self.integration_method = "direct_instance"
            print("✅ Method 1 Success: Using Kamal's carbon_agent directly")
            return self.agent
        except ImportError as e:
            print(f"⚠️ Method 1 Failed: {e}")
        except Exception as e:
            print(f"⚠️ Method 1 Error: {e}")
        
        # Method 2: Try importing components and recreating agent
        try:
            from azurecarbonagent import (
                emission_tool, daily_analyzer_tool, weekly_analyzer_tool, 
                monthly_analyzer_tool, system_message
            )
            
            self.agent = AssistantAgent(
                name="CarbonAgent", 
                model_client=self.client, 
                tools=[emission_tool, daily_analyzer_tool, weekly_analyzer_tool, monthly_analyzer_tool], 
                reflect_on_tool_use=True,
                max_tool_iterations=5,
                system_message=system_message
            )
            self.integration_method = "component_recreation"
            print("✅ Method 2 Success: Recreated agent from Kamal's components")
            return self.agent
            
        except ImportError as e:
            print(f"⚠️ Method 2 Failed: {e}")
        except Exception as e:
            print(f"⚠️ Method 2 Error: {e}")
        
        # Method 3: Try importing just the tools and create compatible agent
        try:
            from azurecarbonagent import emission_tool
            
            # Try to get system message
            try:
                from azurecarbonagent import system_message
            except ImportError:
                system_message = self._create_compatible_system_message()
            
            self.agent = AssistantAgent(
                name="CarbonAgent", 
                model_client=self.client, 
                tools=[emission_tool], 
                reflect_on_tool_use=True,
                max_tool_iterations=3,
                system_message=system_message
            )
            self.integration_method = "minimal_tools"
            print("✅ Method 3 Success: Using Kamal's emission_tool with compatible setup")
            return self.agent
            
        except ImportError as e:
            print(f"⚠️ Method 3 Failed: {e}")
        except Exception as e:
            print(f"⚠️ Method 3 Error: {e}")
        
        # All real integration methods failed
        error_msg = ("All real agent integration methods failed. Cannot create mock/fake agent for evaluation. "
                    "Please ensure azurecarbonagent.py is properly configured with real CO2 analysis tools.")
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _create_compatible_system_message(self) -> str:
        """Create compatible system message based on Kamal's design"""
        return f"""You are an intelligent carbon emissions assistant specialized in Ireland's electricity grid. Today's date and time is: {datetime.now()}.

**ACTIVATION CONDITIONS:** Only respond when specifically instructed by the PlannerAgent or when directly queried about carbon emissions.

### Available Tools:
- **Carbon Data Retriever**: Fetches raw CO2 intensity data (use when you need current emissions data)
- **Daily Analyzer**: For analysis day/days (15 minute granularity) 
- **Weekly Analyzer**: For analysis week/weeks (hourly granularity)
- **Monthly Analyzer**: For analysis of month/months (day granularity)

TOOL USAGE RULES:
- For CO2 intensity queries, ALWAYS use the emission_tool with these exact parameters:
  - Date format MUST be YYYY-MM-DD (e.g., '2025-06-24')
  - Region MUST be one of:
    * 'roi' for Republic of Ireland (Ireland)
    * 'ni' for Northern Ireland
    * 'all' for both Republic of Ireland (Ireland) & Northern Ireland

- **Time Period** determines which analyzer to use:
    - 1 day to 6 days → Daily Analyzer
    - 7 days to 21 days → Weekly Analyzer
    - greater than 21 days → Monthly Analyzer

When providing recommendations for the current day, always consider the current time. 
Only suggest activities or actions for future time slots—never for times that have already passed.

### Response Format Guidelines:
1. Start with analysis type and time period covered
2. Show key findings with emojis (🌱 for low, ⚠️ for medium, 🔥 for high emissions)
3. Provide actionable recommendations
4. Include any notable trends or comparisons

Your findings should be communicated clearly with structured formatting and visual indicators.
"""
    
    # Removed _create_compatible_agent method and all mock functions - no more fake agent creation
    
    def get_integration_info(self) -> Dict[str, str]:
        """Get information about how the agent was integrated"""
        methods = {
            "direct_instance": "Using Kamal's carbon_agent directly",
            "component_recreation": "Recreated from Kamal's components",
            "minimal_tools": "Using Kamal's emission_tool with compatibility layer",
            # "compatible_mock" method removed - no more mock agents
        }
        
        return {
            "method": self.integration_method,
            "description": methods.get(self.integration_method, "Unknown"),
            "agent_name": self.agent.name if self.agent else "None",
            "tools_available": len(self.agent.tools) if self.agent and hasattr(self.agent, 'tools') else 0
        }

# Standalone function for easy integration
async def create_kamal_carbon_agent() -> AssistantAgent:
    """
    Convenience function to create Kamal's carbon agent
    Handles all integration complexity automatically
    """
    integrator = KamalAgentIntegrator()
    agent = await integrator.get_carbon_agent()
    
    # Print integration info
    info = integrator.get_integration_info()
    print(f"🔧 Integration Method: {info['description']}")
    print(f"🤖 Agent Name: {info['agent_name']}")
    print(f"🛠️  Tools Available: {info['tools_available']}")
    
    return agent

# Test function
async def test_kamal_agent():
    """Test function to verify the agent works correctly"""
    print("🧪 Testing Kamal's Carbon Agent Integration...")
    
    try:
        agent = await create_kamal_carbon_agent()
        
        # Test with a simple query
        test_query = "What is the best time to use my appliances today in Ireland?"
        print(f"📝 Test Query: {test_query}")
        
        if hasattr(agent, 'run_stream'):
            print("🔄 Agent Response:")
            async for result in agent.run_stream(task=test_query):
                print(result)
        else:
            result = await agent.run(test_query)
            print(f"🔄 Agent Response: {result}")
        
        print("✅ Integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the integration
    asyncio.run(test_kamal_agent())