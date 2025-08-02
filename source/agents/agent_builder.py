import os
import yaml
import inspect

class AgentFactory:
    def __init__(self, config_path, context=None):
        self.context = context or {}
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        self.agent_configs = raw_config.get("agents", [])
        self.resolved_agents = self._resolve_all(self.agent_configs)

    def _resolve_value(self, value):
        """Resolve a single value based on env/context."""
        if isinstance(value, str):
            # Handle ${VAR} syntax
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                resolved = self.context.get(var_name) or os.environ.get(var_name)
                if resolved is None:
                    raise ValueError(f"Missing value for '{var_name}' in context or environment.")
                return resolved
            # Handle tool name as string
            elif value in self.context:
                return self.context[value]
            else:
                return value
        elif isinstance(value, list):
            return [self._resolve_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        else:
            return value

    def _resolve_all(self, config_list):
        """Resolve all env/context references in config."""
        return [self._resolve_value(entry) for entry in config_list]

    def build_agents(self, agent_class, extra_args=None):
        """Instantiate agents using resolved config and extra args."""
        extra_args = extra_args or {}
        sig = inspect.signature(agent_class.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}

        agents = []
        for cfg in self.resolved_agents:
            combined = {**cfg, **extra_args}
            filtered = {k: v for k, v in combined.items() if k in valid_keys}
            agent = agent_class(**filtered)
            agents.append(agent)
        return agents
