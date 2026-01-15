import yaml
import os

class Config:
    def __init__(self, config_dict=None):
        self._config = config_dict or {}

    @classmethod
    def load_yaml(cls, path):
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found. Using defaults.")
            return cls()
        with open(path, 'r') as f:
            return cls(yaml.safe_load(f))

    def get(self, key, default=None):
        # Support nested keys like 'environment.height'
        keys = key.split('.')
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key, value):
        # Support nested keys like 'environment.height'
        keys = key.split('.')
        val = self._config
        for i, k in enumerate(keys[:-1]):
            if k not in val:
                val[k] = {}
            val = val[k]
        val[keys[-1]] = value

    def to_dict(self):
        return self._config

    def merge(self, other_config):
        """
        Merges another configuration dictionary or Config object into this one.
        Deep merge is preferred for nested configs.
        """
        if isinstance(other_config, Config):
            other = other_config.to_dict()
        else:
            other = other_config

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self._config = deep_update(self._config, other)

# Global instance for default config
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default_configs.yaml")

def get_default_config():
    # Adjust path if needed based on package structure
    abs_path = os.path.abspath(DEFAULT_CONFIG_PATH)
    return Config.load_yaml(abs_path)
