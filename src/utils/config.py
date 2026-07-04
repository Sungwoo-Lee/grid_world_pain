import yaml
import os


class _FlowListDumper(yaml.SafeDumper):
    pass


def _represent_list_flow_if_scalar(dumper, data):
    # Inline (flow style) only when every element is a scalar; nested lists/dicts stay block.
    scalar = all(isinstance(x, (int, float, str, bool, type(None))) for x in data)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=scalar)


_FlowListDumper.add_representer(list, _represent_list_flow_if_scalar)


def dump_config_yaml(data, stream):
    """Dump a plain dict (e.g. Config.to_dict()) to YAML: scalar lists inline,
    all mappings block, source key order preserved."""
    yaml.dump(data, stream, Dumper=_FlowListDumper,
              default_flow_style=False, sort_keys=False)


class Config:
    def __init__(self, config_dict=None):
        self._config = config_dict or {}

    @classmethod
    def load_yaml(cls, path):
        if not os.path.exists(path):
            # Strict Config (H3, diag_fable5_20260704/01 Finding 5): a missing
            # config file must be a hard error, never a silent empty config —
            # a typo'd --config used to train silently on default.yaml.
            # Intentional optional loads must guard with os.path.exists at the
            # call site (all existing ones already do).
            raise FileNotFoundError(
                f"Config file not found: {path!r}. "
                f"Optional loads must check os.path.exists before calling load_yaml.")
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
        
    def get_mandatory(self, key, type_converter=None):
        """
        Retrieves a value from the configuration. Raises ValueError if the key is missing.
        Optional type_converter can be passed to cast the value (e.g., int, float).
        """
        val = self.get(key)
        if val is None:
            raise ValueError(f"Strict Config: Configuration key '{key}' is required but missing.")
        
        if type_converter:
            try:
                val = type_converter(val)
            except ValueError as e:
                raise ValueError(f"Strict Config: Failed to convert key '{key}' value '{val}' to {type_converter.__name__}: {e}")
        return val

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
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "environment", "default.yaml")

def get_default_config():
    # Adjust path if needed based on package structure
    abs_path = os.path.abspath(DEFAULT_CONFIG_PATH)
    return Config.load_yaml(abs_path)
