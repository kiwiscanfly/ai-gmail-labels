"""Configuration management for the email categorization agent."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import yaml
from dotenv import load_dotenv


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""
    host: str = "http://localhost:11434"
    models: Dict[str, str] = Field(default_factory=lambda: {
        "primary": "gemma2:3b",
        "fallback": "llama3.2:3b",
        "reasoning": "llama3.2:3b"
    })
    timeout: int = 60
    max_retries: int = 3
    keep_alive: bool = True


class GmailConfig(BaseModel):
    """Gmail API configuration."""
    credentials_path: str = "./credentials.json"
    token_path: str = "./token.json"
    scopes: List[str] = Field(default_factory=lambda: [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.labels"
    ])
    batch_size: int = 50
    rate_limit: Dict[str, int] = Field(default_factory=lambda: {
        "requests_per_second": 10,
        "quota_per_user_per_second": 250
    })


class CategorizationConfig(BaseModel):
    """Email categorization configuration."""
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_suggestions: int = Field(default=3, ge=1, le=10)
    enable_new_label_creation: bool = True


class SQLiteConfig(BaseModel):
    """SQLite database configuration."""
    database_path: str = "./data/agent_state.db"
    connection_pool_size: int = 5
    busy_timeout: int = 30000  # milliseconds
    journal_mode: str = "WAL"


class StateConfig(BaseModel):
    """State management configuration."""
    checkpoint_interval: int = 60  # seconds
    cleanup_interval: int = 86400  # seconds (24 hours)
    max_history_days: int = 30


class MCPConfig(BaseModel):
    """MCP server configuration."""
    host: str = "localhost"
    port: int = 8080
    auth_token: Optional[str] = None
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    output: str = "./logs/agent.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    structured: bool = True

    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}. Must be one of {valid_levels}')
        return v.upper()


class PerformanceConfig(BaseModel):
    """Performance monitoring configuration."""
    enable_monitoring: bool = True
    metrics_retention_days: int = 7
    health_check_interval: int = 30  # seconds


class SecurityConfig(BaseModel):
    """Security configuration."""
    encrypt_state: bool = False
    credential_storage: str = "keyring"  # keyring | file
    max_login_attempts: int = 3
    session_timeout: int = 3600  # seconds

    @validator('credential_storage')
    def validate_credential_storage(cls, v):
        valid_options = ['keyring', 'file']
        if v not in valid_options:
            raise ValueError(f'Invalid credential storage: {v}. Must be one of {valid_options}')
        return v


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = "Email Categorization Agent"
    version: str = "1.0.0"
    mode: str = "interactive"  # automatic | interactive
    debug: bool = False

    @validator('mode')
    def validate_mode(cls, v):
        valid_modes = ['automatic', 'interactive']
        if v not in valid_modes:
            raise ValueError(f'Invalid mode: {v}. Must be one of {valid_modes}')
        return v


class Config(BaseSettings):
    """Main configuration class."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    categorization: CategorizationConfig = Field(default_factory=CategorizationConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"

    @classmethod
    def load_from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration from YAML file with environment variable overrides."""
        if config_path is None:
            config_path = Path("config/default.yaml")
        
        config_path = Path(config_path)
        
        # Load environment variables
        load_dotenv()
        
        # Load base configuration from YAML
        config_data = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
        
        # Override with environment variables
        env_overrides = cls._get_env_overrides()
        config_data = cls._merge_configs(config_data, env_overrides)
        
        return cls(**config_data)

    @staticmethod
    def _get_env_overrides() -> Dict[str, Any]:
        """Extract configuration overrides from environment variables."""
        overrides = {}
        
        # Map environment variables to configuration keys
        env_mapping = {
            'OLLAMA_HOST': 'ollama.host',
            'OLLAMA_NUM_THREADS': 'ollama.num_threads',
            'OLLAMA_MAX_LOADED_MODELS': 'ollama.max_loaded_models',
            'GMAIL_CREDENTIALS_PATH': 'gmail.credentials_path',
            'GMAIL_TOKEN_PATH': 'gmail.token_path',
            'SQLITE_DATABASE_PATH': 'sqlite.database_path',
            'SQLITE_BUSY_TIMEOUT': 'sqlite.busy_timeout',
            'SQLITE_JOURNAL_MODE': 'sqlite.journal_mode',
            'APP_MODE': 'app.mode',
            'LOG_LEVEL': 'logging.level',
            'CONFIDENCE_THRESHOLD': 'categorization.confidence_threshold',
            'MCP_HOST': 'mcp.host',
            'MCP_PORT': 'mcp.port',
            'MCP_AUTH_TOKEN': 'mcp.auth_token',
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert types
                if config_key in ['mcp.port', 'sqlite.busy_timeout', 'ollama.num_threads', 'ollama.max_loaded_models']:
                    value = int(value)
                elif config_key in ['categorization.confidence_threshold']:
                    value = float(value)
                elif config_key in ['app.debug']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                # Set nested configuration
                keys = config_key.split('.')
                current = overrides
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
        
        return overrides

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            Path(self.sqlite.database_path).parent,
            Path(self.logging.output).parent,
            Path(self.gmail.credentials_path).parent,
            Path(self.gmail.token_path).parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_database_url(self) -> str:
        """Get SQLite database URL."""
        return f"sqlite:///{self.sqlite.database_path}"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.app.debug and self.logging.level in ['INFO', 'WARNING', 'ERROR']

    def validate_paths(self) -> List[str]:
        """Validate that required paths exist and are accessible."""
        errors = []
        
        # Check credentials path
        if not Path(self.gmail.credentials_path).exists():
            errors.append(f"Gmail credentials file not found: {self.gmail.credentials_path}")
        
        # Check if directories are writable
        try:
            Path(self.sqlite.database_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create database directory: {e}")
        
        try:
            Path(self.logging.output).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create logs directory: {e}")
        
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app": self.app.dict(),
            "ollama": self.ollama.dict(),
            "gmail": self.gmail.dict(),
            "categorization": self.categorization.dict(),
            "sqlite": self.sqlite.dict(),
            "state": self.state.dict(),
            "mcp": self.mcp.dict(),
            "logging": self.logging.dict(),
            "performance": self.performance.dict(),
            "security": self.security.dict(),
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load_from_yaml()
    return _config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config.load_from_yaml(config_path)
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance (mainly for testing)."""
    global _config
    _config = config