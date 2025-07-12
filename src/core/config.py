"""Configuration management for the email categorization agent."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, model_validator
from pydantic_settings import BaseSettings
import yaml
import structlog
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""
    host: str = "http://localhost:11434"
    models: Dict[str, str] = Field(default_factory=lambda: {
        "primary": "gemma3:4b",
        "fallback": "llama3.2:3b",
        "reasoning": "llama3.2:3b"
    })
    timeout: int = 60
    max_retries: int = 3
    keep_alive: bool = True


class LLMConfig(BaseModel):
    """LLM generation parameters."""
    temperature: float = 0.1
    top_p: float = 0.95
    num_predict: int = 100
    max_tokens: int = 150


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


class EmailConfig(BaseModel):
    """Email processing configuration."""
    default_limit: int = 50
    max_results: int = 1000
    cache_ttl: int = 3600
    cache_max_size: int = 1000


class ClassificationConfig(BaseModel):
    """Classification threshold configuration."""
    priority: Dict[str, float] = Field(default_factory=lambda: {
        "high_confidence": 0.8,
        "medium_confidence": 0.6
    })
    marketing_threshold: float = 0.7
    receipt_threshold: float = 0.7
    notifications_threshold: float = 0.7
    custom_threshold: float = 0.7


class CategorizationConfig(BaseModel):
    """Email categorization configuration."""
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_suggestions: int = Field(default=3, ge=1, le=10)
    enable_new_label_creation: bool = True


class SQLiteConfig(BaseModel):
    """SQLite database configuration."""
    database_path: str = "./data/agent_state.db"
    connection_pool_size: int = 5
    pool_timeout: float = 10.0
    max_overflow: int = 10
    busy_timeout: int = 30000  # milliseconds
    journal_mode: str = "WAL"


class StorageConfig(BaseModel):
    """Email storage configuration."""
    cache_expiry: int = 3600
    max_email_size: int = 10485760  # 10MB
    compression_enabled: bool = True


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
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    categorization: CategorizationConfig = Field(default_factory=CategorizationConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
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
        
    @model_validator(mode='after')
    def validate_configuration(self):
        """Validate interdependent configuration fields."""
        logger.debug("Performing configuration cross-field validation")
        
        # Extract configuration sections
        app = self.app
        ollama = self.ollama
        gmail = self.gmail
        categorization = self.categorization
        sqlite = self.sqlite
        state = self.state
        mcp = self.mcp
        logging = self.logging
        performance = self.performance
        security = self.security
        
        # Validate file paths exist (for required files)
        if gmail and hasattr(gmail, 'credentials_path'):
            cred_path = Path(gmail.credentials_path)
            if not cred_path.exists() and app and not app.debug:
                logger.warning(
                    "Gmail credentials file not found",
                    path=str(cred_path),
                    suggestion="Run authentication setup first"
                )
        
        # Validate Ollama models consistency
        if ollama and hasattr(ollama, 'models'):
            models = ollama.models
            if 'primary' not in models:
                raise ValueError("Ollama primary model must be specified")
            
            # Ensure all model names are valid
            for model_type, model_name in models.items():
                if not model_name or not isinstance(model_name, str):
                    raise ValueError(f"Invalid model name for {model_type}: {model_name}")
        
        # Validate rate limiting consistency
        if gmail and hasattr(gmail, 'rate_limit') and hasattr(gmail, 'batch_size'):
            rate_limit = gmail.rate_limit
            batch_size = gmail.batch_size
            
            if 'requests_per_second' in rate_limit:
                max_rps = rate_limit['requests_per_second']
                if batch_size > max_rps * 10:  # 10 seconds worth
                    logger.warning(
                        "Gmail batch size may exceed rate limits",
                        batch_size=batch_size,
                        max_rps=max_rps,
                        suggestion="Consider reducing batch_size"
                    )
        
        # Validate database configuration
        if sqlite and hasattr(sqlite, 'database_path'):
            db_path = Path(sqlite.database_path)
            db_dir = db_path.parent
            
            # Ensure database directory exists or can be created
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ValueError(f"Cannot create database directory: {db_dir}")
            
            # Validate connection pool size
            if hasattr(sqlite, 'connection_pool_size'):
                pool_size = sqlite.connection_pool_size
                if pool_size < 1 or pool_size > 20:
                    raise ValueError(f"Invalid connection pool size: {pool_size}. Must be 1-20")
        
        # Validate confidence threshold consistency
        if categorization and hasattr(categorization, 'confidence_threshold'):
            threshold = categorization.confidence_threshold
            if app and app.mode == 'automatic' and threshold < 0.8:
                logger.warning(
                    "Low confidence threshold for automatic mode",
                    threshold=threshold,
                    mode=app.mode,
                    suggestion="Consider raising threshold for automatic mode"
                )
        
        # Validate monitoring configuration
        if performance and state:
            if hasattr(performance, 'health_check_interval') and hasattr(state, 'checkpoint_interval'):
                health_interval = performance.health_check_interval
                checkpoint_interval = state.checkpoint_interval
                
                if health_interval > checkpoint_interval:
                    logger.warning(
                        "Health check interval longer than checkpoint interval",
                        health_interval=health_interval,
                        checkpoint_interval=checkpoint_interval,
                        suggestion="Consider more frequent health checks"
                    )
        
        # Validate logging configuration
        if logging and hasattr(logging, 'output'):
            log_path = Path(logging.output)
            log_dir = log_path.parent
            
            # Ensure log directory exists or can be created
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ValueError(f"Cannot create log directory: {log_dir}")
        
        # Validate security settings
        if security and gmail:
            if hasattr(security, 'credential_storage') and hasattr(gmail, 'credentials_path'):
                if security.credential_storage == 'file':
                    cred_path = Path(gmail.credentials_path)
                    if cred_path.exists():
                        # Check file permissions
                        stat_info = cred_path.stat()
                        # Warn if file is readable by others (Unix-like systems)
                        if hasattr(stat_info, 'st_mode') and (stat_info.st_mode & 0o044):
                            logger.warning(
                                "Credential file has broad read permissions",
                                path=str(cred_path),
                                suggestion="Consider restricting file permissions"
                            )
        
        # Validate MCP configuration
        if mcp and hasattr(mcp, 'port'):
            port = mcp.port
            # Check for privileged ports on Unix-like systems
            if port < 1024 and hasattr(os, 'geteuid'):
                try:
                    if os.geteuid() != 0:
                        logger.warning(
                            "MCP port requires root privileges",
                            port=port,
                            suggestion="Use port > 1024 or run as root"
                        )
                except AttributeError:
                    pass  # Not a Unix system
        
        logger.info("Configuration validation completed successfully")
        return self
        
    def validate_startup_requirements(self) -> Dict[str, Any]:
        """Validate configuration for startup readiness and return status report."""
        logger.info("Performing startup configuration validation")
        
        validation_results = {
            "status": "valid",
            "errors": [],
            "warnings": [],
            "checks": []
        }
        
        # Check database accessibility
        try:
            db_path = Path(self.sqlite.database_path)
            db_dir = db_path.parent
            
            if not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
                validation_results["checks"].append("✅ Database directory created")
            else:
                validation_results["checks"].append("✅ Database directory exists")
                
            # Test write access
            test_file = db_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
                validation_results["checks"].append("✅ Database directory writable")
            except PermissionError:
                validation_results["errors"].append("❌ No write access to database directory")
                validation_results["status"] = "error"
                
        except Exception as e:
            validation_results["errors"].append(f"❌ Database path validation failed: {e}")
            validation_results["status"] = "error"
        
        # Check log directory
        try:
            log_path = Path(self.logging.output)
            log_dir = log_path.parent
            
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
                validation_results["checks"].append("✅ Log directory created")
            else:
                validation_results["checks"].append("✅ Log directory exists")
                
        except Exception as e:
            validation_results["warnings"].append(f"⚠️ Log directory issue: {e}")
            if validation_results["status"] == "valid":
                validation_results["status"] = "warning"
        
        # Check Ollama configuration
        try:
            models = self.ollama.models
            if not models.get('primary'):
                validation_results["errors"].append("❌ No primary Ollama model configured")
                validation_results["status"] = "error"
            else:
                validation_results["checks"].append(f"✅ Primary model: {models['primary']}")
                
        except Exception as e:
            validation_results["errors"].append(f"❌ Ollama configuration error: {e}")
            validation_results["status"] = "error"
        
        # Check Gmail credentials
        try:
            cred_path = Path(self.gmail.credentials_path)
            if cred_path.exists():
                validation_results["checks"].append("✅ Gmail credentials file found")
            else:
                validation_results["warnings"].append(f"⚠️ Gmail credentials not found: {cred_path}")
                if validation_results["status"] == "valid":
                    validation_results["status"] = "warning"
                    
        except Exception as e:
            validation_results["warnings"].append(f"⚠️ Gmail credentials check failed: {e}")
            if validation_results["status"] == "valid":
                validation_results["status"] = "warning"
        
        # Validate configuration consistency
        if self.app.mode == "automatic" and self.categorization.confidence_threshold < 0.8:
            validation_results["warnings"].append(
                f"⚠️ Low confidence threshold ({self.categorization.confidence_threshold}) for automatic mode"
            )
            if validation_results["status"] == "valid":
                validation_results["status"] = "warning"
        
        # Check performance configuration
        if self.performance.enable_monitoring:
            validation_results["checks"].append("✅ Performance monitoring enabled")
        else:
            validation_results["warnings"].append("⚠️ Performance monitoring disabled")
            if validation_results["status"] == "valid":
                validation_results["status"] = "warning"
        
        logger.info(
            "Startup validation completed",
            status=validation_results["status"],
            errors=len(validation_results["errors"]),
            warnings=len(validation_results["warnings"]),
            checks=len(validation_results["checks"])
        )
        
        return validation_results

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