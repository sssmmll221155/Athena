"""
Parser Configuration
Configuration settings for code parser agent.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParserConfig:
    """
    Configuration for code parser.
    All settings can be overridden via environment variables.
    """

    # File Processing Limits
    MAX_FILE_SIZE_MB: int = 1  # Maximum file size to parse in MB
    MAX_FILE_SIZE_BYTES: int = field(init=False)  # Computed from MAX_FILE_SIZE_MB

    # Concurrency Settings
    CONCURRENT_FILES: int = 50  # Maximum concurrent file fetches
    BATCH_INSERT_SIZE: int = 100  # Number of rows to insert per batch

    # Language Support
    SUPPORTED_LANGUAGES: List[str] = field(default_factory=lambda: ['python'])

    # Parser Settings
    PARSER_VERSION: str = '1.0.0'
    SKIP_BINARY_FILES: bool = True
    SKIP_TEST_FILES: bool = False  # Whether to skip test files

    # GitHub API Settings
    GITHUB_API_TIMEOUT: int = 30  # Request timeout in seconds
    GITHUB_RATE_LIMIT_DELAY: float = 1.0  # Delay between requests

    # Database Settings
    DB_COMMIT_BATCH_SIZE: int = 10  # Commit after this many files

    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Feature Flags
    ENABLE_RADON_COMPLEXITY: bool = True  # Use radon for complexity metrics
    ENABLE_DOCSTRING_EXTRACTION: bool = True
    ENABLE_FUNCTION_CALLS_TRACKING: bool = True

    # File Extensions
    PYTHON_EXTENSIONS: List[str] = field(default_factory=lambda: ['.py', '.pyw'])
    BINARY_EXTENSIONS: List[str] = field(default_factory=lambda: [
        '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.bin',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.pdf', '.zip',
        '.tar', '.gz', '.7z', '.rar', '.whl', '.egg'
    ])

    # Test File Patterns
    TEST_FILE_PATTERNS: List[str] = field(default_factory=lambda: [
        'test_*.py',
        '*_test.py',
        'tests/*.py',
        'test/*.py',
        'testing/*.py',
    ])

    def __post_init__(self):
        """Post-initialization to set computed fields and load from environment"""
        # Compute derived values
        self.MAX_FILE_SIZE_BYTES = self.MAX_FILE_SIZE_MB * 1024 * 1024

        # Load from environment variables if present
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # File Processing
        if 'PARSER_MAX_FILE_SIZE_MB' in os.environ:
            self.MAX_FILE_SIZE_MB = int(os.environ['PARSER_MAX_FILE_SIZE_MB'])
            self.MAX_FILE_SIZE_BYTES = self.MAX_FILE_SIZE_MB * 1024 * 1024

        # Concurrency
        if 'PARSER_CONCURRENT_FILES' in os.environ:
            self.CONCURRENT_FILES = int(os.environ['PARSER_CONCURRENT_FILES'])

        if 'PARSER_BATCH_INSERT_SIZE' in os.environ:
            self.BATCH_INSERT_SIZE = int(os.environ['PARSER_BATCH_INSERT_SIZE'])

        # Database
        if 'PARSER_DB_COMMIT_BATCH_SIZE' in os.environ:
            self.DB_COMMIT_BATCH_SIZE = int(os.environ['PARSER_DB_COMMIT_BATCH_SIZE'])

        # Logging
        if 'PARSER_LOG_LEVEL' in os.environ:
            self.LOG_LEVEL = os.environ['PARSER_LOG_LEVEL']

        # Feature Flags
        if 'PARSER_ENABLE_RADON' in os.environ:
            self.ENABLE_RADON_COMPLEXITY = os.environ['PARSER_ENABLE_RADON'].lower() == 'true'

        if 'PARSER_SKIP_TEST_FILES' in os.environ:
            self.SKIP_TEST_FILES = os.environ['PARSER_SKIP_TEST_FILES'].lower() == 'true'

        # GitHub API
        if 'PARSER_GITHUB_TIMEOUT' in os.environ:
            self.GITHUB_API_TIMEOUT = int(os.environ['PARSER_GITHUB_TIMEOUT'])

        if 'PARSER_GITHUB_RATE_LIMIT_DELAY' in os.environ:
            self.GITHUB_RATE_LIMIT_DELAY = float(os.environ['PARSER_GITHUB_RATE_LIMIT_DELAY'])

    def is_supported_language(self, language: str) -> bool:
        """Check if language is supported"""
        return language.lower() in [lang.lower() for lang in self.SUPPORTED_LANGUAGES]

    def is_binary_extension(self, extension: str) -> bool:
        """Check if file extension indicates binary file"""
        return extension.lower() in [ext.lower() for ext in self.BINARY_EXTENSIONS]

    def is_test_file(self, file_path: str) -> bool:
        """Check if file appears to be a test file"""
        import fnmatch
        file_path_lower = file_path.lower()

        for pattern in self.TEST_FILE_PATTERNS:
            if fnmatch.fnmatch(file_path_lower, pattern.lower()):
                return True

        return False

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'MAX_FILE_SIZE_MB': self.MAX_FILE_SIZE_MB,
            'MAX_FILE_SIZE_BYTES': self.MAX_FILE_SIZE_BYTES,
            'CONCURRENT_FILES': self.CONCURRENT_FILES,
            'BATCH_INSERT_SIZE': self.BATCH_INSERT_SIZE,
            'SUPPORTED_LANGUAGES': self.SUPPORTED_LANGUAGES,
            'PARSER_VERSION': self.PARSER_VERSION,
            'SKIP_BINARY_FILES': self.SKIP_BINARY_FILES,
            'SKIP_TEST_FILES': self.SKIP_TEST_FILES,
            'GITHUB_API_TIMEOUT': self.GITHUB_API_TIMEOUT,
            'GITHUB_RATE_LIMIT_DELAY': self.GITHUB_RATE_LIMIT_DELAY,
            'DB_COMMIT_BATCH_SIZE': self.DB_COMMIT_BATCH_SIZE,
            'LOG_LEVEL': self.LOG_LEVEL,
            'ENABLE_RADON_COMPLEXITY': self.ENABLE_RADON_COMPLEXITY,
            'ENABLE_DOCSTRING_EXTRACTION': self.ENABLE_DOCSTRING_EXTRACTION,
            'ENABLE_FUNCTION_CALLS_TRACKING': self.ENABLE_FUNCTION_CALLS_TRACKING,
        }

    def __repr__(self):
        """String representation"""
        return f"ParserConfig(languages={self.SUPPORTED_LANGUAGES}, concurrent={self.CONCURRENT_FILES})"


# ============================================================================
# Database Configuration
# ============================================================================

@dataclass
class DatabaseConfig:
    """
    Database connection configuration.
    Loads from environment variables.
    """

    host: str = 'localhost'
    port: int = 5432
    database: str = 'athena'
    user: str = 'athena'
    password: str = ''
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    echo: bool = False

    def __post_init__(self):
        """Load from environment variables"""
        self.host = os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', self.host))
        self.port = int(os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', self.port)))
        self.database = os.getenv('POSTGRES_DB', os.getenv('DB_NAME', self.database))
        self.user = os.getenv('POSTGRES_USER', os.getenv('DB_USER', self.user))
        self.password = os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASSWORD', self.password))

        if 'DB_POOL_SIZE' in os.environ:
            self.pool_size = int(os.environ['DB_POOL_SIZE'])

        if 'DB_MAX_OVERFLOW' in os.environ:
            self.max_overflow = int(os.environ['DB_MAX_OVERFLOW'])

        if 'DB_ECHO' in os.environ:
            self.echo = os.environ['DB_ECHO'].lower() == 'true'

    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_async_connection_string(self) -> str:
        """Get async SQLAlchemy connection string"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def __repr__(self):
        """String representation (hide password)"""
        return f"DatabaseConfig(host={self.host}, port={self.port}, database={self.database}, user={self.user})"


# ============================================================================
# GitHub Configuration
# ============================================================================

@dataclass
class GitHubConfig:
    """
    GitHub API configuration.
    Loads from environment variables.
    """

    token: str = ''
    base_url: str = 'https://api.github.com'
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0

    def __post_init__(self):
        """Load from environment variables"""
        self.token = os.getenv('GITHUB_TOKEN', self.token)

        if 'GITHUB_API_URL' in os.environ:
            self.base_url = os.environ['GITHUB_API_URL']

        if 'GITHUB_API_TIMEOUT' in os.environ:
            self.timeout = int(os.environ['GITHUB_API_TIMEOUT'])

        if 'GITHUB_MAX_RETRIES' in os.environ:
            self.max_retries = int(os.environ['GITHUB_MAX_RETRIES'])

        if 'GITHUB_RATE_LIMIT_DELAY' in os.environ:
            self.rate_limit_delay = float(os.environ['GITHUB_RATE_LIMIT_DELAY'])

        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

    def __repr__(self):
        """String representation (hide token)"""
        token_preview = self.token[:8] + '...' if self.token else 'NOT_SET'
        return f"GitHubConfig(token={token_preview}, base_url={self.base_url})"


# ============================================================================
# Default Configuration Instances
# ============================================================================

def get_default_config() -> ParserConfig:
    """Get default parser configuration"""
    return ParserConfig()


def get_database_config() -> DatabaseConfig:
    """Get database configuration from environment"""
    return DatabaseConfig()


def get_github_config() -> GitHubConfig:
    """Get GitHub configuration from environment"""
    return GitHubConfig()
