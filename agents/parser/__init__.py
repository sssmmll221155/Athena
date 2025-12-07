"""
Athena Code Parser - Agent 2
Python AST parser for extracting code structure and metrics.
"""

__version__ = '1.0.0'

from agents.parser.models import (
    ParsedFile,
    CodeFunction,
    CodeImport,
    CodeClass
)

from agents.parser.python_parser import (
    PythonASTParser,
    ParsedPythonFile,
    FunctionInfo,
    ClassInfo,
    ImportInfo
)

from agents.parser.fetcher import (
    FileContentFetcher,
    FileContent,
    FetchResult
)

from agents.parser.writer import (
    DatabaseWriter,
    WriteResult,
    BatchWriteStats
)

from agents.parser.pipeline import (
    ParsingPipeline,
    PipelineStats,
    run_parser_pipeline
)

from agents.parser.config import (
    ParserConfig,
    DatabaseConfig,
    GitHubConfig
)

__all__ = [
    # Models
    'ParsedFile',
    'CodeFunction',
    'CodeImport',
    'CodeClass',

    # Parser
    'PythonASTParser',
    'ParsedPythonFile',
    'FunctionInfo',
    'ClassInfo',
    'ImportInfo',

    # Fetcher
    'FileContentFetcher',
    'FileContent',
    'FetchResult',

    # Writer
    'DatabaseWriter',
    'WriteResult',
    'BatchWriteStats',

    # Pipeline
    'ParsingPipeline',
    'PipelineStats',
    'run_parser_pipeline',

    # Config
    'ParserConfig',
    'DatabaseConfig',
    'GitHubConfig',
]
