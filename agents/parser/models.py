"""
Agent 2 Parser - SQLAlchemy ORM Models
Extends base models for code parsing and analysis.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float,
    DateTime, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import hashlib

# Import base from crawler models
from agents.crawler.models import Base, File, Repository, Commit


# ============================================================================
# Parser Models
# ============================================================================

class ParsedFile(Base):
    """
    Extends File table with parsing metadata.
    Tracks which files have been parsed at which commits.
    """
    __tablename__ = 'parsed_files'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    commit_sha = Column(String(40), nullable=False, index=True)

    # Parsing metadata
    language = Column(String(100), nullable=False, index=True)
    parser_version = Column(String(50), default='1.0.0')
    parse_status = Column(String(50), default='pending', index=True)  # pending, success, failed, skipped
    parse_error = Column(Text)

    # File content info
    content_hash = Column(String(64), index=True)  # SHA-256 of file content
    file_size_bytes = Column(Integer)
    encoding = Column(String(50), default='utf-8')

    # Code metrics
    total_lines = Column(Integer, default=0)
    code_lines = Column(Integer, default=0)
    comment_lines = Column(Integer, default=0)
    blank_lines = Column(Integer, default=0)
    total_functions = Column(Integer, default=0)
    total_classes = Column(Integer, default=0)
    total_imports = Column(Integer, default=0)
    average_complexity = Column(Float)
    max_complexity = Column(Float)

    # Parsing timestamps
    parsed_at = Column(DateTime, default=datetime.utcnow, index=True)
    parse_duration_ms = Column(Integer)  # Time taken to parse in milliseconds

    # Additional metadata
    ast_metadata = Column(JSONB, default=dict)  # Store AST-specific info
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    file = relationship("File", backref="parsed_versions")
    repository = relationship("Repository")
    functions = relationship("CodeFunction", back_populates="parsed_file", cascade="all, delete-orphan")
    imports = relationship("CodeImport", back_populates="parsed_file", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_parsed_files_file_commit', 'file_id', 'commit_sha', unique=True),
        Index('idx_parsed_files_repo_commit', 'repository_id', 'commit_sha'),
        Index('idx_parsed_files_hash', 'content_hash'),
        CheckConstraint('file_size_bytes >= 0', name='check_file_size_positive'),
    )

    def __repr__(self):
        return f"<ParsedFile(id={self.id}, file_id={self.file_id}, commit='{self.commit_sha[:7]}')>"


class CodeFunction(Base):
    """
    Individual function/method extracted from code.
    Tracks function-level metrics and complexity.
    """
    __tablename__ = 'code_functions'

    id = Column(Integer, primary_key=True)
    parsed_file_id = Column(Integer, ForeignKey('parsed_files.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False, index=True)
    commit_sha = Column(String(40), nullable=False, index=True)

    # Function identification
    name = Column(String(255), nullable=False, index=True)
    qualified_name = Column(Text)  # Full qualified name (e.g., ClassName.method_name)
    signature = Column(Text)  # Function signature
    signature_hash = Column(String(64), index=True)  # Hash of signature for deduplication

    # Function metadata
    function_type = Column(String(50))  # function, method, async_function, lambda, etc.
    parent_class = Column(String(255))  # Parent class name if it's a method
    is_async = Column(Boolean, default=False)
    is_private = Column(Boolean, default=False)
    is_static = Column(Boolean, default=False)
    is_classmethod = Column(Boolean, default=False)

    # Code metrics
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    lines_of_code = Column(Integer, default=0)
    cyclomatic_complexity = Column(Integer, index=True)
    cognitive_complexity = Column(Integer)
    parameter_count = Column(Integer, default=0)
    return_count = Column(Integer, default=0)

    # Function details
    parameters = Column(JSONB, default=list)  # List of parameter names and types
    return_type = Column(String(255))
    decorators = Column(JSONB, default=list)  # List of decorator names
    docstring = Column(Text)
    docstring_length = Column(Integer)

    # Dependencies
    calls_functions = Column(JSONB, default=list)  # Functions called within this function
    called_by_count = Column(Integer, default=0)  # How many functions call this one

    # Additional metadata
    function_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parsed_file = relationship("ParsedFile", back_populates="functions")
    repository = relationship("Repository")
    file = relationship("File")

    __table_args__ = (
        Index('idx_functions_file_name', 'file_id', 'name'),
        Index('idx_functions_complexity', 'cyclomatic_complexity', postgresql_using='btree'),
        Index('idx_functions_signature_hash', 'signature_hash'),
        Index('idx_functions_repo_commit', 'repository_id', 'commit_sha'),
        CheckConstraint('start_line <= end_line', name='check_line_numbers'),
        CheckConstraint('cyclomatic_complexity >= 1', name='check_complexity_positive'),
    )

    @staticmethod
    def generate_signature_hash(signature: str) -> str:
        """Generate SHA-256 hash of function signature"""
        return hashlib.sha256(signature.encode('utf-8')).hexdigest()

    def __repr__(self):
        return f"<CodeFunction(id={self.id}, name='{self.name}', complexity={self.cyclomatic_complexity})>"


class CodeImport(Base):
    """
    Import statements extracted from code.
    Tracks dependencies between files and modules.
    """
    __tablename__ = 'code_imports'

    id = Column(Integer, primary_key=True)
    parsed_file_id = Column(Integer, ForeignKey('parsed_files.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False, index=True)
    commit_sha = Column(String(40), nullable=False, index=True)

    # Import details
    import_type = Column(String(50), nullable=False)  # import, from_import, relative_import
    module_name = Column(String(500), nullable=False, index=True)  # The imported module
    imported_names = Column(JSONB, default=list)  # List of specific names imported
    alias = Column(String(255))  # Import alias (as ...)

    # Import classification
    is_standard_library = Column(Boolean, default=False)
    is_third_party = Column(Boolean, default=False, index=True)
    is_local = Column(Boolean, default=False)
    is_relative = Column(Boolean, default=False)

    # Location
    line_number = Column(Integer, nullable=False)

    # Additional metadata
    import_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parsed_file = relationship("ParsedFile", back_populates="imports")
    repository = relationship("Repository")
    file = relationship("File")

    __table_args__ = (
        Index('idx_imports_module', 'module_name'),
        Index('idx_imports_file', 'file_id', 'module_name'),
        Index('idx_imports_third_party', 'is_third_party', 'module_name'),
        Index('idx_imports_repo_commit', 'repository_id', 'commit_sha'),
    )

    def __repr__(self):
        return f"<CodeImport(id={self.id}, module='{self.module_name}', type='{self.import_type}')>"


class CodeClass(Base):
    """
    Class definitions extracted from code.
    Tracks class-level metrics and relationships.
    """
    __tablename__ = 'code_classes'

    id = Column(Integer, primary_key=True)
    parsed_file_id = Column(Integer, ForeignKey('parsed_files.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False, index=True)
    commit_sha = Column(String(40), nullable=False, index=True)

    # Class identification
    name = Column(String(255), nullable=False, index=True)
    qualified_name = Column(Text)  # Full qualified name including parent classes

    # Class metadata
    parent_classes = Column(JSONB, default=list)  # List of base classes
    is_abstract = Column(Boolean, default=False)
    is_dataclass = Column(Boolean, default=False)
    decorators = Column(JSONB, default=list)

    # Code metrics
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    lines_of_code = Column(Integer, default=0)
    method_count = Column(Integer, default=0)
    attribute_count = Column(Integer, default=0)

    # Class details
    docstring = Column(Text)
    methods = Column(JSONB, default=list)  # List of method names
    attributes = Column(JSONB, default=list)  # List of class attributes

    # Additional metadata
    class_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    repository = relationship("Repository")
    file = relationship("File")

    __table_args__ = (
        Index('idx_classes_file_name', 'file_id', 'name'),
        Index('idx_classes_repo_commit', 'repository_id', 'commit_sha'),
        CheckConstraint('start_line <= end_line', name='check_class_line_numbers'),
    )

    def __repr__(self):
        return f"<CodeClass(id={self.id}, name='{self.name}', methods={self.method_count})>"
