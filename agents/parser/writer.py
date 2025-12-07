"""
Database Writer
Writes parsed code data to PostgreSQL database.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert

from agents.parser.models import (
    ParsedFile, CodeFunction, CodeImport, CodeClass
)
from agents.parser.python_parser import ParsedPythonFile

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WriteResult:
    """Result of writing parsed data to database"""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchWriteStats:
    """Statistics from batch write operation"""
    parsed_files_inserted: int = 0
    functions_inserted: int = 0
    classes_inserted: int = 0
    imports_inserted: int = 0
    duration_ms: int = 0


# ============================================================================
# Database Writer
# ============================================================================

class DatabaseWriter:
    """
    Writes parsed code data to database.
    Handles batching, deduplication, and transactions.
    """

    def __init__(self, session: Session, batch_size: int = 100):
        """
        Initialize database writer.

        Args:
            session: SQLAlchemy database session
            batch_size: Number of rows to insert per batch
        """
        self.session = session
        self.batch_size = batch_size

    def check_already_parsed(
        self,
        file_id: int,
        commit_sha: str
    ) -> bool:
        """
        Check if file at commit has already been parsed.

        Args:
            file_id: Database file ID
            commit_sha: Commit SHA

        Returns:
            True if already parsed successfully
        """
        stmt = select(ParsedFile).where(
            and_(
                ParsedFile.file_id == file_id,
                ParsedFile.commit_sha == commit_sha,
                ParsedFile.parse_status == 'success'
            )
        )

        result = self.session.execute(stmt).scalar_one_or_none()
        return result is not None

    def write_parsed_file(
        self,
        parsed_file: ParsedPythonFile,
        file_id: int,
        repository_id: int,
        commit_sha: str
    ) -> Optional[ParsedFile]:
        """
        Write a single parsed file to database.

        Args:
            parsed_file: Parsed file data
            file_id: Database file ID
            repository_id: Database repository ID
            commit_sha: Commit SHA

        Returns:
            ParsedFile record if successful, None otherwise
        """
        try:
            # Check if already exists
            existing = self.session.execute(
                select(ParsedFile).where(
                    and_(
                        ParsedFile.file_id == file_id,
                        ParsedFile.commit_sha == commit_sha
                    )
                )
            ).scalar_one_or_none()

            if existing:
                # Update existing record
                existing.language = parsed_file.language
                existing.parser_version = parsed_file.parser_version
                existing.parse_status = parsed_file.parse_status
                existing.parse_error = parsed_file.parse_error
                existing.content_hash = parsed_file.content_hash
                existing.file_size_bytes = parsed_file.file_size_bytes
                existing.encoding = parsed_file.encoding
                existing.total_lines = parsed_file.total_lines
                existing.code_lines = parsed_file.code_lines
                existing.comment_lines = parsed_file.comment_lines
                existing.blank_lines = parsed_file.blank_lines
                existing.total_functions = parsed_file.total_functions
                existing.total_classes = parsed_file.total_classes
                existing.total_imports = parsed_file.total_imports
                existing.average_complexity = parsed_file.average_complexity
                existing.max_complexity = parsed_file.max_complexity
                existing.parsed_at = parsed_file.parsed_at
                existing.parse_duration_ms = parsed_file.parse_duration_ms
                existing.ast_metadata = parsed_file.ast_metadata
                existing.updated_at = datetime.utcnow()

                parsed_file_record = existing
                logger.debug(f"Updated existing ParsedFile: file_id={file_id}, commit={commit_sha[:7]}")
            else:
                # Create new record
                parsed_file_record = ParsedFile(
                    file_id=file_id,
                    repository_id=repository_id,
                    commit_sha=commit_sha,
                    language=parsed_file.language,
                    parser_version=parsed_file.parser_version,
                    parse_status=parsed_file.parse_status,
                    parse_error=parsed_file.parse_error,
                    content_hash=parsed_file.content_hash,
                    file_size_bytes=parsed_file.file_size_bytes,
                    encoding=parsed_file.encoding,
                    total_lines=parsed_file.total_lines,
                    code_lines=parsed_file.code_lines,
                    comment_lines=parsed_file.comment_lines,
                    blank_lines=parsed_file.blank_lines,
                    total_functions=parsed_file.total_functions,
                    total_classes=parsed_file.total_classes,
                    total_imports=parsed_file.total_imports,
                    average_complexity=parsed_file.average_complexity,
                    max_complexity=parsed_file.max_complexity,
                    parsed_at=parsed_file.parsed_at,
                    parse_duration_ms=parsed_file.parse_duration_ms,
                    ast_metadata=parsed_file.ast_metadata
                )

                self.session.add(parsed_file_record)
                logger.debug(f"Created new ParsedFile: file_id={file_id}, commit={commit_sha[:7]}")

            # Flush to get the ID
            self.session.flush()

            return parsed_file_record

        except Exception as e:
            logger.error(f"Error writing ParsedFile: {e}")
            raise

    def write_functions_batch(
        self,
        parsed_file_id: int,
        file_id: int,
        repository_id: int,
        commit_sha: str,
        functions: List,
    ) -> int:
        """
        Write function records in batch.

        Args:
            parsed_file_id: ParsedFile record ID
            file_id: File ID
            repository_id: Repository ID
            commit_sha: Commit SHA
            functions: List of FunctionInfo objects

        Returns:
            Number of functions inserted
        """
        if not functions:
            return 0

        function_dicts = []
        for func in functions:
            function_dict = {
                'parsed_file_id': parsed_file_id,
                'repository_id': repository_id,
                'file_id': file_id,
                'commit_sha': commit_sha,
                'name': func.name,
                'qualified_name': func.qualified_name,
                'signature': func.signature,
                'signature_hash': func.signature_hash,
                'function_type': func.function_type,
                'parent_class': func.parent_class,
                'is_async': func.is_async,
                'is_private': func.is_private,
                'is_static': func.is_static,
                'is_classmethod': func.is_classmethod,
                'start_line': func.start_line,
                'end_line': func.end_line,
                'lines_of_code': func.lines_of_code,
                'cyclomatic_complexity': func.cyclomatic_complexity,
                'cognitive_complexity': func.cognitive_complexity,
                'parameter_count': func.parameter_count,
                'return_count': func.return_count,
                'parameters': func.parameters,
                'return_type': func.return_type,
                'decorators': func.decorators,
                'docstring': func.docstring,
                'docstring_length': func.docstring_length,
                'calls_functions': func.calls_functions
            }
            function_dicts.append(function_dict)

        try:
            # Batch insert using PostgreSQL INSERT ... ON CONFLICT
            # This handles deduplication based on unique constraints
            for i in range(0, len(function_dicts), self.batch_size):
                batch = function_dicts[i:i + self.batch_size]
                for func_dict in batch:
                    func_record = CodeFunction(**func_dict)
                    self.session.add(func_record)

            self.session.flush()
            logger.debug(f"Inserted {len(function_dicts)} functions for parsed_file_id={parsed_file_id}")
            return len(function_dicts)

        except Exception as e:
            logger.error(f"Error writing functions batch: {e}")
            raise

    def write_classes_batch(
        self,
        parsed_file_id: int,
        file_id: int,
        repository_id: int,
        commit_sha: str,
        classes: List,
    ) -> int:
        """
        Write class records in batch.

        Args:
            parsed_file_id: ParsedFile record ID
            file_id: File ID
            repository_id: Repository ID
            commit_sha: Commit SHA
            classes: List of ClassInfo objects

        Returns:
            Number of classes inserted
        """
        if not classes:
            return 0

        class_dicts = []
        for cls in classes:
            class_dict = {
                'parsed_file_id': parsed_file_id,
                'repository_id': repository_id,
                'file_id': file_id,
                'commit_sha': commit_sha,
                'name': cls.name,
                'qualified_name': cls.qualified_name,
                'parent_classes': cls.parent_classes,
                'is_abstract': cls.is_abstract,
                'is_dataclass': cls.is_dataclass,
                'decorators': cls.decorators,
                'start_line': cls.start_line,
                'end_line': cls.end_line,
                'lines_of_code': cls.lines_of_code,
                'method_count': cls.method_count,
                'attribute_count': cls.attribute_count,
                'docstring': cls.docstring,
                'methods': cls.methods,
                'attributes': cls.attributes
            }
            class_dicts.append(class_dict)

        try:
            for i in range(0, len(class_dicts), self.batch_size):
                batch = class_dicts[i:i + self.batch_size]
                for class_dict in batch:
                    class_record = CodeClass(**class_dict)
                    self.session.add(class_record)

            self.session.flush()
            logger.debug(f"Inserted {len(class_dicts)} classes for parsed_file_id={parsed_file_id}")
            return len(class_dicts)

        except Exception as e:
            logger.error(f"Error writing classes batch: {e}")
            raise

    def write_imports_batch(
        self,
        parsed_file_id: int,
        file_id: int,
        repository_id: int,
        commit_sha: str,
        imports: List,
    ) -> int:
        """
        Write import records in batch.

        Args:
            parsed_file_id: ParsedFile record ID
            file_id: File ID
            repository_id: Repository ID
            commit_sha: Commit SHA
            imports: List of ImportInfo objects

        Returns:
            Number of imports inserted
        """
        if not imports:
            return 0

        import_dicts = []
        for imp in imports:
            import_dict = {
                'parsed_file_id': parsed_file_id,
                'repository_id': repository_id,
                'file_id': file_id,
                'commit_sha': commit_sha,
                'import_type': imp.import_type,
                'module_name': imp.module_name,
                'imported_names': imp.imported_names,
                'alias': imp.alias,
                'is_standard_library': imp.is_standard_library,
                'is_third_party': imp.is_third_party,
                'is_local': imp.is_local,
                'is_relative': imp.is_relative,
                'line_number': imp.line_number
            }
            import_dicts.append(import_dict)

        try:
            for i in range(0, len(import_dicts), self.batch_size):
                batch = import_dicts[i:i + self.batch_size]
                for import_dict in batch:
                    import_record = CodeImport(**import_dict)
                    self.session.add(import_record)

            self.session.flush()
            logger.debug(f"Inserted {len(import_dicts)} imports for parsed_file_id={parsed_file_id}")
            return len(import_dicts)

        except Exception as e:
            logger.error(f"Error writing imports batch: {e}")
            raise

    def write_complete_parsed_file(
        self,
        parsed_file: ParsedPythonFile,
        file_id: int,
        repository_id: int,
        commit_sha: str
    ) -> BatchWriteStats:
        """
        Write complete parsed file with all associated data.

        Args:
            parsed_file: Parsed file data
            file_id: Database file ID
            repository_id: Database repository ID
            commit_sha: Commit SHA

        Returns:
            BatchWriteStats with insert counts
        """
        start_time = datetime.utcnow()
        stats = BatchWriteStats()

        try:
            # Write ParsedFile record
            parsed_file_record = self.write_parsed_file(
                parsed_file=parsed_file,
                file_id=file_id,
                repository_id=repository_id,
                commit_sha=commit_sha
            )

            if not parsed_file_record:
                logger.error(f"Failed to create ParsedFile record for file_id={file_id}")
                return stats

            stats.parsed_files_inserted = 1

            # Only write details if parsing was successful
            if parsed_file.parse_status == 'success':
                # Delete existing related records to avoid duplicates
                self.session.query(CodeFunction).filter(
                    CodeFunction.parsed_file_id == parsed_file_record.id
                ).delete()
                self.session.query(CodeClass).filter(
                    CodeClass.parsed_file_id == parsed_file_record.id
                ).delete()
                self.session.query(CodeImport).filter(
                    CodeImport.parsed_file_id == parsed_file_record.id
                ).delete()

                # Write functions
                stats.functions_inserted = self.write_functions_batch(
                    parsed_file_id=parsed_file_record.id,
                    file_id=file_id,
                    repository_id=repository_id,
                    commit_sha=commit_sha,
                    functions=parsed_file.functions
                )

                # Write classes
                stats.classes_inserted = self.write_classes_batch(
                    parsed_file_id=parsed_file_record.id,
                    file_id=file_id,
                    repository_id=repository_id,
                    commit_sha=commit_sha,
                    classes=parsed_file.classes
                )

                # Write imports
                stats.imports_inserted = self.write_imports_batch(
                    parsed_file_id=parsed_file_record.id,
                    file_id=file_id,
                    repository_id=repository_id,
                    commit_sha=commit_sha,
                    imports=parsed_file.imports
                )

            # Commit transaction
            self.session.commit()

            end_time = datetime.utcnow()
            stats.duration_ms = int((end_time - start_time).total_seconds() * 1000)

            logger.info(
                f"Wrote ParsedFile file_id={file_id}: "
                f"{stats.functions_inserted} functions, "
                f"{stats.classes_inserted} classes, "
                f"{stats.imports_inserted} imports"
            )

            return stats

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error writing complete parsed file: {e}")
            raise

    def write_batch(
        self,
        parsed_files: List[tuple]  # (ParsedPythonFile, file_id, repository_id, commit_sha)
    ) -> WriteResult:
        """
        Write multiple parsed files in batch.

        Args:
            parsed_files: List of tuples (ParsedPythonFile, file_id, repository_id, commit_sha)

        Returns:
            WriteResult with statistics
        """
        result = WriteResult(total_files=len(parsed_files))

        for parsed_file, file_id, repository_id, commit_sha in parsed_files:
            try:
                # Check if already parsed
                if self.check_already_parsed(file_id, commit_sha):
                    logger.debug(f"Skipping already parsed file: file_id={file_id}, commit={commit_sha[:7]}")
                    result.skipped += 1
                    continue

                # Write complete parsed file
                stats = self.write_complete_parsed_file(
                    parsed_file=parsed_file,
                    file_id=file_id,
                    repository_id=repository_id,
                    commit_sha=commit_sha
                )

                result.successful += 1
                result.total_functions += stats.functions_inserted
                result.total_classes += stats.classes_inserted
                result.total_imports += stats.imports_inserted

            except Exception as e:
                result.failed += 1
                error_msg = f"file_id={file_id}: {str(e)}"
                result.errors.append(error_msg)
                logger.error(f"Failed to write parsed file: {error_msg}")
                # Continue with next file

        logger.info(
            f"Batch write complete: {result.successful} successful, "
            f"{result.skipped} skipped, {result.failed} failed"
        )

        return result
