"""
Python AST Parser
Extracts functions, classes, imports, and metrics from Python source code.
"""

import ast
import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    logging.warning("radon not available, complexity metrics will be limited")

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Parsed Results
# ============================================================================

@dataclass
class ImportInfo:
    """Information about an import statement"""
    import_type: str  # 'import', 'from_import', 'relative_import'
    module_name: str
    imported_names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    line_number: int = 0
    is_standard_library: bool = False
    is_third_party: bool = False
    is_local: bool = False
    is_relative: bool = False


@dataclass
class FunctionInfo:
    """Information about a function or method"""
    name: str
    qualified_name: str
    signature: str
    signature_hash: str
    function_type: str  # 'function', 'method', 'async_function', 'lambda'
    parent_class: Optional[str] = None
    is_async: bool = False
    is_private: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    start_line: int = 0
    end_line: int = 0
    lines_of_code: int = 0
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    parameter_count: int = 0
    return_count: int = 0
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    docstring_length: int = 0
    calls_functions: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class definition"""
    name: str
    qualified_name: str
    parent_classes: List[str] = field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False
    decorators: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    lines_of_code: int = 0
    method_count: int = 0
    attribute_count: int = 0
    docstring: Optional[str] = None
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)


@dataclass
class ParsedPythonFile:
    """Complete parsing result for a Python file"""
    file_path: str
    language: str = 'python'
    parser_version: str = '1.0.0'
    parse_status: str = 'success'
    parse_error: Optional[str] = None

    # File content
    content_hash: str = ''
    file_size_bytes: int = 0
    encoding: str = 'utf-8'

    # Line counts
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0

    # Extracted elements
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)

    # Aggregate metrics
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0
    average_complexity: float = 0.0
    max_complexity: float = 0.0

    # Metadata
    ast_metadata: Dict[str, Any] = field(default_factory=dict)
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    parse_duration_ms: int = 0


# ============================================================================
# AST Visitors
# ============================================================================

class FunctionVisitor(ast.NodeVisitor):
    """Visitor to extract function/method information"""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.functions: List[FunctionInfo] = []
        self.current_class: Optional[str] = None
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        """Track current class context"""
        self.class_stack.append(node.name)
        self.current_class = '.'.join(self.class_stack)
        self.generic_visit(node)
        self.class_stack.pop()
        self.current_class = '.'.join(self.class_stack) if self.class_stack else None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function information"""
        func_info = self._extract_function_info(node, is_async=False)
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Extract async function information"""
        func_info = self._extract_function_info(node, is_async=True)
        self.functions.append(func_info)
        self.generic_visit(node)

    def _extract_function_info(self, node, is_async: bool = False) -> FunctionInfo:
        """Extract detailed function information from AST node"""
        name = node.name
        is_private = name.startswith('_') and not name.startswith('__')

        # Determine function type
        if self.current_class:
            function_type = 'async_method' if is_async else 'method'
            qualified_name = f"{self.current_class}.{name}"
        else:
            function_type = 'async_function' if is_async else 'function'
            qualified_name = name

        # Extract decorators
        decorators = []
        is_static = False
        is_classmethod = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                dec_name = decorator.id
                decorators.append(dec_name)
                if dec_name == 'staticmethod':
                    is_static = True
                elif dec_name == 'classmethod':
                    is_classmethod = True
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorators.append(decorator.func.id)

        # Extract parameters
        parameters = []
        args = node.args
        for arg in args.args:
            param_info = {'name': arg.arg}
            if arg.annotation:
                param_info['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else None
            parameters.append(param_info)

        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else None

        # Generate signature
        param_str = ', '.join([p['name'] for p in parameters])
        signature = f"{name}({param_str})"
        if return_type:
            signature += f" -> {return_type}"

        signature_hash = hashlib.sha256(signature.encode('utf-8')).hexdigest()

        # Extract docstring
        docstring = ast.get_docstring(node)
        docstring_length = len(docstring) if docstring else 0

        # Count lines
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        lines_of_code = end_line - start_line + 1

        # Count return statements
        return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))

        # Extract function calls
        calls_functions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls_functions.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls_functions.append(child.func.attr)

        # Calculate cyclomatic complexity (basic)
        complexity = self._calculate_complexity(node)

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            signature=signature,
            signature_hash=signature_hash,
            function_type=function_type,
            parent_class=self.current_class,
            is_async=is_async,
            is_private=is_private,
            is_static=is_static,
            is_classmethod=is_classmethod,
            start_line=start_line,
            end_line=end_line,
            lines_of_code=lines_of_code,
            cyclomatic_complexity=complexity,
            parameter_count=len(parameters),
            return_count=return_count,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            docstring=docstring,
            docstring_length=docstring_length,
            calls_functions=calls_functions
        )

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity (basic implementation)"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points add complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, (ast.And, ast.Or)):
                complexity += len(child.values) - 1

        return complexity


class ClassVisitor(ast.NodeVisitor):
    """Visitor to extract class information"""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.classes: List[ClassInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class information"""
        name = node.name

        # Extract parent classes
        parent_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                parent_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                parent_classes.append(base.attr)

        # Extract decorators
        decorators = []
        is_dataclass = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                dec_name = decorator.id
                decorators.append(dec_name)
                if dec_name == 'dataclass':
                    is_dataclass = True
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                dec_name = decorator.func.id
                decorators.append(dec_name)
                if dec_name == 'dataclass':
                    is_dataclass = True

        # Check if abstract
        is_abstract = 'ABC' in parent_classes or 'abstractmethod' in decorators

        # Extract methods and attributes
        methods = []
        attributes = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attributes.append(item.target.id)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Count lines
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        lines_of_code = end_line - start_line + 1

        class_info = ClassInfo(
            name=name,
            qualified_name=name,  # Can be enhanced with module name
            parent_classes=parent_classes,
            is_abstract=is_abstract,
            is_dataclass=is_dataclass,
            decorators=decorators,
            start_line=start_line,
            end_line=end_line,
            lines_of_code=lines_of_code,
            method_count=len(methods),
            attribute_count=len(attributes),
            docstring=docstring,
            methods=methods,
            attributes=attributes
        )

        self.classes.append(class_info)
        self.generic_visit(node)


class ImportVisitor(ast.NodeVisitor):
    """Visitor to extract import statements"""

    def __init__(self):
        self.imports: List[ImportInfo] = []
        self.stdlib_modules = self._get_stdlib_modules()

    def visit_Import(self, node: ast.Import):
        """Extract regular import statements"""
        for alias in node.names:
            import_info = ImportInfo(
                import_type='import',
                module_name=alias.name,
                alias=alias.asname,
                line_number=node.lineno
            )
            self._classify_import(import_info)
            self.imports.append(import_info)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Extract from...import statements"""
        module_name = node.module or ''
        is_relative = node.level > 0

        if is_relative:
            module_name = '.' * node.level + module_name

        imported_names = [alias.name for alias in node.names]

        import_info = ImportInfo(
            import_type='from_import' if not is_relative else 'relative_import',
            module_name=module_name,
            imported_names=imported_names,
            line_number=node.lineno,
            is_relative=is_relative
        )
        self._classify_import(import_info)
        self.imports.append(import_info)

    def _classify_import(self, import_info: ImportInfo):
        """Classify import as standard library, third-party, or local"""
        module_root = import_info.module_name.split('.')[0]

        if import_info.is_relative:
            import_info.is_local = True
        elif module_root in self.stdlib_modules:
            import_info.is_standard_library = True
        else:
            # Heuristic: assume third-party if not stdlib and not relative
            import_info.is_third_party = True

    def _get_stdlib_modules(self) -> Set[str]:
        """Get set of standard library module names"""
        # Core standard library modules (Python 3.x)
        return {
            'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'copy',
            'csv', 'datetime', 'decimal', 'email', 'functools', 'glob', 'hashlib',
            'io', 'itertools', 'json', 'logging', 'math', 'os', 're', 'shutil',
            'socket', 'sqlite3', 'string', 'subprocess', 'sys', 'tempfile', 'time',
            'typing', 'unittest', 'urllib', 'uuid', 'warnings', 'xml', 'zipfile',
            'pathlib', 'dataclasses', 'enum', 'contextlib', 'pickle', 'random',
            'threading', 'multiprocessing', 'queue', 'concurrent', 'gzip', 'tarfile'
        }


# ============================================================================
# Main Parser Class
# ============================================================================

class PythonASTParser:
    """
    Python AST parser for extracting code structure and metrics.
    """

    def __init__(self):
        self.parser_version = '1.0.0'

    def parse_file(self, file_path: str, content: str) -> ParsedPythonFile:
        """
        Parse Python source code and extract all information.

        Args:
            file_path: Path to the file being parsed
            content: Python source code content

        Returns:
            ParsedPythonFile with all extracted information
        """
        start_time = datetime.utcnow()

        result = ParsedPythonFile(
            file_path=file_path,
            file_size_bytes=len(content.encode('utf-8')),
            content_hash=hashlib.sha256(content.encode('utf-8')).hexdigest()
        )

        try:
            # Parse AST
            tree = ast.parse(content)

            # Count lines
            source_lines = content.split('\n')
            result.total_lines = len(source_lines)
            result.blank_lines = sum(1 for line in source_lines if not line.strip())
            result.comment_lines = sum(1 for line in source_lines if line.strip().startswith('#'))
            result.code_lines = result.total_lines - result.blank_lines - result.comment_lines

            # Extract functions
            function_visitor = FunctionVisitor(source_lines)
            function_visitor.visit(tree)
            result.functions = function_visitor.functions
            result.total_functions = len(result.functions)

            # Extract classes
            class_visitor = ClassVisitor(source_lines)
            class_visitor.visit(tree)
            result.classes = class_visitor.classes
            result.total_classes = len(result.classes)

            # Extract imports
            import_visitor = ImportVisitor()
            import_visitor.visit(tree)
            result.imports = import_visitor.imports
            result.total_imports = len(result.imports)

            # Calculate complexity metrics
            if result.functions:
                complexities = [f.cyclomatic_complexity for f in result.functions]
                result.average_complexity = sum(complexities) / len(complexities)
                result.max_complexity = max(complexities)

            # Use radon for more accurate complexity if available
            if RADON_AVAILABLE:
                try:
                    complexity_results = cc_visit(content)
                    if complexity_results:
                        # Update function complexities with radon values
                        for i, func in enumerate(result.functions):
                            for cc_result in complexity_results:
                                if cc_result.name == func.name:
                                    func.cyclomatic_complexity = cc_result.complexity

                        # Recalculate aggregate metrics
                        complexities = [f.cyclomatic_complexity for f in result.functions]
                        if complexities:
                            result.average_complexity = sum(complexities) / len(complexities)
                            result.max_complexity = max(complexities)
                except Exception as e:
                    logger.debug(f"Radon complexity calculation failed: {e}")

            result.parse_status = 'success'

        except SyntaxError as e:
            result.parse_status = 'failed'
            result.parse_error = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Failed to parse {file_path}: {result.parse_error}")

        except Exception as e:
            result.parse_status = 'failed'
            result.parse_error = str(e)
            logger.error(f"Error parsing {file_path}: {e}")

        # Calculate parse duration
        end_time = datetime.utcnow()
        result.parse_duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return result

    def is_valid_python(self, content: str) -> bool:
        """Check if content is valid Python code"""
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False
