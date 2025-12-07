-- ============================================================================
-- Agent 2 Parser Schema - Code Parsing and Analysis
-- Extends base schema with code-level parsing tables
-- ============================================================================

-- ============================================================================
-- Parsed Files Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS parsed_files (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,

    -- Parsing metadata
    language VARCHAR(100) NOT NULL,
    parser_version VARCHAR(50) DEFAULT '1.0.0',
    parse_status VARCHAR(50) DEFAULT 'pending',  -- pending, success, failed, skipped
    parse_error TEXT,

    -- File content info
    content_hash VARCHAR(64),  -- SHA-256 of file content
    file_size_bytes INTEGER,
    encoding VARCHAR(50) DEFAULT 'utf-8',

    -- Code metrics
    total_lines INTEGER DEFAULT 0,
    code_lines INTEGER DEFAULT 0,
    comment_lines INTEGER DEFAULT 0,
    blank_lines INTEGER DEFAULT 0,
    total_functions INTEGER DEFAULT 0,
    total_classes INTEGER DEFAULT 0,
    total_imports INTEGER DEFAULT 0,
    average_complexity FLOAT,
    max_complexity FLOAT,

    -- Parsing timestamps
    parsed_at TIMESTAMP DEFAULT NOW(),
    parse_duration_ms INTEGER,  -- Time taken to parse in milliseconds

    -- Additional metadata
    ast_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_parsed_file UNIQUE(file_id, commit_sha),
    CONSTRAINT check_file_size_positive CHECK (file_size_bytes >= 0)
);

-- Indexes for parsed_files
CREATE INDEX idx_parsed_files_file ON parsed_files(file_id);
CREATE INDEX idx_parsed_files_repo ON parsed_files(repository_id);
CREATE INDEX idx_parsed_files_commit ON parsed_files(commit_sha);
CREATE INDEX idx_parsed_files_status ON parsed_files(parse_status);
CREATE INDEX idx_parsed_files_language ON parsed_files(language);
CREATE INDEX idx_parsed_files_hash ON parsed_files(content_hash);
CREATE INDEX idx_parsed_files_repo_commit ON parsed_files(repository_id, commit_sha);
CREATE INDEX idx_parsed_files_parsed_at ON parsed_files(parsed_at DESC);

-- ============================================================================
-- Code Functions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS code_functions (
    id SERIAL PRIMARY KEY,
    parsed_file_id INTEGER NOT NULL REFERENCES parsed_files(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,

    -- Function identification
    name VARCHAR(255) NOT NULL,
    qualified_name TEXT,  -- Full qualified name (e.g., ClassName.method_name)
    signature TEXT,  -- Function signature
    signature_hash VARCHAR(64),  -- Hash of signature for deduplication

    -- Function metadata
    function_type VARCHAR(50),  -- function, method, async_function, lambda, etc.
    parent_class VARCHAR(255),  -- Parent class name if it's a method
    is_async BOOLEAN DEFAULT FALSE,
    is_private BOOLEAN DEFAULT FALSE,
    is_static BOOLEAN DEFAULT FALSE,
    is_classmethod BOOLEAN DEFAULT FALSE,

    -- Code metrics
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    lines_of_code INTEGER DEFAULT 0,
    cyclomatic_complexity INTEGER,
    cognitive_complexity INTEGER,
    parameter_count INTEGER DEFAULT 0,
    return_count INTEGER DEFAULT 0,

    -- Function details
    parameters JSONB DEFAULT '[]'::jsonb,  -- List of parameter names and types
    return_type VARCHAR(255),
    decorators JSONB DEFAULT '[]'::jsonb,  -- List of decorator names
    docstring TEXT,
    docstring_length INTEGER,

    -- Dependencies
    calls_functions JSONB DEFAULT '[]'::jsonb,  -- Functions called within this function
    called_by_count INTEGER DEFAULT 0,  -- How many functions call this one

    -- Additional metadata
    function_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT check_line_numbers CHECK (start_line <= end_line),
    CONSTRAINT check_complexity_positive CHECK (cyclomatic_complexity >= 1)
);

-- Indexes for code_functions
CREATE INDEX idx_functions_parsed_file ON code_functions(parsed_file_id);
CREATE INDEX idx_functions_file ON code_functions(file_id);
CREATE INDEX idx_functions_repo ON code_functions(repository_id);
CREATE INDEX idx_functions_name ON code_functions(name);
CREATE INDEX idx_functions_file_name ON code_functions(file_id, name);
CREATE INDEX idx_functions_complexity ON code_functions(cyclomatic_complexity) WHERE cyclomatic_complexity IS NOT NULL;
CREATE INDEX idx_functions_signature_hash ON code_functions(signature_hash);
CREATE INDEX idx_functions_repo_commit ON code_functions(repository_id, commit_sha);
CREATE INDEX idx_functions_parent_class ON code_functions(parent_class) WHERE parent_class IS NOT NULL;

-- ============================================================================
-- Code Imports Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS code_imports (
    id SERIAL PRIMARY KEY,
    parsed_file_id INTEGER NOT NULL REFERENCES parsed_files(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,

    -- Import details
    import_type VARCHAR(50) NOT NULL,  -- import, from_import, relative_import
    module_name VARCHAR(500) NOT NULL,  -- The imported module
    imported_names JSONB DEFAULT '[]'::jsonb,  -- List of specific names imported
    alias VARCHAR(255),  -- Import alias (as ...)

    -- Import classification
    is_standard_library BOOLEAN DEFAULT FALSE,
    is_third_party BOOLEAN DEFAULT FALSE,
    is_local BOOLEAN DEFAULT FALSE,
    is_relative BOOLEAN DEFAULT FALSE,

    -- Location
    line_number INTEGER NOT NULL,

    -- Additional metadata
    import_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for code_imports
CREATE INDEX idx_imports_parsed_file ON code_imports(parsed_file_id);
CREATE INDEX idx_imports_file ON code_imports(file_id);
CREATE INDEX idx_imports_repo ON code_imports(repository_id);
CREATE INDEX idx_imports_module ON code_imports(module_name);
CREATE INDEX idx_imports_file_module ON code_imports(file_id, module_name);
CREATE INDEX idx_imports_third_party ON code_imports(is_third_party, module_name);
CREATE INDEX idx_imports_repo_commit ON code_imports(repository_id, commit_sha);

-- ============================================================================
-- Code Classes Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS code_classes (
    id SERIAL PRIMARY KEY,
    parsed_file_id INTEGER NOT NULL REFERENCES parsed_files(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,

    -- Class identification
    name VARCHAR(255) NOT NULL,
    qualified_name TEXT,  -- Full qualified name including parent classes

    -- Class metadata
    parent_classes JSONB DEFAULT '[]'::jsonb,  -- List of base classes
    is_abstract BOOLEAN DEFAULT FALSE,
    is_dataclass BOOLEAN DEFAULT FALSE,
    decorators JSONB DEFAULT '[]'::jsonb,

    -- Code metrics
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    lines_of_code INTEGER DEFAULT 0,
    method_count INTEGER DEFAULT 0,
    attribute_count INTEGER DEFAULT 0,

    -- Class details
    docstring TEXT,
    methods JSONB DEFAULT '[]'::jsonb,  -- List of method names
    attributes JSONB DEFAULT '[]'::jsonb,  -- List of class attributes

    -- Additional metadata
    class_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT check_class_line_numbers CHECK (start_line <= end_line)
);

-- Indexes for code_classes
CREATE INDEX idx_classes_parsed_file ON code_classes(parsed_file_id);
CREATE INDEX idx_classes_file ON code_classes(file_id);
CREATE INDEX idx_classes_repo ON code_classes(repository_id);
CREATE INDEX idx_classes_name ON code_classes(name);
CREATE INDEX idx_classes_file_name ON code_classes(file_id, name);
CREATE INDEX idx_classes_repo_commit ON code_classes(repository_id, commit_sha);

-- ============================================================================
-- Analytics Views
-- ============================================================================

-- Code Complexity View
CREATE OR REPLACE VIEW code_complexity_view AS
SELECT
    r.full_name as repository,
    f.path as file_path,
    pf.language,
    pf.total_lines,
    pf.code_lines,
    pf.total_functions,
    pf.total_classes,
    pf.average_complexity,
    pf.max_complexity,
    cf.name as function_name,
    cf.cyclomatic_complexity,
    cf.lines_of_code as function_loc
FROM parsed_files pf
JOIN files f ON f.id = pf.file_id
JOIN repositories r ON r.id = pf.repository_id
LEFT JOIN code_functions cf ON cf.parsed_file_id = pf.id
WHERE pf.parse_status = 'success';

-- Import Dependencies View
CREATE OR REPLACE VIEW import_dependencies_view AS
SELECT
    r.full_name as repository,
    f.path as file_path,
    ci.module_name,
    ci.import_type,
    ci.is_third_party,
    ci.is_standard_library,
    COUNT(*) OVER (PARTITION BY ci.module_name) as usage_count
FROM code_imports ci
JOIN files f ON f.id = ci.file_id
JOIN repositories r ON r.id = ci.repository_id
ORDER BY usage_count DESC;

-- Most Complex Functions View
CREATE OR REPLACE VIEW most_complex_functions AS
SELECT
    r.full_name as repository,
    f.path as file_path,
    cf.name as function_name,
    cf.qualified_name,
    cf.cyclomatic_complexity,
    cf.lines_of_code,
    cf.parameter_count,
    cf.parent_class
FROM code_functions cf
JOIN files f ON f.id = cf.file_id
JOIN repositories r ON r.id = cf.repository_id
WHERE cf.cyclomatic_complexity IS NOT NULL
ORDER BY cf.cyclomatic_complexity DESC
LIMIT 100;

-- File Statistics View
CREATE OR REPLACE VIEW file_statistics AS
SELECT
    r.full_name as repository,
    COUNT(DISTINCT pf.id) as total_parsed_files,
    SUM(pf.total_lines) as total_lines,
    SUM(pf.code_lines) as total_code_lines,
    SUM(pf.total_functions) as total_functions,
    SUM(pf.total_classes) as total_classes,
    SUM(pf.total_imports) as total_imports,
    AVG(pf.average_complexity) as avg_file_complexity,
    MAX(pf.max_complexity) as max_function_complexity
FROM parsed_files pf
JOIN repositories r ON r.id = pf.repository_id
WHERE pf.parse_status = 'success'
GROUP BY r.id, r.full_name;

-- ============================================================================
-- Triggers
-- ============================================================================

-- Update timestamp trigger for parsed_files
CREATE TRIGGER update_parsed_files_updated_at BEFORE UPDATE ON parsed_files
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update timestamp trigger for code_functions
CREATE TRIGGER update_code_functions_updated_at BEFORE UPDATE ON code_functions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update timestamp trigger for code_classes
CREATE TRIGGER update_code_classes_updated_at BEFORE UPDATE ON code_classes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update timestamp trigger for code_imports
CREATE TRIGGER update_code_imports_updated_at BEFORE UPDATE ON code_imports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function to clean up old parsing results
CREATE OR REPLACE FUNCTION cleanup_old_parsed_files(days_old INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM parsed_files
    WHERE parse_status = 'failed'
    AND parsed_at < NOW() - INTERVAL '1 day' * days_old;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to reparse files with errors
CREATE OR REPLACE FUNCTION get_files_to_reparse(limit_count INTEGER DEFAULT 100)
RETURNS TABLE (
    file_id INTEGER,
    repository_id INTEGER,
    commit_sha VARCHAR(40),
    parse_error TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pf.file_id,
        pf.repository_id,
        pf.commit_sha,
        pf.parse_error
    FROM parsed_files pf
    WHERE pf.parse_status = 'failed'
    ORDER BY pf.parsed_at DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE parsed_files IS 'Tracks parsing status and metrics for each file at each commit';
COMMENT ON TABLE code_functions IS 'Individual functions/methods extracted from code with complexity metrics';
COMMENT ON TABLE code_imports IS 'Import statements and dependencies between modules';
COMMENT ON TABLE code_classes IS 'Class definitions with inheritance and composition information';

COMMENT ON VIEW code_complexity_view IS 'Comprehensive view of code complexity metrics across repositories';
COMMENT ON VIEW import_dependencies_view IS 'Analysis of import dependencies and third-party usage';
COMMENT ON VIEW most_complex_functions IS 'Top 100 most complex functions by cyclomatic complexity';
COMMENT ON VIEW file_statistics IS 'Aggregate statistics per repository';

-- ============================================================================
-- Sample Queries
-- ============================================================================

/*
-- Find files with highest complexity
SELECT
    repository,
    file_path,
    average_complexity,
    max_complexity,
    total_functions
FROM code_complexity_view
WHERE average_complexity > 10
ORDER BY max_complexity DESC
LIMIT 20;

-- Find most imported third-party modules
SELECT
    module_name,
    COUNT(*) as import_count,
    COUNT(DISTINCT repository) as repo_count
FROM import_dependencies_view
WHERE is_third_party = TRUE
GROUP BY module_name
ORDER BY import_count DESC
LIMIT 20;

-- Find functions that need refactoring (high complexity)
SELECT
    repository,
    file_path,
    function_name,
    cyclomatic_complexity,
    lines_of_code,
    parameter_count
FROM most_complex_functions
WHERE cyclomatic_complexity > 15
ORDER BY cyclomatic_complexity DESC;

-- Repository parsing status
SELECT
    r.full_name,
    COUNT(DISTINCT pf.id) FILTER (WHERE pf.parse_status = 'success') as parsed,
    COUNT(DISTINCT pf.id) FILTER (WHERE pf.parse_status = 'failed') as failed,
    COUNT(DISTINCT pf.id) FILTER (WHERE pf.parse_status = 'pending') as pending
FROM repositories r
LEFT JOIN parsed_files pf ON pf.repository_id = r.id
GROUP BY r.id, r.full_name
ORDER BY parsed DESC;
*/
