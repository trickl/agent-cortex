"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

SQL Database Tool - Manages SQL database operations including queries, schema management, and data manipulation across different database systems.
"""

import sqlite3
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import re
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from contextlib import contextmanager
import hashlib
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SQLDatabaseTool:
    """
    Advanced SQL database query tool with security, optimization, and analysis features.
    Supports multiple database types and provides comprehensive database management.
    """
    
    def __init__(self, connection_string: str = None, database_type: str = None):
        """
        Initialize the SQL database tool.
        
        Args:
            connection_string: Database connection string
            database_type: Type of database (sqlite, postgresql, mysql, oracle, mssql)
        """
        self.connection_string = connection_string or os.getenv('SQL_CONNECTION_STRING', 'sqlite:///:memory:')
        self.database_type = database_type or os.getenv('SQL_DATABASE_TYPE', 'sqlite').lower()
        self.engine = None
        self.metadata = None
        self.schema_cache = {}
        self.query_history = []
        self.query_cache = {}
        self.connection_pool_size = int(os.getenv('SQL_POOL_SIZE', '5'))
        
        # SQL injection protection patterns
        self.dangerous_patterns = [
            r';\s*(drop|delete|truncate|alter|create|insert|update)\s+',
            r'union\s+select',
            r'exec\s*\(',
            r'sp_\w+',
            r'xp_\w+',
            r'--\s*$',
            r'/\*.*\*/',
        ]
        
        # Query performance thresholds (in seconds)
        self.performance_thresholds = {
            'fast': 1.0,
            'moderate': 5.0,
            'slow': 10.0
        }
        
        if self.connection_string:
            self.connect()
    
    def connect(self, connection_string: str = None) -> Dict[str, Any]:
        """
        Establish database connection.
        
        Args:
            connection_string: Database connection string
            
        Returns:
            Connection status and information
        """
        try:
            if connection_string:
                self.connection_string = connection_string
            
            if not self.connection_string:
                # Default to in-memory SQLite for testing
                self.connection_string = "sqlite:///:memory:"
                self.database_type = 'sqlite'
            
            # Create engine with connection pooling
            engine_kwargs = {
                'echo': False,
                'pool_pre_ping': True,
                'pool_recycle': int(os.getenv('SQL_POOL_RECYCLE', '3600')),
            }
            
            if self.database_type != 'sqlite':
                engine_kwargs.update({
                    'pool_size': self.connection_pool_size,
                    'max_overflow': int(os.getenv('SQL_MAX_OVERFLOW', '10')),
                    'pool_timeout': int(os.getenv('SQL_POOL_TIMEOUT', '30'))
                })
            
            self.engine = create_engine(self.connection_string, **engine_kwargs)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            # Initialize metadata
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            
            # Cache schema information
            self._cache_schema_info()
            
            return {
                'status': 'success',
                'message': f'Connected to {self.database_type} database',
                'database_type': self.database_type,
                'tables_count': len(self.metadata.tables),
                'connection_pool_size': self.connection_pool_size
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to connect to database: {str(e)}'
            }
    
    def _cache_schema_info(self):
        """Cache database schema information for performance."""
        try:
            inspector = inspect(self.engine)
            
            self.schema_cache = {
                'tables': {},
                'views': inspector.get_view_names() if hasattr(inspector, 'get_view_names') else [],
                'schemas': inspector.get_schema_names() if hasattr(inspector, 'get_schema_names') else []
            }
            
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                indexes = inspector.get_indexes(table_name)
                primary_key = inspector.get_pk_constraint(table_name)
                
                self.schema_cache['tables'][table_name] = {
                    'columns': columns,
                    'foreign_keys': foreign_keys,
                    'indexes': indexes,
                    'primary_key': primary_key,
                    'row_count': self._get_table_row_count(table_name)
                }
                
        except Exception as e:
            logging.warning(f"Failed to cache schema info: {str(e)}")
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
        except:
            return 0
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate SQL query for security and syntax.
        
        Args:
            query: SQL query to validate
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check for dangerous patterns
        query_lower = query.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern detected: {pattern}")
        
        # Check for common issues
        if 'select *' in query_lower:
            warnings.append("Using SELECT * can impact performance")
        
        if 'order by' not in query_lower and 'limit' in query_lower:
            warnings.append("LIMIT without ORDER BY may return inconsistent results")
        
        if len(query.split()) > 100:
            warnings.append("Query is very long and might be complex")
        
        # Basic syntax validation
        try:
            # Try to parse the query using SQLAlchemy
            parsed = text(query)
            syntax_valid = True
        except Exception as e:
            syntax_valid = False
            issues.append(f"Syntax error: {str(e)}")
        
        validation_result = {
            'is_valid': len(issues) == 0,
            'is_safe': len([i for i in issues if 'dangerous' in i]) == 0,
            'syntax_valid': syntax_valid,
            'issues': issues,
            'warnings': warnings,
            'query_type': self._detect_query_type(query)
        }
        
        return validation_result
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of SQL query."""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return 'SELECT'
        elif query_lower.startswith('insert'):
            return 'INSERT'
        elif query_lower.startswith('update'):
            return 'UPDATE'
        elif query_lower.startswith('delete'):
            return 'DELETE'
        elif query_lower.startswith('create'):
            return 'CREATE'
        elif query_lower.startswith('drop'):
            return 'DROP'
        elif query_lower.startswith('alter'):
            return 'ALTER'
        else:
            return 'OTHER'
    
    def execute_query(self, query: str, params: Dict[str, Any] = None, fetch_size: int = None) -> Dict[str, Any]:
        """
        Execute SQL query with security validation and performance monitoring.
        
        Args:
            query: SQL query to execute
            params: Query parameters for prepared statements
            fetch_size: Maximum number of rows to fetch
            
        Returns:
            Query results and metadata
        """
        if not self.engine:
            return {'status': 'error', 'message': 'No database connection'}
        
        # Validate query
        validation = self.validate_query(query)
        if not validation['is_safe']:
            return {
                'status': 'error',
                'message': 'Query failed security validation',
                'validation': validation
            }
        
        try:
            start_time = time.time()
            
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                execution_time = time.time() - start_time
                
                # Handle different query types
                if validation['query_type'] == 'SELECT':
                    if fetch_size:
                        rows = result.fetchmany(fetch_size)
                    else:
                        rows = result.fetchall()
                    
                    # Convert to DataFrame for easier handling
                    if rows:
                        df = pd.DataFrame(rows, columns=result.keys())
                        data = df.to_dict('records')
                        columns = list(df.columns)
                        row_count = len(df)
                    else:
                        data = []
                        columns = list(result.keys()) if result.keys() else []
                        row_count = 0
                    
                    query_result = {
                        'status': 'success',
                        'data': data,
                        'columns': columns,
                        'row_count': row_count,
                        'execution_time': execution_time,
                        'performance_category': self._categorize_performance(execution_time),
                        'query_type': validation['query_type'],
                        'has_more_data': fetch_size and len(rows) == fetch_size if rows else False
                    }
                    
                else:
                    # For non-SELECT queries
                    affected_rows = result.rowcount if hasattr(result, 'rowcount') else 0
                    
                    query_result = {
                        'status': 'success',
                        'affected_rows': affected_rows,
                        'execution_time': execution_time,
                        'performance_category': self._categorize_performance(execution_time),
                        'query_type': validation['query_type']
                    }
                
                # Add to query history
                self._add_to_history(query, params, query_result, validation)
                
                return query_result
                
        except Exception as e:
            error_result = {
                'status': 'error',
                'message': str(e),
                'query_type': validation['query_type'],
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
            
            self._add_to_history(query, params, error_result, validation)
            return error_result
    
    def _categorize_performance(self, execution_time: float) -> str:
        """Categorize query performance based on execution time."""
        if execution_time <= self.performance_thresholds['fast']:
            return 'fast'
        elif execution_time <= self.performance_thresholds['moderate']:
            return 'moderate'
        elif execution_time <= self.performance_thresholds['slow']:
            return 'slow'
        else:
            return 'very_slow'
    
    def _add_to_history(self, query: str, params: Dict, result: Dict, validation: Dict):
        """Add query to execution history."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'params': params,
            'result_status': result['status'],
            'execution_time': result.get('execution_time', 0),
            'query_type': validation['query_type'],
            'query_hash': hashlib.md5(query.encode()).hexdigest()
        }
        
        self.query_history.append(history_entry)
        
        # Keep only last 100 queries
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
    
    def get_schema_info(self, table_name: str = None) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            table_name: Specific table to get info for (optional)
            
        Returns:
            Schema information
        """
        if not self.engine:
            return {'status': 'error', 'message': 'No database connection'}
        
        try:
            if table_name:
                if table_name not in self.schema_cache['tables']:
                    return {'status': 'error', 'message': f'Table {table_name} not found'}
                
                table_info = self.schema_cache['tables'][table_name].copy()
                
                # Add sample data
                sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                sample_result = self.execute_query(sample_query)
                table_info['sample_data'] = sample_result.get('data', [])
                
                return {
                    'status': 'success',
                    'table_name': table_name,
                    'table_info': table_info
                }
            else:
                return {
                    'status': 'success',
                    'schema_info': self.schema_cache,
                    'total_tables': len(self.schema_cache['tables'])
                }
                
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get schema info: {str(e)}'}
    
    def close_connection(self) -> Dict[str, Any]:
        """Close database connection and cleanup."""
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
                self.metadata = None
                self.schema_cache = {}
            
            return {'status': 'success', 'message': 'Database connection closed'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to close connection: {str(e)}'}

# Factory function
def create_sql_database_tool(connection_string: str = None, database_type: str = None) -> SQLDatabaseTool:
    """Create SQL database tool"""
    return SQLDatabaseTool(connection_string, database_type)

# Quick functions
def quick_query(connection_string: str, query: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to execute a single query.
    """
    tool = SQLDatabaseTool()
    connect_result = tool.connect(connection_string)
    
    if connect_result['status'] == 'error':
        return connect_result
    
    result = tool.execute_query(query, **kwargs)
    tool.close_connection()
    
    return result

def create_sample_database(file_path: str = "sample.db") -> SQLDatabaseTool:
    """
    Create a sample SQLite database for testing.
    """
    tool = SQLDatabaseTool()
    tool.connect(f"sqlite:///{file_path}")
    
    # Create sample tables
    sample_queries = [
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_name TEXT,
            price DECIMAL(10,2),
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        INSERT INTO users (name, email, age) VALUES 
        ('John Doe', 'john@example.com', 30),
        ('Jane Smith', 'jane@example.com', 25),
        ('Bob Johnson', 'bob@example.com', 35)
        """,
        """
        INSERT INTO orders (user_id, product_name, price) VALUES 
        (1, 'Laptop', 999.99),
        (1, 'Mouse', 29.99),
        (2, 'Keyboard', 79.99),
        (3, 'Monitor', 299.99)
        """
    ]
    
    for query in sample_queries:
        tool.execute_query(query)
    
    # Refresh schema cache
    tool._cache_schema_info()
    
    return tool 