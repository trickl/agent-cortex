"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

File System Tool - Manages file system operations including directory traversal, file manipulation, and path operations with cross-platform support.
"""

import os
import json
import logging
from typing import List, Dict, Union, Optional, Any # Ensure Any is imported if used in type hints
from llmflow.tools.tool_decorator import register_tool # Updated path

# Setup logging for this module
# ... rest of the file ...

@register_tool(tags=["file_system", "read", "file_io"])
def read_file(file_path: str, num_lines: Optional[int] = None) -> Dict[str, Any]:
    """Reads the content of a specified file.

    Args:
        file_path: The path to the file to read (relative to project root).
        num_lines: Optional. If provided, reads only this many lines from the start of the file.

    Returns:
        A dictionary containing the file path and its content or an error message.
    """
    print(f"[Tool:read_file] Attempting to read: {file_path}")

    # Prevent absolute paths and path traversal
    if os.path.isabs(file_path) or ".." in file_path:
        return {"file_path": file_path, "error": "Path must be relative to the project root and not contain '..'."}

    # Construct full path relative to project root (assuming script is run from project root)
    # This might need adjustment if the execution context changes.
    # For now, we assume direct relative paths are fine.
    actual_file_path = os.path.normpath(file_path)

    try:
        if not os.path.exists(actual_file_path):
            return {"file_path": actual_file_path, "error": f"File not found: {actual_file_path}"}
        
        if not os.path.isfile(actual_file_path):
            return {"file_path": actual_file_path, "error": f"Path is not a file: {actual_file_path}"}

        with open(actual_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        content = "".join(lines)
            
        read_lines_list = content.splitlines()
        if num_lines is not None and num_lines > 0:
            content_to_return = "\\n".join(read_lines_list[:num_lines])
        else:
            content_to_return = content
        
        return {"file_path": actual_file_path, "content": content_to_return, "lines_read": len(content_to_return.splitlines()) if content_to_return else 0}
    
    except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
        return {"file_path": actual_file_path, "error": f"File not found: {actual_file_path}"}
    except Exception as e:
        return {"file_path": actual_file_path, "error": f"An unexpected error occurred while reading file '{actual_file_path}': {str(e)}"}

@register_tool(tags=["file_system", "write", "file_io"])
def write_file(file_path: str, content: str, mode: Optional[str] = 'w') -> Dict[str, Any]:
    """Writes content to a specified file (relative to project root).

    Args:
        file_path: The relative path to the file.
        content: The content to write to the file.
        mode: File opening mode ('w' for write/overwrite, 'a' for append). Defaults to 'w'.

    Returns:
        A dictionary with the status of the write operation.
    """
    print(f"[Tool:write_file] Attempting to write to: {file_path} with mode '{mode}'")
    
    if os.path.isabs(file_path) or ".." in file_path:
        return {"file_path": file_path, "status": "error", "message": "Path must be relative to the project root and not contain '..'."}

    actual_file_path = os.path.normpath(file_path)

    if mode not in ['w', 'a']:
        return {"file_path": actual_file_path, "status": "error", "message": f"Invalid write mode '{mode}'. Allowed modes are 'w' (overwrite) and 'a' (append)."}

    try:
        # Ensure directory exists if writing to a nested path
        dir_name = os.path.dirname(actual_file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"[Tool:write_file] Created directory: {dir_name}")

        with open(actual_file_path, mode, encoding='utf-8') as f:
            f.write(content)
        char_count = len(content)
        return {"file_path": actual_file_path, "status": "success", "message": f"Content written to {actual_file_path} ({char_count} characters).", "chars_written": char_count}
    except Exception as e:
        return {"file_path": actual_file_path, "status": "error", "message": f"An error occurred while writing to file '{actual_file_path}': {str(e)}"}

@register_tool(tags=["file_system", "list", "file_io"])
def list_directory(directory_path: Optional[str] = ".") -> Dict[str, Any]:
    """Lists the contents of a specified directory (relative to project root).

    Args:
        directory_path: Optional. The relative path to the directory. Defaults to the project root ('.').

    Returns:
        A dictionary containing the directory path and a list of its contents, or an error message.
    """
    print(f"[Tool:list_directory] Attempting to list: {directory_path}")

    if directory_path is None:
        directory_path = "." # Default to current directory

    if os.path.isabs(directory_path) or ".." in directory_path and directory_path != ".": # allow "." but not "../foo"
         # Correcting the condition to allow "." explicitly while disallowing ".." for traversal.
        if directory_path == ".":
            actual_dir_path = os.path.normpath(".") # current directory
        else:
            return {"directory_path": directory_path, "error": "Path must be relative to the project root and not contain '..' unless it's just '.'."}
    else:
        actual_dir_path = os.path.normpath(directory_path)
    
    try:
        if not os.path.exists(actual_dir_path):
            return {"directory_path": actual_dir_path, "error": f"Directory not found: {actual_dir_path}"}
        if not os.path.isdir(actual_dir_path):
            return {"directory_path": actual_dir_path, "error": f"Path is not a directory: {actual_dir_path}"}
        
        contents = os.listdir(actual_dir_path)
        return {"directory_path": actual_dir_path, "contents": contents, "count": len(contents)}
    except Exception as e:
        return {"directory_path": actual_dir_path, "error": f"An error occurred while listing directory '{actual_dir_path}': {str(e)}"}


if __name__ == '__main__':
    print("--- File System Tools Example Usage (direct execution) ---")

    # Create a dummy test_file.txt in the project root for these examples to work if run directly
    if not os.path.exists("test_file.txt"):
        with open("test_file.txt", "w") as f:
            f.write("This is a test file created for direct execution of file_system_tools.py.")
    
    # Test write operations
    write_result1 = write_file("test_output/new_doc.txt", "Hello from file_system_tools!\\nThis is a test.")
    print(f"Write Result 1: {json.dumps(write_result1, indent=2)}")
    
    if write_result1.get("status") == "success":
        write_result2 = write_file("test_output/new_doc.txt", "\\nAppending a new line.", mode='a')
        print(f"Write Result 2 (append): {json.dumps(write_result2, indent=2)}")

    write_result_invalid_path = write_file("../outside_project.txt", "Attempting to write outside.")
    print(f"Write Result (invalid path): {json.dumps(write_result_invalid_path, indent=2)}")

    # Test read operations
    # Ensure the file to read exists, e.g. the one created by write_file
    if write_result1.get("status") == "success":
        read_result1 = read_file("test_output/new_doc.txt")
        print(f"Read Result 1 (new_doc.txt): {json.dumps(read_result1, indent=2)}")

    read_result_example = read_file("example.txt") # Assumes example.txt exists in project root
    print(f"Read Result (example.txt): {json.dumps(read_result_example, indent=2)}")

    if read_result_example.get("content"):
        read_result_lines = read_file("example.txt", num_lines=1) # Read only 1 line from example.txt
        print(f"Read Result (example.txt, 1 line): {json.dumps(read_result_lines, indent=2)}")

    read_result_nonexistent = read_file("non_existent_file.txt")
    print(f"Read Result (non_existent_file.txt): {json.dumps(read_result_nonexistent, indent=2)}")

    # Test list directory
    list_result_root = list_directory(".") 
    print(f"List Directory ('.', project root): {json.dumps(list_result_root, indent=2)}")

    # Create a subdirectory for testing listing it
    test_sub_dir = "test_output/my_test_subdir"
    if not os.path.exists(test_sub_dir):
        os.makedirs(test_sub_dir)
    with open(os.path.join(test_sub_dir, "sub_file.txt"), 'w') as f:
        f.write("File in subdirectory for testing list_directory.")
    
    list_result_subdir = list_directory("test_output/my_test_subdir")
    print(f"List Directory ('test_output/my_test_subdir'): {json.dumps(list_result_subdir, indent=2)}")

    list_result_invalid = list_directory("../another_place")
    print(f"List Directory (invalid path): {json.dumps(list_result_invalid, indent=2)}")

    print("\\nFile system tool examples finished. Check the 'test_output' directory.") 