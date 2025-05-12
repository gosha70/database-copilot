#!/usr/bin/env python3
"""
Script to list all enhancement files and their descriptions.
This helps users navigate the enhancement documentation.
"""
import os
import re
from typing import Dict, List, Tuple

def get_file_description(file_path: str) -> str:
    """
    Extract a brief description from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        A brief description of the file
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read(1000)  # Read first 1000 characters
            
            # For Python files, extract docstring
            if file_path.endswith('.py'):
                docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if docstring_match:
                    docstring = docstring_match.group(1).strip()
                    first_line = docstring.split('\n')[0].strip()
                    return first_line
            
            # For Markdown files, extract first heading and paragraph
            elif file_path.endswith('.md'):
                # Extract title (first heading)
                title_match = re.search(r'# (.*?)\n', content)
                if title_match:
                    title = title_match.group(1).strip()
                    
                    # Try to find the first paragraph after the title
                    paragraphs = content.split('\n\n')
                    if len(paragraphs) > 1:
                        # Find first non-heading paragraph
                        for para in paragraphs[1:]:
                            if not para.strip().startswith('#'):
                                # Clean up paragraph (remove markdown, limit length)
                                para = re.sub(r'\[.*?\]\(.*?\)', '', para)  # Remove links
                                para = re.sub(r'\*\*(.*?)\*\*', r'\1', para)  # Remove bold
                                para = re.sub(r'\*(.*?)\*', r'\1', para)  # Remove italic
                                para = para.strip()
                                if para:
                                    # Limit to first sentence or 100 chars
                                    first_sentence = para.split('.')[0].strip()
                                    if len(first_sentence) > 100:
                                        first_sentence = first_sentence[:97] + '...'
                                    return f"{title} - {first_sentence}"
                    
                    return title
            
            # Default: return first non-empty line
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('"""'):
                    return line[:100] + ('...' if len(line) > 100 else '')
            
            return "No description available"
    
    except Exception as e:
        return f"Error reading file: {e}"

def list_enhancement_files(directory: str = 'docs') -> List[Tuple[str, str]]:
    """
    List all enhancement files in the specified directory.
    
    Args:
        directory: Directory to search for enhancement files
        
    Returns:
        A list of tuples (file_path, description)
    """
    enhancement_files = []
    
    # List of enhancement-related file patterns
    enhancement_patterns = [
        r'enhancement.*\.md',
        r'.*_example\.py',
        r'benchmark.*\.py',
        r'cascade.*\.py',
        r'performance.*\.py'
    ]
    
    # Compile patterns
    compiled_patterns = [re.compile(pattern) for pattern in enhancement_patterns]
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if file matches any enhancement pattern
            if any(pattern.match(file) for pattern in compiled_patterns):
                file_path = os.path.join(root, file)
                description = get_file_description(file_path)
                enhancement_files.append((file_path, description))
    
    # Sort by file name
    enhancement_files.sort(key=lambda x: x[0])
    
    return enhancement_files

def print_enhancement_files(enhancement_files: List[Tuple[str, str]]) -> None:
    """
    Print enhancement files and their descriptions in a formatted table.
    
    Args:
        enhancement_files: A list of tuples (file_path, description)
    """
    if not enhancement_files:
        print("No enhancement files found.")
        return
    
    # Determine the maximum width for the file path column
    max_path_width = max(len(file_path) for file_path, _ in enhancement_files)
    max_path_width = min(max_path_width, 40)  # Limit to 40 characters
    
    # Print header
    print("\nDatabase Copilot Enhancement Files:")
    print("-" * (max_path_width + 60))
    print(f"{'File':<{max_path_width}} | {'Description':<56}")
    print("-" * (max_path_width + 60))
    
    # Print each file and its description
    for file_path, description in enhancement_files:
        # Truncate file path if too long
        if len(file_path) > max_path_width:
            display_path = file_path[:max_path_width-3] + '...'
        else:
            display_path = file_path
        
        # Truncate description if too long
        if len(description) > 56:
            display_desc = description[:53] + '...'
        else:
            display_desc = description
        
        print(f"{display_path:<{max_path_width}} | {display_desc:<56}")
    
    print("-" * (max_path_width + 60))
    print(f"Total: {len(enhancement_files)} enhancement files")

def main():
    """
    Main function to list enhancement files.
    """
    # Get enhancement files
    enhancement_files = list_enhancement_files()
    
    # Print enhancement files
    print_enhancement_files(enhancement_files)

if __name__ == '__main__':
    main()
