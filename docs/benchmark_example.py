"""
Example benchmark script for the Database Copilot enhancements.

This script demonstrates how to benchmark the performance of the enhanced
components and compare them with the original implementation.
"""
import os
import time
import logging
import argparse
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_example_migration(file_path: str) -> str:
    """
    Load an example migration file.
    
    Args:
        file_path: Path to the migration file
        
    Returns:
        The content of the migration file
    """
    with open(file_path, 'r') as f:
        return f.read()

def benchmark_original_reviewer(migration_content: str, format_type: str) -> Dict[str, Any]:
    """
    Benchmark the original LiquibaseReviewer implementation.
    
    Args:
        migration_content: The content of the migration file
        format_type: The format of the migration file (xml or yaml)
        
    Returns:
        A dictionary with benchmark results
    """
    from backend.models.liquibase_reviewer import LiquibaseReviewer
    
    # Initialize the reviewer
    start_init = time.time()
    reviewer = LiquibaseReviewer()
    init_time = time.time() - start_init
    
    # Review the migration
    start_review = time.time()
    review = reviewer.review_migration(migration_content, format_type)
    review_time = time.time() - start_review
    
    return {
        "implementation": "original",
        "init_time": init_time,
        "review_time": review_time,
        "total_time": init_time + review_time,
        "review_length": len(review)
    }

def benchmark_enhanced_reviewer(migration_content: str, format_type: str) -> Dict[str, Any]:
    """
    Benchmark the enhanced LiquibaseReviewer implementation.
    
    Args:
        migration_content: The content of the migration file
        format_type: The format of the migration file (xml or yaml)
        
    Returns:
        A dictionary with benchmark results
    """
    # Import the enhanced reviewer from the example implementation
    import sys
    import importlib.util
    
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(
        "cascade_retriever_example", 
        os.path.join(os.path.dirname(__file__), "cascade_retriever_example.py")
    )
    cascade_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cascade_module)
    
    # Initialize the enhanced reviewer
    start_init = time.time()
    reviewer = cascade_module.EnhancedLiquibaseReviewer()
    init_time = time.time() - start_init
    
    # Review the migration
    start_review = time.time()
    review = reviewer.review_migration(migration_content, format_type)
    review_time = time.time() - start_review
    
    return {
        "implementation": "enhanced_cascade",
        "init_time": init_time,
        "review_time": review_time,
        "total_time": init_time + review_time,
        "review_length": len(review)
    }

def benchmark_optimized_reviewer(migration_content: str, format_type: str) -> Dict[str, Any]:
    """
    Benchmark the optimized LiquibaseReviewer implementation.
    
    Args:
        migration_content: The content of the migration file
        format_type: The format of the migration file (xml or yaml)
        
    Returns:
        A dictionary with benchmark results
    """
    # Import the optimized reviewer from the example implementation
    import sys
    import importlib.util
    
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(
        "performance_optimization_example", 
        os.path.join(os.path.dirname(__file__), "performance_optimization_example.py")
    )
    perf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_module)
    
    # Initialize the optimized reviewer
    start_init = time.time()
    reviewer = perf_module.OptimizedLiquibaseReviewer()
    init_time = time.time() - start_init
    
    # Review the migration
    start_review = time.time()
    review = reviewer.review_migration(migration_content, format_type)
    review_time = time.time() - start_review
    
    return {
        "implementation": "optimized",
        "init_time": init_time,
        "review_time": review_time,
        "total_time": init_time + review_time,
        "review_length": len(review)
    }

def benchmark_cached_reviewer(migration_content: str, format_type: str, reviewer=None) -> Dict[str, Any]:
    """
    Benchmark the optimized LiquibaseReviewer implementation with caching (second run).
    
    Args:
        migration_content: The content of the migration file
        format_type: The format of the migration file (xml or yaml)
        reviewer: An existing reviewer instance (to test caching)
        
    Returns:
        A dictionary with benchmark results
    """
    # Import the optimized reviewer from the example implementation
    import sys
    import importlib.util
    
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(
        "performance_optimization_example", 
        os.path.join(os.path.dirname(__file__), "performance_optimization_example.py")
    )
    perf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_module)
    
    # Initialize the optimized reviewer if not provided
    if reviewer is None:
        start_init = time.time()
        reviewer = perf_module.OptimizedLiquibaseReviewer()
        init_time = time.time() - start_init
    else:
        init_time = 0
    
    # Review the migration
    start_review = time.time()
    review = reviewer.review_migration(migration_content, format_type)
    review_time = time.time() - start_review
    
    return {
        "implementation": "optimized_cached",
        "init_time": init_time,
        "review_time": review_time,
        "total_time": init_time + review_time,
        "review_length": len(review),
        "reviewer": reviewer
    }

def run_benchmarks(migration_path: str, format_type: str = None) -> List[Dict[str, Any]]:
    """
    Run benchmarks for different implementations.
    
    Args:
        migration_path: Path to the migration file
        format_type: The format of the migration file (xml or yaml)
        
    Returns:
        A list of benchmark results
    """
    # Determine format type from file extension if not provided
    if format_type is None:
        if migration_path.endswith('.xml'):
            format_type = 'xml'
        elif migration_path.endswith('.yaml') or migration_path.endswith('.yml'):
            format_type = 'yaml'
        else:
            raise ValueError(f"Could not determine format type from file extension: {migration_path}")
    
    # Load the migration content
    migration_content = load_example_migration(migration_path)
    
    # Run benchmarks
    results = []
    
    logger.info("Benchmarking original implementation...")
    results.append(benchmark_original_reviewer(migration_content, format_type))
    
    logger.info("Benchmarking enhanced cascade implementation...")
    results.append(benchmark_enhanced_reviewer(migration_content, format_type))
    
    logger.info("Benchmarking optimized implementation (first run)...")
    results.append(benchmark_optimized_reviewer(migration_content, format_type))
    
    logger.info("Benchmarking optimized implementation (second run, with caching)...")
    cached_result = benchmark_cached_reviewer(migration_content, format_type)
    results.append(cached_result)
    
    logger.info("Benchmarking optimized implementation (third run, with caching)...")
    results.append(benchmark_cached_reviewer(migration_content, format_type, cached_result["reviewer"]))
    
    return results

def print_benchmark_results(results: List[Dict[str, Any]]) -> None:
    """
    Print benchmark results in a table format.
    
    Args:
        results: A list of benchmark results
    """
    # Print header
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Implementation':<20} {'Init Time (s)':<15} {'Review Time (s)':<15} {'Total Time (s)':<15} {'Review Length':<15}")
    print("-" * 80)
    
    # Print results
    for result in results:
        print(f"{result['implementation']:<20} {result['init_time']:<15.2f} {result['review_time']:<15.2f} {result['total_time']:<15.2f} {result['review_length']:<15}")
    
    # Print speedup
    if len(results) > 1:
        original_time = results[0]['review_time']
        for result in results[1:]:
            speedup = original_time / result['review_time'] if result['review_time'] > 0 else float('inf')
            print(f"\n{result['implementation']} speedup vs original: {speedup:.2f}x")

def main():
    """
    Main function to run the benchmark script.
    """
    parser = argparse.ArgumentParser(description='Benchmark Database Copilot enhancements')
    parser.add_argument('--migration', type=str, default='examples/20250505-create-custom-table.yaml',
                        help='Path to the migration file to benchmark')
    parser.add_argument('--format', type=str, choices=['xml', 'yaml'], default=None,
                        help='Format of the migration file (xml or yaml)')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(args.migration, args.format)
    
    # Print results
    print_benchmark_results(results)

if __name__ == '__main__':
    main()
