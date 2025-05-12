#!/usr/bin/env python3
"""
Test script for the enhanced Liquibase reviewer.

This script demonstrates how to use the enhanced Liquibase reviewer
to review a Liquibase migration file.
"""
import os
import sys
import argparse
import logging
import importlib.util
import subprocess
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = [
    "langchain",
    "langchain_community",
    "langchain_core",
    "langchain_chroma",
    "pydantic"
]

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        A tuple of (all_installed, missing_packages)
    """
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(package.split(">=")[0])
            if spec is None:
                missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def install_package(package: str) -> bool:
    """
    Install a package using pip.
    
    Args:
        package: The package to install
        
    Returns:
        True if installation was successful, False otherwise
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies(packages: List[str]) -> bool:
    """
    Install missing dependencies.
    
    Args:
        packages: List of packages to install
        
    Returns:
        True if all packages were installed successfully, False otherwise
    """
    all_installed = True
    
    for package in packages:
        print(f"Installing {package}...")
        if not install_package(package):
            print(f"Failed to install {package}")
            all_installed = False
    
    return all_installed

def load_migration_file(file_path: str) -> str:
    """
    Load a migration file.
    
    Args:
        file_path: Path to the migration file
        
    Returns:
        The content of the migration file
    """
    with open(file_path, 'r') as f:
        return f.read()

def test_original_reviewer(migration_content: str, format_type: str) -> Optional[str]:
    """
    Test the original LiquibaseReviewer implementation.
    
    Args:
        migration_content: The content of the migration file
        format_type: The format of the migration file (xml or yaml)
        
    Returns:
        The review of the migration, or None if an error occurred
    """
    try:
        from backend.models.liquibase_reviewer import LiquibaseReviewer
        
        # Initialize the reviewer
        reviewer = LiquibaseReviewer()
        
        # Review the migration
        logger.info("Reviewing migration with original reviewer...")
        review = reviewer.review_migration(migration_content, format_type)
        
        return review
    except ImportError as e:
        logger.error(f"Error importing original reviewer: {e}")
        logger.error("Make sure all dependencies are installed by running: ./install_dependencies.py")
        return None
    except Exception as e:
        logger.error(f"Error using original reviewer: {e}")
        return None

def test_enhanced_reviewer(migration_content: str, format_type: str) -> Optional[str]:
    """
    Test the enhanced LiquibaseReviewer implementation.
    
    Args:
        migration_content: The content of the migration file
        format_type: The format of the migration file (xml or yaml)
        
    Returns:
        The review of the migration, or None if an error occurred
    """
    try:
        from backend.models.enhanced_liquibase_reviewer import EnhancedLiquibaseReviewer
        
        # Initialize the reviewer
        reviewer = EnhancedLiquibaseReviewer()
        
        # Review the migration
        logger.info("Reviewing migration with enhanced reviewer...")
        review = reviewer.review_migration(migration_content, format_type)
        
        return review
    except ImportError as e:
        logger.error(f"Error importing enhanced reviewer: {e}")
        logger.error("Make sure all dependencies are installed by running: ./install_dependencies.py")
        return None
    except Exception as e:
        logger.error(f"Error using enhanced reviewer: {e}")
        return None

def compare_reviews(original_review: Optional[str], enhanced_review: Optional[str]) -> None:
    """
    Compare the original and enhanced reviews.
    
    Args:
        original_review: The review from the original reviewer
        enhanced_review: The review from the enhanced reviewer
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF REVIEWS")
    print("=" * 80)
    
    # Print lengths
    print(f"Original review length: {len(original_review)} characters")
    print(f"Enhanced review length: {len(enhanced_review)} characters")
    
    # Extract sections from reviews
    original_sections = extract_sections(original_review)
    enhanced_sections = extract_sections(enhanced_review)
    
    # Compare sections
    print("\nSections in original review:")
    for section in original_sections:
        print(f"- {section}")
    
    print("\nSections in enhanced review:")
    for section in enhanced_sections:
        print(f"- {section}")
    
    # Check for unique sections
    original_only = set(original_sections) - set(enhanced_sections)
    enhanced_only = set(enhanced_sections) - set(original_sections)
    
    if original_only:
        print("\nSections only in original review:")
        for section in original_only:
            print(f"- {section}")
    
    if enhanced_only:
        print("\nSections only in enhanced review:")
        for section in enhanced_only:
            print(f"- {section}")
    
    print("\nNote: The enhanced review should prioritize information from internal guidelines and example migrations over official Liquibase documentation.")

def extract_sections(review: str) -> List[str]:
    """
    Extract section headings from a review.
    
    Args:
        review: The review text
        
    Returns:
        A list of section headings
    """
    sections = []
    for line in review.split('\n'):
        if line.startswith('##'):
            sections.append(line.strip('# '))
        elif line.startswith('#'):
            sections.append(line.strip('# '))
    
    return sections

def save_reviews(original_review: Optional[str], enhanced_review: Optional[str], output_dir: str = "reviews") -> None:
    """
    Save the reviews to files.
    
    Args:
        original_review: The review from the original reviewer
        enhanced_review: The review from the enhanced reviewer
        output_dir: The directory to save the reviews to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original review if available
    if original_review:
        with open(os.path.join(output_dir, "original_review.md"), 'w') as f:
            f.write(original_review)
    
    # Save enhanced review if available
    if enhanced_review:
        with open(os.path.join(output_dir, "enhanced_review.md"), 'w') as f:
            f.write(enhanced_review)
    
    logger.info(f"Reviews saved to {output_dir}")

def main() -> int:
    """
    Main function to test the enhanced Liquibase reviewer.
    """
    parser = argparse.ArgumentParser(description='Test the enhanced Liquibase reviewer')
    parser.add_argument('--migration', type=str, default='examples/20250505-create-custom-table.yaml',
                        help='Path to the migration file to review')
    parser.add_argument('--format', type=str, choices=['xml', 'yaml'], default=None,
                        help='Format of the migration file (xml or yaml)')
    parser.add_argument('--output-dir', type=str, default='reviews',
                        help='Directory to save the reviews to')
    
    args = parser.parse_args()
    
    # Determine format type from file extension if not provided
    format_type = args.format
    if format_type is None:
        if args.migration.endswith('.xml'):
            format_type = 'xml'
        elif args.migration.endswith('.yaml') or args.migration.endswith('.yml'):
            format_type = 'yaml'
        else:
            raise ValueError(f"Could not determine format type from file extension: {args.migration}")
    
    # Check dependencies
    all_installed, missing_packages = check_dependencies()
    if not all_installed:
        print("Missing required dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        
        # Ask user if they want to install missing dependencies
        response = input("\nDo you want to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if install_dependencies(missing_packages):
                print("\nAll dependencies installed successfully.")
                # Reload modules
                for package in missing_packages:
                    if package in sys.modules:
                        importlib.reload(sys.modules[package])
            else:
                print("\nFailed to install some dependencies.")
                print("Please install them manually using:")
                print(f"  pip install {' '.join(missing_packages)}")
                return 1
        else:
            print("\nPlease install the missing dependencies and try again.")
            return 1
    
    # Load the migration content
    try:
        migration_content = load_migration_file(args.migration)
    except Exception as e:
        logger.error(f"Error loading migration file: {e}")
        return 1
    
    # Test the original reviewer
    original_review = test_original_reviewer(migration_content, format_type)
    
    # Test the enhanced reviewer
    enhanced_review = test_enhanced_reviewer(migration_content, format_type)
    
    # Check if at least one review was successful
    if original_review is None and enhanced_review is None:
        logger.error("Both reviewers failed. Please check the logs for details.")
        return 1
    
    # Compare the reviews if both are available
    if original_review is not None and enhanced_review is not None:
        compare_reviews(original_review, enhanced_review)
    
    # Save the reviews
    save_reviews(original_review, enhanced_review, args.output_dir)
    
    print(f"\nReviews saved to {args.output_dir}")
    if original_review:
        print(f"- Original review: {os.path.join(args.output_dir, 'original_review.md')}")
    if enhanced_review:
        print(f"- Enhanced review: {os.path.join(args.output_dir, 'enhanced_review.md')}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
