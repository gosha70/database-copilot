#!/bin/bash
# Script to run the optimized Database Copilot application with different configurations

# Function to display usage information
show_usage() {
    echo "Usage: ./run_optimized.sh [option]"
    echo "Options:"
    echo "  low      - Run with minimal resource usage (1 CPU core, 1GB memory)"
    echo "  medium   - Run with moderate resource usage (2 CPU cores, 2GB memory)"
    echo "  high     - Run with higher resource usage (4 CPU cores, 4GB memory)"
    echo "  external - Run using external LLM API instead of local model"
    echo "  help     - Show this help message"
    echo ""
    echo "Example: ./run_optimized.sh medium"
}

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No option specified."
    show_usage
    exit 1
fi

# Process the argument
case "$1" in
    low)
        echo "Running with minimal resource usage..."
        python run_app_optimized.py --cpu-limit 1 --memory-limit 1024 --disable-vector-store --lazy-load
        ;;
    medium)
        echo "Running with moderate resource usage..."
        python run_app_optimized.py --cpu-limit 2 --memory-limit 2048 --lazy-load
        ;;
    high)
        echo "Running with higher resource usage..."
        python run_app_optimized.py --cpu-limit 4 --memory-limit 4096
        ;;
    external)
        echo "Running with external LLM API..."
        echo "Note: Make sure you have set up your API keys in .streamlit/secrets.toml"
        python run_app_optimized.py --use-external-llm --lazy-load
        ;;
    help)
        show_usage
        ;;
    *)
        echo "Error: Unknown option '$1'"
        show_usage
        exit 1
        ;;
esac
