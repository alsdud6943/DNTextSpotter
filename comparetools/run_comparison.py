#!/usr/bin/env python3
"""
Simple test script to run the comparison using the YAML configuration.
"""

import subprocess
import sys
import os

def main():
    # Path to the compare script and config
    # compare_script = "/root/DNTextSpotter/comparetools/compare.py"
    compare_script = "/root/DNTextSpotter/comparetools/compare_only_numbers.py"
    config_file = "/root/DNTextSpotter/comparetools/config.yaml"
    
    print("Running comparison using configuration file...")
    print(f"Config file: {config_file}")
    print()
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        print("Please create the config.yaml file first.")
        return 1
    
    # Run the comparison with config file
    cmd = [sys.executable, compare_script, "--config", config_file]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Comparison failed with error: {e}")
        return e.returncode
    except Exception as e:
        print(f"Error running comparison: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# python comparetools/run_comparison.py