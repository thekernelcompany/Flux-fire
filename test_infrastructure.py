#!/usr/bin/env python3
"""
Quick test to verify test infrastructure is working
"""

import subprocess
import sys

def test_pytest_installed():
    """Check if pytest is installed"""
    try:
        import pytest
        print("✅ pytest is installed")
        return True
    except ImportError:
        print("❌ pytest is not installed")
        return False

def test_basic_unit_test():
    """Run a simple unit test"""
    cmd = ["pytest", "tests/unit/test_utils.py::TestDetailedTimer::test_basic_timing", "-v"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Basic unit test passed")
        return True
    else:
        print("❌ Basic unit test failed")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    print("Testing FLUX.1-Kontext Test Infrastructure")
    print("=" * 50)
    
    # Install test dependencies
    print("\n1. Installing test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov", "pytest-mock"], 
                   capture_output=True)
    
    # Run checks
    tests = [
        test_pytest_installed,
        test_basic_unit_test,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "=" * 50)
    print("Infrastructure Test Summary:")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✅ Test infrastructure is working correctly!")
        print("\nYou can now run the full test suite with:")
        print("  python run_tests.py")
    else:
        print("\n❌ Some infrastructure tests failed.")
        print("Please check the output above for details.")

if __name__ == "__main__":
    main()