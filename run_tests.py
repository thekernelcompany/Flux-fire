#!/usr/bin/env python3
"""
Test runner script for FLUX.1-Kontext comprehensive test suite
"""

import os
import sys
import subprocess
import datetime
import json
from pathlib import Path
import argparse


def run_test_category(category, markers=None, verbose=False):
    """Run a specific category of tests"""
    cmd = ["pytest", f"tests/{category}/", "-v" if verbose else "-q"]
    
    if markers:
        cmd.extend(["-m", markers])
    
    # Add coverage for source code
    cmd.extend(["--cov=src", "--cov-append"])
    
    print(f"\n{'='*60}")
    print(f"Running {category} tests...")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "category": category,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0
    }


def run_all_tests(skip_slow=False, skip_gpu=False):
    """Run the complete test suite"""
    results = []
    
    # Define test categories and their markers
    test_categories = [
        ("unit", None),
        ("integration", None),
        ("performance", "performance and not slow" if skip_slow else "performance"),
        ("quality", "quality"),
        ("stress", "stress and not slow" if skip_slow else "stress"),
        ("edge_cases", "edge_case"),
    ]
    
    # Skip GPU tests if requested
    if skip_gpu:
        for i, (category, markers) in enumerate(test_categories):
            if markers:
                test_categories[i] = (category, f"{markers} and not gpu_required")
            else:
                test_categories[i] = (category, "not gpu_required")
    
    # Run each category
    for category, markers in test_categories:
        result = run_test_category(category, markers)
        results.append(result)
    
    return results


def generate_test_report(results, output_dir="test_results"):
    """Generate comprehensive test report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Create summary
    total_passed = sum(1 for r in results if r["passed"])
    total_categories = len(results)
    
    summary = {
        "timestamp": timestamp,
        "total_categories": total_categories,
        "passed_categories": total_passed,
        "failed_categories": total_categories - total_passed,
        "success_rate": f"{(total_passed / total_categories) * 100:.1f}%",
        "categories": {}
    }
    
    # Add category details
    for result in results:
        summary["categories"][result["category"]] = {
            "passed": result["passed"],
            "return_code": result["return_code"]
        }
    
    # Save JSON report
    json_report = output_dir / f"test_report_{timestamp.replace(':', '-')}.json"
    with open(json_report, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FLUX.1-Kontext Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .passed {{ border-left-color: #4CAF50; background-color: #f1f8f4; }}
        .failed {{ border-left-color: #f44336; background-color: #fef1f0; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-pass {{ color: #4CAF50; font-weight: bold; }}
        .status-fail {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FLUX.1-Kontext Comprehensive Test Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Test Categories: {total_categories}</p>
        <p>Passed: <span class="status-pass">{total_passed}</span></p>
        <p>Failed: <span class="status-fail">{total_categories - total_passed}</span></p>
        <p>Success Rate: <strong>{(total_passed / total_categories) * 100:.1f}%</strong></p>
    </div>
    
    <h2>Test Categories</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Status</th>
            <th>Details</th>
        </tr>
"""
    
    for result in results:
        status_class = "status-pass" if result["passed"] else "status-fail"
        status_text = "PASSED" if result["passed"] else "FAILED"
        
        html_content += f"""
        <tr>
            <td>{result["category"].title()}</td>
            <td class="{status_class}">{status_text}</td>
            <td>Return code: {result["return_code"]}</td>
        </tr>
"""
    
    html_content += """
    </table>
    
    <h2>Detailed Results</h2>
"""
    
    for result in results:
        category_class = "passed" if result["passed"] else "failed"
        html_content += f"""
    <div class="category {category_class}">
        <h3>{result["category"].title()} Tests</h3>
        <pre>{result["stdout"][:1000]}...</pre>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    html_report = output_dir / f"test_report_{timestamp.replace(':', '-')}.html"
    with open(html_report, "w") as f:
        f.write(html_content)
    
    # Generate coverage report
    subprocess.run(["coverage", "html", "-d", str(output_dir / "coverage")])
    
    return json_report, html_report


def main():
    parser = argparse.ArgumentParser(description="Run FLUX.1-Kontext test suite")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU-required tests")
    parser.add_argument("--category", help="Run only specific category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for reports")
    
    args = parser.parse_args()
    
    print("FLUX.1-Kontext Comprehensive Test Suite")
    print("=" * 60)
    
    # Install test dependencies if needed
    print("Checking test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_test.txt"], 
                   capture_output=True)
    
    # Run tests
    if args.category:
        results = [run_test_category(args.category, verbose=args.verbose)]
    else:
        results = run_all_tests(skip_slow=args.skip_slow, skip_gpu=args.skip_gpu)
    
    # Generate reports
    json_report, html_report = generate_test_report(results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for result in results:
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"{result['category']:15} {status}")
    
    print(f"\nReports saved to:")
    print(f"  JSON: {json_report}")
    print(f"  HTML: {html_report}")
    print(f"  Coverage: {args.output_dir}/coverage/index.html")
    
    # Exit with failure if any tests failed
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()