#!/usr/bin/env python3
"""
Benchmark script for Hugging Face Spaces Gradio Queue API with SSE stream handling.

This script measures latency metrics for model inference through HF Spaces queue API:
- queue_submit_time: Time to submit request and get event_id
- time_to_first_sse: Time from POST to first SSE data line
- stream_time: Duration of SSE stream processing
- total_time: End-to-end request time

Usage:
    python hf_benchmark.py "system prompt" "user prompt" N
"""

import sys
import json
import time
import statistics
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

SPACE_URL = "https://darbynova-demo-model-deployment.hf.space/gradio_api/call/predict"

# Debug flags - set to True to enable detailed logging
DEBUG_POST_RESPONSE = False  # Print full POST JSON response
DEBUG_SSE_LINES = False      # Print each raw SSE line as received
DEBUG_FINAL_TEXT = True      # Print the final extracted model text


# ============================================================================
# Core Functions
# ============================================================================

def run_once(system_prompt, user_prompt):
    """
    Execute one complete benchmark cycle:
      1) POST to queue endpoint -> get event_id
      2) GET to stream endpoint -> process SSE events
      3) Extract model output and capture timing metrics

    Returns:
        dict: Metrics including timing data and extracted text
    """
    payload = {
        "data": [
            system_prompt,
            user_prompt,
            512,  # max_tokens
            0.1,  # temperature
            0.1   # top_p
        ]
    }

    t_start = time.monotonic()

    # ========================================================================
    # Step 1: POST to queue endpoint
    # ========================================================================
    if DEBUG_POST_RESPONSE or DEBUG_SSE_LINES:
        print("→ Submitting POST request...")

    response = requests.post(
        SPACE_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120
    )
    t_after_post = time.monotonic()

    if response.status_code != 200:
        raise RuntimeError(
            f"POST failed with status {response.status_code}: {response.text[:500]}"
        )

    try:
        resp_json = response.json()
    except ValueError as e:
        raise RuntimeError(
            f"POST returned invalid JSON: {response.text[:500]}"
        ) from e

    if DEBUG_POST_RESPONSE:
        print(f"POST Response JSON:\n{json.dumps(resp_json, indent=2)}\n")

    # Extract event_id from response
    event_id = None
    if isinstance(resp_json, dict):
        if "event_id" in resp_json:
            event_id = resp_json["event_id"]
        elif "data" in resp_json and isinstance(resp_json["data"], list) and resp_json["data"]:
            event_id = resp_json["data"][0]

    if not event_id:
        raise RuntimeError(
            f"Could not extract event_id from response: {resp_json}"
        )

    if DEBUG_POST_RESPONSE or DEBUG_SSE_LINES:
        print(f"✓ Got event_id: {event_id}\n")

    queue_submit_time = t_after_post - t_start

    # ========================================================================
    # Step 2: GET from stream endpoint and process SSE
    # ========================================================================
    stream_url = f"{SPACE_URL}/{event_id}"

    if DEBUG_SSE_LINES:
        print(f"→ Starting GET request to: {stream_url}\n")

    stream_resp = requests.get(stream_url, stream=True, timeout=600)

    if stream_resp.status_code != 200:
        raise RuntimeError(
            f"GET failed with status {stream_resp.status_code}: {stream_resp.text[:500]}"
        )

    # Process SSE stream
    model_output = None
    t_first_sse = None
    t_last_sse = None
    sse_count = 0

    for line in stream_resp.iter_lines(decode_unicode=True):
        if not line:
            continue

        if DEBUG_SSE_LINES:
            print(f"SSE Line: {line}")

        # Only process lines that start with "data:"
        if not line.startswith("data:"):
            continue

        # Mark first SSE data line
        if t_first_sse is None:
            t_first_sse = time.monotonic()

        # Update last SSE timestamp
        t_last_sse = time.monotonic()
        sse_count += 1

        # Extract JSON payload after "data:" prefix
        json_str = line[5:].strip()  # Remove "data:" prefix

        try:
            sse_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if DEBUG_SSE_LINES:
                print(f"⚠ Warning: Could not parse SSE JSON: {json_str[:100]}")
            continue

        if DEBUG_SSE_LINES:
            print(f"  Parsed: {json.dumps(sse_data, indent=2)}\n")

        # Extract model output from the JSON payload
        # Format 1: {"msg": "process_completed", "output": {"data": ["text", ...]}}
        # Format 2: {"data": ["text", ...]}
        # Format 3: ["text", ...]  (direct array)
        extracted_text = None

        if isinstance(sse_data, dict):
            # Try Format 1: output.data[0]
            if "output" in sse_data and isinstance(sse_data["output"], dict):
                output_obj = sse_data["output"]
                if "data" in output_obj and isinstance(output_obj["data"], list):
                    if output_obj["data"]:
                        extracted_text = output_obj["data"][0]

            # Try Format 2: data[0]
            elif "data" in sse_data and isinstance(sse_data["data"], list):
                if sse_data["data"]:
                    extracted_text = sse_data["data"][0]

            # Update model_output if we extracted text
            if extracted_text is not None:
                model_output = extracted_text

            # Stop when we see process_completed message
            if sse_data.get("msg") == "process_completed":
                if DEBUG_SSE_LINES:
                    print("✓ Received process_completed message\n")
                break

        # Try Format 3: Direct array ["text", ...]
        elif isinstance(sse_data, list) and sse_data:
            extracted_text = sse_data[0]
            if extracted_text is not None:
                model_output = extracted_text

    t_end = time.monotonic()

    # Calculate metrics
    time_to_first_sse = (t_first_sse - t_start) if t_first_sse else 0
    stream_time = (t_last_sse - t_first_sse) if (t_first_sse and t_last_sse) else 0
    total_time = t_end - t_start

    if DEBUG_FINAL_TEXT and model_output:
        print("\n" + "="*80)
        print("FINAL MODEL OUTPUT:")
        print("="*80)
        print(model_output)
        print("="*80 + "\n")

    return {
        "queue_submit_time": queue_submit_time,
        "time_to_first_sse": time_to_first_sse,
        "stream_time": stream_time,
        "total_time": total_time,
        "model_output": model_output,
        "sse_events_count": sse_count
    }


def summarize_metric(name, data):
    """
    Generate summary statistics for a single metric.

    Args:
        name: Metric name
        data: List of numeric values

    Returns:
        str: Formatted summary string
    """
    if not data:
        return f"{name}: (no data)"

    avg = statistics.mean(data)
    min_val = min(data)
    max_val = max(data)

    return (
        f"{name}:\n"
        f"  Average: {avg:.3f}s\n"
        f"  Minimum: {min_val:.3f}s\n"
        f"  Maximum: {max_val:.3f}s"
    )


def generate_stacked_bar_chart(results, output_path="benchmark_stacked.png"):
    """
    Generate a stacked bar chart showing where time is spent in each run.

    Args:
        results: List of result dictionaries from benchmark runs
        output_path: Path to save the chart
    """
    if not results:
        print("⚠ No data to visualize")
        return

    runs = list(range(1, len(results) + 1))

    # Extract timing components for each run
    queue_times = [r["queue_submit_time"] for r in results]

    # Calculate time between queue submission and first SSE (waiting time)
    wait_times = [r["time_to_first_sse"] - r["queue_submit_time"] for r in results]

    # Stream time (actual generation)
    stream_times = [r["stream_time"] for r in results]

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.6
    x = np.arange(len(runs))

    # Stack the bars
    p1 = ax.bar(x, queue_times, bar_width, label='Queue Submit', color='#3498db')
    p2 = ax.bar(x, wait_times, bar_width, bottom=queue_times,
                label='Wait for First SSE', color='#e74c3c')
    p3 = ax.bar(x, stream_times, bar_width,
                bottom=np.array(queue_times) + np.array(wait_times),
                label='Stream Time', color='#2ecc71')

    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Benchmark Time Breakdown - Stacked Bar Chart',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Run {i}' for i in runs])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total time labels on top of bars
    for i, (q, w, s) in enumerate(zip(queue_times, wait_times, stream_times)):
        total = q + w + s
        ax.text(i, total + max(stream_times) * 0.02, f'{total:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stacked bar chart to: {output_path}")
    plt.close()


def generate_grouped_bar_chart(results, output_path="benchmark_grouped.png"):
    """
    Generate a grouped bar chart showing avg/min/max for each metric.

    Args:
        results: List of result dictionaries from benchmark runs
        output_path: Path to save the chart
    """
    if not results:
        print("⚠ No data to visualize")
        return

    # Extract metrics
    metrics = {
        'Queue Submit': [r["queue_submit_time"] for r in results],
        'Time to First SSE': [r["time_to_first_sse"] for r in results],
        'Stream Time': [r["stream_time"] for r in results],
        'Total Time': [r["total_time"] for r in results]
    }

    # Calculate statistics for each metric
    stats = {}
    for name, values in metrics.items():
        stats[name] = {
            'avg': statistics.mean(values),
            'min': min(values),
            'max': max(values)
        }

    # Prepare data for plotting
    metric_names = list(stats.keys())
    avg_values = [stats[m]['avg'] for m in metric_names]
    min_values = [stats[m]['min'] for m in metric_names]
    max_values = [stats[m]['max'] for m in metric_names]

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(metric_names))
    bar_width = 0.25

    # Create bars for each statistic
    bars1 = ax.bar(x - bar_width, avg_values, bar_width,
                   label='Average', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, min_values, bar_width,
                   label='Minimum', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + bar_width, max_values, bar_width,
                   label='Maximum', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Benchmark Performance Variation - Grouped Bar Chart',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s',
                    ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved grouped bar chart to: {output_path}")
    plt.close()


# ============================================================================
# Main Benchmark Loop
# ============================================================================

def main():
    """
    Main benchmark execution:
      - Parse command line arguments
      - Run N benchmark iterations
      - Collect and summarize metrics
    """
    if len(sys.argv) < 4:
        print("Usage: python hf_benchmark.py \"system prompt\" \"user prompt\" N")
        print("\nExample:")
        print('  python hf_benchmark.py "You are a helpful assistant" "What is the JVM?" 5')
        sys.exit(1)

    system_prompt = sys.argv[1]
    user_prompt = sys.argv[2]

    try:
        runs = int(sys.argv[3])
        if runs <= 0:
            raise ValueError("Number of runs must be positive")
    except ValueError as e:
        print(f"Error: Invalid number of runs: {sys.argv[3]}")
        print(f"  {e}")
        sys.exit(1)

    results = []

    print("\n" + "="*80)
    print(f"BENCHMARK: Running {runs} iteration(s)")
    print("="*80)
    print(f"System prompt: {system_prompt[:60]}...")
    print(f"User prompt:   {user_prompt[:60]}...")
    print("="*80 + "\n")

    # Run benchmark iterations
    for i in range(1, runs + 1):
        print(f"{'─'*80}")
        print(f"RUN {i}/{runs}")
        print(f"{'─'*80}")

        try:
            metrics = run_once(system_prompt, user_prompt)
            results.append(metrics)

            print(f"✓ Queue submit time:    {metrics['queue_submit_time']:.3f}s")
            print(f"✓ Time to first SSE:    {metrics['time_to_first_sse']:.3f}s")
            print(f"✓ Stream time:          {metrics['stream_time']:.3f}s")
            print(f"✓ Total time:           {metrics['total_time']:.3f}s")
            print(f"✓ SSE events received:  {metrics['sse_events_count']}")

            if metrics['model_output']:
                output_preview = metrics['model_output'][:100].replace('\n', ' ')
                print(f"✓ Output preview:       {output_preview}...")
            else:
                print("⚠ Warning: No model output extracted")

            print()

        except Exception as e:
            print(f"✗ Run {i} failed: {e}\n")
            # Continue with remaining runs

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    if not results:
        print("✗ No successful runs to summarize")
        sys.exit(1)

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Successful runs: {len(results)}/{runs}\n")

    # Extract metric arrays
    queue_times = [r["queue_submit_time"] for r in results]
    first_sse_times = [r["time_to_first_sse"] for r in results]
    stream_times = [r["stream_time"] for r in results]
    total_times = [r["total_time"] for r in results]

    print(summarize_metric("Queue Submit Time", queue_times))
    print()
    print(summarize_metric("Time to First SSE", first_sse_times))
    print()
    print(summarize_metric("Stream Time", stream_times))
    print()
    print(summarize_metric("Total Time", total_times))
    print("\n" + "="*80 + "\n")

    # ========================================================================
    # Generate Visualizations
    # ========================================================================
    print("Generating visualizations...")
    print()

    try:
        # Generate stacked bar chart showing time breakdown
        generate_stacked_bar_chart(results, "benchmark_stacked.png")

        # Generate grouped bar chart showing performance variation
        generate_grouped_bar_chart(results, "benchmark_grouped.png")

        print()
        print("="*80)
        print("✓ Benchmark complete! Check the generated PNG files for visualizations.")
        print("="*80)

    except Exception as e:
        print(f"⚠ Warning: Could not generate visualizations: {e}")
        print("  Make sure matplotlib and numpy are installed:")
        print("  pip install matplotlib numpy")


if __name__ == "__main__":
    main()
