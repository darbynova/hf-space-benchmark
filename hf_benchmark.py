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


if __name__ == "__main__":
    main()
