# HF Space Benchmark

HF Space Benchmark is a lightweight benchmarking tool for measuring inference latency, streaming speed, and end-to-end response time for models deployed on Hugging Face Spaces using the Gradio Queue API with SSE (Server-Sent Events) streaming.

It sends repeated requests to a Gradio queue-based Space, properly handles SSE event streams, measures comprehensive performance metrics, and produces aggregated statistics. This is useful for validating deployments, comparing model setups, tuning infrastructure, and analyzing streaming performance.

---

## Features

* **SSE Stream Handling**: Properly parses Server-Sent Events (SSE) from Gradio Queue API
* **Run N Benchmark Iterations**: Execute multiple runs with automatic aggregation
* **Comprehensive Performance Metrics**:
  * Queue submit time (POST request latency)
  * Time to first SSE (initial response latency)
  * Stream time (duration of SSE streaming)
  * Total time (end-to-end latency)
* **Debug Modes**: Configurable debug flags to inspect POST responses, SSE lines, and model output
* **Model Output Extraction**: Automatically extracts generated text from SSE stream
* **Robust Error Handling**: Continues benchmarking even if individual runs fail
* **Clear Output Formatting**: Well-structured console output with visual separators
* **Minimal Dependencies**: Only requires the `requests` library (no sseclient)
* **Easy to Extend**: Clean code structure with clear separation of concerns

---

## Project Structure

```
hf_space_benchmark/
│
├── hf_benchmark.py
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/darbynova/hf-space-benchmark.git
cd hf-space-benchmark
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the script with:

```bash
python hf_benchmark.py "SYSTEM_PROMPT" "USER_PROMPT" N
```

Parameters:

* SYSTEM_PROMPT: Instruction or system message for the model
* USER_PROMPT: Query or user message
* N: Number of benchmark iterations

Example:

```bash
python hf_benchmark.py \
  "You are a helpful assistant." \
  "Explain the difference between JVM and JDK." \
  5
```

---

## Example Output

```
================================================================================
BENCHMARK: Running 5 iteration(s)
================================================================================
System prompt: You are a helpful assistant...
User prompt:   Explain the difference between JVM and JDK...
================================================================================

────────────────────────────────────────────────────────────────────────────────
RUN 1/5
────────────────────────────────────────────────────────────────────────────────
================================================================================
FINAL MODEL OUTPUT:
================================================================================
The Java Virtual Machine (JVM) is a runtime environment for running, debugging,
and executing Java code. It provides several benefits over the classpath-based
version of the Java Development Kit (JDK)...
================================================================================

✓ Queue submit time:    0.188s
✓ Time to first SSE:    1.932s
✓ Stream time:          3.284s
✓ Total time:           5.216s
✓ SSE events received:  15
✓ Output preview:       The Java Virtual Machine (JVM) is a runtime environment...

────────────────────────────────────────────────────────────────────────────────
RUN 2/5
────────────────────────────────────────────────────────────────────────────────
...

================================================================================
BENCHMARK SUMMARY
================================================================================
Successful runs: 5/5

Queue Submit Time:
  Average: 0.201s
  Minimum: 0.188s
  Maximum: 0.221s

Time to First SSE:
  Average: 1.912s
  Minimum: 1.876s
  Maximum: 1.942s

Stream Time:
  Average: 3.331s
  Minimum: 3.301s
  Maximum: 3.384s

Total Time:
  Average: 5.258s
  Minimum: 5.189s
  Maximum: 5.326s

================================================================================
```

---

## Understanding the Metrics

### Queue Submit Time
Time required for the initial POST request that places the job in the Gradio queue and returns an `event_id`. This represents the queue API overhead.

### Time to First SSE
The delay from the initial POST request until the first SSE data line is received. This includes queue wait time, model loading (if cold start), and the time to generate the first token. **This is the latency users feel before seeing any response.**

### Stream Time
The duration from receiving the first SSE data line to the last SSE data line (typically when `msg: "process_completed"` is received). This represents the actual streaming/generation time as tokens are produced.

### Total Time
End-to-end duration from the initial POST request to the completion of the SSE stream. This is the sum of all phases: `queue_submit_time + (time_to_first_sse - queue_submit_time) + stream_time`.

---

## How SSE Streaming Works

The benchmark script properly handles the Gradio Queue API's SSE streaming protocol:

1. **POST Request**: Submits the inference request and receives an `event_id`
2. **GET Request**: Opens an SSE stream using the `event_id`
3. **SSE Event Parsing**:
   - Reads lines starting with `data:`
   - Parses JSON payloads from each SSE event
   - Extracts model output from either `output.data[0]` or `data[0]`
   - Stops when `msg == "process_completed"`

The extracted text matches exactly what you would see using `curl`, ensuring accurate benchmarking of the complete model response.

These metrics help evaluate responsiveness, cold starts, GPU warm-up behavior, streaming performance, and overall inference efficiency.

---

## Cold Start and Warm Start

Hugging Face Spaces can go idle and shut down after a period of inactivity. When the container restarts:

* The model weights must reload
* GPU drivers initialize
* Queues warm up

The first run may take significantly longer than subsequent runs.

You can run a warm up iteration before benchmarking:

```bash
python hf_benchmark.py "system" "warmup" 1
```

Then run the real benchmark:

```bash
python hf_benchmark.py "system" "user" 10
```

---

## Configuration

### Space URL

Update the Space URL in [hf_benchmark.py:25](hf_benchmark.py#L25):

```python
SPACE_URL = "https://YOUR-SPACE.hf.space/gradio_api/call/predict"
```

### Debug Flags

Enable debug output by modifying the flags at [hf_benchmark.py:28-30](hf_benchmark.py#L28-L30):

```python
DEBUG_POST_RESPONSE = False  # Set to True to print full POST JSON response
DEBUG_SSE_LINES = False      # Set to True to print each raw SSE line as received
DEBUG_FINAL_TEXT = True      # Set to True to print the final extracted model text
```

### Request Payload

Modify the payload structure in [hf_benchmark.py:47-55](hf_benchmark.py#L47-L55) to match your model's input format:

```python
payload = {
    "data": [
        system_prompt,
        user_prompt,
        512,  # max_tokens
        0.1,  # temperature
        0.1   # top_p
    ]
}
```

Adjust the parameters based on your model's API requirements.

---

## Technical Details

### SSE Response Format

The Gradio Queue API returns SSE events in three common formats:

**Format 1** (with explicit message type):
```json
{
  "msg": "process_completed",
  "output": {
    "data": ["The generated text here...", null]
  }
}
```

**Format 2** (data object):
```json
{
  "data": ["The generated text here...", null]
}
```

**Format 3** (direct array):
```json
["The generated text here..."]
```

The benchmark script handles all three formats automatically by checking for `output.data[0]` first, then `data[0]`, and finally treating the response as a direct array.

### Timing Measurements

All timing uses `time.monotonic()` for accurate, non-adjustable measurements that are not affected by system clock changes. This ensures reliable benchmarking even during daylight saving time changes or NTP adjustments.

### Error Handling

- Individual run failures do not stop the benchmark
- Network timeouts are set to 120s for POST and 600s for GET (streaming)
- JSON parsing errors are caught and logged when debug mode is enabled
- Missing or malformed SSE events are handled gracefully

---

## Troubleshooting

### No Model Output Extracted

If you see "⚠ Warning: No model output extracted":

1. Enable `DEBUG_SSE_LINES = True` to inspect the raw SSE events
2. Check the JSON structure matches one of the expected formats
3. Verify your Space is returning `data` fields correctly

### Connection Timeouts

If requests timeout:

1. Check if your Space is awake (may need a cold start)
2. Increase timeout values in [hf_benchmark.py:69](hf_benchmark.py#L69) (POST) and [hf_benchmark.py:114](hf_benchmark.py#L114) (GET)
3. Try a warm-up run first

### Event ID Not Found

If "Could not extract event_id from response":

1. Enable `DEBUG_POST_RESPONSE = True` to see the full POST response
2. Verify the Space URL is correct and includes `/gradio_api/call/predict`
3. Check if the API contract has changed

---


