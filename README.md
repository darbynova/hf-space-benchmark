# HF Space Benchmark

HF Space Benchmark is a lightweight benchmarking tool for measuring inference latency, streaming speed, and end-to-end response time for models deployed on Hugging Face Spaces.

It sends repeated requests to a Gradio queue based Space, measures performance metrics, and produces aggregated averages, minimums, and maximums. This is useful for validating deployments, comparing model setups, and tuning infrastructure.

---

## Features

* Run N benchmark iterations
* Collect performance metrics

  * Queue submit latency
  * Time to first token
  * Streaming duration
  * Total time
* Clear separation between cold and warm behavior
* Minimal Python dependencies
* Easy to extend and customize

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
=== Running 5 benchmark iterations ===

--- Run 1 ---
Queue submit: 0.188s
Time to first token: 1.932s
Streaming:           3.284s
Total:               5.216s

--- Run 2 ---
...

=== Benchmark Summary ===

Queue submit latency:
  avg: 0.201s
  min: 0.188s
  max: 0.221s

Time to first token:
  avg: 1.912s
  min: 1.876s
  max: 1.942s

Streaming duration:
  avg: 3.331s
  min: 3.301s
  max: 3.384s

Total time:
  avg: 5.258s
  min: 5.189s
  max: 5.326s
```

---

## Understanding the Metrics

**Queue submit latency**
Time required for the initial POST request that places the job in the Space queue.

**Time to first token**
The delay before the first streamed token is received. This is the latency the user feels.

**Streaming duration**
The time from the first token to the final token.

**Total time**
End to end duration from the initial POST to the final event.

These metrics help evaluate responsiveness, cold starts, GPU warm-up behavior, and overall inference performance.

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

Update the Space URL in `hf_benchmark.py`:

```python
SPACE_URL = "https://YOUR-SPACE.hf.space/gradio_api/call/predict"
```

You can modify the request payload to match your model input format.

---


