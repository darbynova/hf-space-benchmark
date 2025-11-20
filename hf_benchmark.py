#!/usr/bin/env python3
import sys
import json
import time
import statistics
import requests
import sseclient

SPACE_URL = "https://darbynova-demo-model-deployment.hf.space/gradio_api/call/predict"


def run_once(system_prompt, user_prompt):
    payload = {
        "data": [
            system_prompt,
            user_prompt,
            16,
            0.1,
            0.1
        ]
    }

    t_start = time.monotonic()

    response = requests.post(
        SPACE_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    t_after_post = time.monotonic()

    if response.status_code != 200:
        raise RuntimeError("POST error: " + response.text)

    event_id = response.json().get("event_id")
    if not event_id:
        raise RuntimeError("Missing event_id in response")

    stream_url = f"{SPACE_URL}/{event_id}"

    # SSEClient takes a URL and optionally a session
    client = sseclient.SSEClient(stream_url, session=requests.Session())

    first_event_time = None
    last_event_time = None

    # Iterate directly over client, no .events()
    for event in client:
        now = time.monotonic()
        if first_event_time is None:
            first_event_time = now
        last_event_time = now

        # If you want to see the payload, uncomment:
        # if event.data:
        #     print(event.data)

    if first_event_time is None:
        total_time = time.monotonic() - t_start
        return {
            "queue_submit": t_after_post - t_start,
            "first_token": None,
            "stream": None,
            "total": total_time
        }

    queue_submit = t_after_post - t_start
    first_token = first_event_time - t_start
    stream_duration = last_event_time - first_event_time
    total_duration = last_event_time - t_start

    return {
        "queue_submit": queue_submit,
        "first_token": first_token,
        "stream": stream_duration,
        "total": total_duration
    }


def summarize(name, data):
    cleaned = [x for x in data if x is not None]
    if not cleaned:
        return f"{name}: (no data)"
    return (
        f"{name}:\n"
        f"  avg: {statistics.mean(cleaned):.3f}s\n"
        f"  min: {min(cleaned):.3f}s\n"
        f"  max: {max(cleaned):.3f}s"
    )


def main():
    if len(sys.argv) < 4:
        print("Usage: python hf_benchmark.py \"system prompt\" \"user prompt\" N")
        sys.exit(1)

    system_prompt = sys.argv[1]
    user_prompt = sys.argv[2]
    runs = int(sys.argv[3])

    results = []

    print(f"\n=== Running {runs} benchmark iterations ===\n")

    for i in range(1, runs + 1):
        print(f"--- Run {i} ---")
        metrics = run_once(system_prompt, user_prompt)
        results.append(metrics)

        print(f"Queue submit: {metrics['queue_submit']:.3f}s")
        if metrics["first_token"] is None:
            print("First token:  (no SSE events)")
            print(f"Total:        {metrics['total']:.3f}s\n")
        else:
            print(f"Time to first token: {metrics['first_token']:.3f}s")
            print(f"Streaming:           {metrics['stream']:.3f}s")
            print(f"Total:               {metrics['total']:.3f}s\n")

    print("\n=== Benchmark Summary ===\n")

    queue_times = [r["queue_submit"] for r in results]
    first_token_times = [r["first_token"] for r in results]
    stream_times = [r["stream"] for r in results]
    total_times = [r["total"] for r in results]

    print(summarize("Queue submit latency", queue_times))
    print()
    print(summarize("Time to first token", first_token_times))
    print()
    print(summarize("Streaming duration", stream_times))
    print()
    print(summarize("Total time", total_times))
    print()


if __name__ == "__main__":
    main()
