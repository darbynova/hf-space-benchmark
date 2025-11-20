#!/usr/bin/env python3
import sys
import json
import time
import statistics
import requests
import sseclient

SPACE_URL = "https://YOUR-SPACE.hf.space/gradio_api/call/predict"


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
    stream_resp = requests.get(stream_url, stream=True)
    client = sseclient.SSEClient(stream_resp)

    first_event_time = None
    last_event_time = None

    for event in client.events():
        now = time.monotonic()
        if first_event_time is None:
            first_event_time = now
        last_event_time = now

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
    if not
