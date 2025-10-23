import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{BASE_URL}/predict"

NUM_REQUESTS = 50
MAX_WORKERS = 20
TIMEOUT = 10

TEST_INPUT = "This is a great day for testing concurrency and stability."

def log_result(thread_id: int, status: str, response_data: Dict[str, Any], elapsed_time: float):
    status_code = response_data.get('status_code', 'N/A')
    label = response_data.get('label', response_data.get('error', 'N/A'))

    print(f"[Thread {thread_id:02d}] {status:<10} | HTTP:{status_code} | Time:{elapsed_time:.4f}s | Result: {label}")

def send_request(thread_id: int, api_url: str, text: str):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    start_time = time.time()

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=TIMEOUT)
        elapsed_time = time.time() - start_time

        try:
            data = response.json()
        except json.JSONDecodeError:
            data = {"error": "Invalid JSON response", "raw_response": response.text}

        if response.status_code == 200:
            log_result(thread_id, "SUCCESS", data, elapsed_time)
            return "success"
        else:
            data['status_code'] = response.status_code
            log_result(thread_id, "FAIL_HTTP", data, elapsed_time)
            return "fail"

    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        data = {"error": str(e), "status_code": "EXCEPTION"}
        log_result(thread_id, "ERROR_CONN", data, elapsed_time)
        return "error"

if __name__ == '__main__':
    print("--- Starting Concurrent API Test ---")
    print(f"Target URL: {API_ENDPOINT}")
    print(f"Total Requests: {NUM_REQUESTS}, Max Workers: {MAX_WORKERS}\n")

    results_map = {"success": 0, "fail": 0, "error": 0}

    overall_start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(send_request, i + 1, API_ENDPOINT, TEST_INPUT)
            for i in range(NUM_REQUESTS)
        ]

        for future in as_completed(futures):
            try:
                result_type = future.result()
                results_map[result_type] += 1
            except Exception as e:
                print(f"[CRITICAL] Worker failed unexpectedly: {e}")
                results_map["error"] += 1

    overall_elapsed_time = time.time() - overall_start_time

    print("\n--- Summary ---")
    print(f"Total time for all requests: {overall_elapsed_time:.4f}s")
    print(f"Requests sent: {NUM_REQUESTS}")
    print(f"Successful: {results_map['success']}")
    print(f"Failed (HTTP): {results_map['fail']}")
    print(f"Errors (Conn/Code): {results_map['error']}")
    print("-----------------")
