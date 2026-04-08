#!/usr/bin/env python3
"""
test_server.py — 测试 server HTTP 层
用法：
  1. 先启动 mock server：
       ./build/server --mock --port 8080
  2. 运行测试：
       python3 tests/test_server.py [--port 8080]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:{port}"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = 0


def request(method: str, url: str, body: dict | None = None):
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name: str, cond: bool, detail: str = ""):
    global failures
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures += 1


def run_tests(base: str):
    print(f"\n=== Server: {base} ===\n")

    # ── GET /health ──────────────────────────────────────────────────────────
    print("GET /health")
    status, body = request("GET", f"{base}/health")
    check("status 200", status == 200, str(status))
    check("body.status == 'ok'", body.get("status") == "ok", str(body))

    # ── GET /v1/models ───────────────────────────────────────────────────────
    print("\nGET /v1/models")
    status, body = request("GET", f"{base}/v1/models")
    check("status 200", status == 200, str(status))
    check("object == 'list'", body.get("object") == "list")
    check("data is list", isinstance(body.get("data"), list))
    check("data[0].id exists", bool(body.get("data", [{}])[0].get("id")))

    # ── POST /v1/chat/completions — 正常请求 ─────────────────────────────────
    print("\nPOST /v1/chat/completions (normal)")
    status, body = request("POST", f"{base}/v1/chat/completions", {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 32,
    })
    check("status 200", status == 200, str(status))
    check("object == 'chat.completion'", body.get("object") == "chat.completion")
    check("choices[0].message.role == 'assistant'",
          body.get("choices", [{}])[0].get("message", {}).get("role") == "assistant")
    check("choices[0].message.content is str",
          isinstance(body.get("choices", [{}])[0].get("message", {}).get("content"), str))
    check("choices[0].finish_reason present",
          body.get("choices", [{}])[0].get("finish_reason") in ("stop", "length"))
    check("usage.prompt_tokens > 0",
          body.get("usage", {}).get("prompt_tokens", 0) > 0)
    check("usage.completion_tokens >= 0",
          body.get("usage", {}).get("completion_tokens", -1) >= 0)

    # ── POST — system + user messages ────────────────────────────────────────
    print("\nPOST /v1/chat/completions (system+user)")
    status, body = request("POST", f"{base}/v1/chat/completions", {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "What is 2+2?"},
        ],
    })
    check("status 200", status == 200, str(status))
    check("has choices", len(body.get("choices", [])) == 1)

    # ── POST — 缺少 messages 字段（期望 400） ────────────────────────────────
    print("\nPOST /v1/chat/completions (missing messages → 400)")
    status, body = request("POST", f"{base}/v1/chat/completions", {
        "model": "qwen3",
    })
    check("status 400", status == 400, str(status))
    check("error.type == 'invalid_request_error'",
          body.get("error", {}).get("type") == "invalid_request_error")

    # ── POST — 非法 JSON（期望 400） ─────────────────────────────────────────
    print("\nPOST /v1/chat/completions (invalid JSON → 400)")
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=b"not-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status, body = resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        status, body = e.code, json.loads(e.read())
    check("status 400", status == 400, str(status))

    # ── POST — messages 为空数组（期望成功）──────────────────────────────────
    print("\nPOST /v1/chat/completions (empty messages)")
    status, body = request("POST", f"{base}/v1/chat/completions", {
        "model": "qwen3",
        "messages": [],
    })
    check("status 200", status == 200, str(status))

    # ── 结果汇总 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    if failures == 0:
        print(f"All tests {PASS}")
    else:
        print(f"{FAIL}: {failures} test(s) failed")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    opt = parser.parse_args()
    run_tests(BASE.format(port=opt.port))
    sys.exit(1 if failures else 0)
