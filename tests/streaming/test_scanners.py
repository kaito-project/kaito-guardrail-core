# Copyright (c) KAITO authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from unittest.mock import Mock

import pytest

from guardrail_core.config import build_scanner, parse_scanner_config
from guardrail_core.scanning import ScanResult
from guardrail_core.streaming import scanners as scanner_module
from guardrail_core.streaming.scanners import ScannerWindowAdapter, StreamingPolicyError
from guardrail_core.streaming.window import (
    StreamingBufferWindow,
    WindowEmitResult,
    WindowScanError,
    WindowScanResult,
)


def adapter(*raw, **kwargs):
    return ScannerWindowAdapter(
        tuple(parse_scanner_config(config) for config in raw), **kwargs
    )


def ban(action="block", substring="secret"):
    return {
        "type": "ban_substrings",
        "substrings": [substring],
        "match_type": "str",
        "action": action,
    }


@pytest.mark.parametrize("action", ["block", "redact"])
@pytest.mark.parametrize("split", range(1, len("secret")))
def test_real_substring_scanner_across_chunks(action, split):
    scanner = adapter(ban(action))
    assert scanner.holdback_len == 5
    window = StreamingBufferWindow(scanner, holdback_len=scanner.holdback_len)
    first = window.feed("safe " + "secret"[:split])
    second = window.feed("secret"[split:] + " end")
    final = window.flush()
    if action == "block":
        assert second == WindowEmitResult(chunks=(), blocked=True)
        assert final.blocked
        assert "safe ".startswith("".join(first.chunks))
    else:
        direct = build_scanner(parse_scanner_config(ban(action))).scan(
            "", "safe secret end"
        )
        assert "".join(first.chunks + second.chunks + final.chunks) == direct.output
        assert window.redacted


@pytest.mark.parametrize("action", ["block", "redact"])
@pytest.mark.parametrize(
    "substring,text", [("\u4f60\u597d", "\u4f60\u597d"), ("secret", "SECRET")]
)
def test_unicode_holdback_and_matches(action, substring, text):
    raw = {**ban(action, substring), "case_sensitive": not substring.isascii()}
    scanner = adapter(raw)
    window = StreamingBufferWindow(scanner, holdback_len=scanner.holdback_len)
    results = [window.feed(char) for char in text] + [window.flush()]
    if action == "block":
        assert any(result.blocked for result in results)
        assert not any(result.chunks for result in results)
    else:
        direct = build_scanner(parse_scanner_config(raw)).scan("", text)
        assert "".join(chunk for result in results for chunk in result.chunks) == (
            direct.output
        )


@pytest.mark.parametrize("action", ["block", "redact"])
def test_real_invisible_text_scanner(action):
    scanner = adapter({"type": "invisible_text", "action": action})
    assert scanner.holdback_len == 0
    result = scanner.scan("hello\u200bworld")
    assert result == (
        WindowScanResult(blocked=True)
        if action == "block"
        else WindowScanResult(sanitized_text="helloworld")
    )


def test_removal_is_applied_before_following_block_scanner():
    scanner = adapter({"type": "invisible_text", "action": "redact"}, ban())
    assert scanner.scan("se\u200bcret") == WindowScanResult(blocked=True)


def test_duplicate_scanner_types_keep_per_scanner_actions():
    scanner = adapter(ban("redact", "secret"), ban("block", "token"))
    assert scanner.scan("secret safe") == WindowScanResult(
        sanitized_text="[REDACTED] safe"
    )
    assert scanner.scan("secret token") == WindowScanResult(blocked=True)


@pytest.mark.parametrize(
    "config",
    [
        {"type": "ban_substrings", "substrings": ["secret"]},
        {**ban(), "contains_all": True},
        ban(substring=" "),
        ban(substring="\u03c2"),
        {"type": "regex", "patterns": ["secret"]},
        {"type": "json"},
        {"type": "reading_time", "max_time": 1},
        {"type": "token_limit", "limit": 10},
        {"type": "secrets"},
        {"type": "sensitive", "detectors": ["email"]},
    ],
)
def test_unsupported_streaming_config_is_rejected_before_construction(
    config, monkeypatch
):
    builder = Mock()
    monkeypatch.setattr(scanner_module, "build_scanner", builder)
    with pytest.raises(StreamingPolicyError, match=r"scanner_configs\[1\]"):
        adapter(ban(), config)
    builder.assert_not_called()


@pytest.mark.parametrize("holdback", [-1, 0, 4])
def test_insufficient_holdback_is_rejected(holdback):
    with pytest.raises(StreamingPolicyError, match="at least 5"):
        adapter(ban(), holdback_len=holdback)


@pytest.mark.parametrize("holdback", [True, False, 1.0, "5"])
def test_invalid_holdback_type_is_rejected(holdback):
    with pytest.raises(TypeError, match="integer"):
        adapter(ban(), holdback_len=holdback)


def test_empty_config_is_rejected():
    with pytest.raises(StreamingPolicyError, match="must not be empty"):
        adapter()


def test_allow_and_empty_redaction_results():
    scanner = adapter({"type": "invisible_text", "action": "redact"})
    assert scanner.scan("") == WindowScanResult()
    assert scanner.scan(" \n") == WindowScanResult()
    assert scanner.scan("safe") == WindowScanResult()
    assert scanner.scan("\u200b") == WindowScanResult(sanitized_text="")


def test_non_idempotent_redaction_is_rejected():
    scanner = adapter(ban("redact", "REDACTED"))
    with pytest.raises(StreamingPolicyError, match="not stable"):
        scanner.scan("REDACTED")


def test_case_sensitive_matching_is_preserved():
    scanner = adapter({**ban(), "case_sensitive": True})
    assert scanner.scan("SECRET") == WindowScanResult()
    assert scanner.scan("secret") == WindowScanResult(blocked=True)


def test_invalid_redaction_without_changes_is_rejected(monkeypatch):
    real = Mock()
    real.scan.return_value = ScanResult("secret", False, 1.0)
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = adapter(ban("redact"))
    with pytest.raises(StreamingPolicyError, match="without sanitizing"):
        scanner.scan("secret")


@pytest.mark.parametrize("cancelled", [False, True])
def test_real_scanner_exceptions_and_prompt_are_preserved(monkeypatch, cancelled):
    cause = asyncio.CancelledError() if cancelled else RuntimeError("scanner failed")
    real = Mock()
    real.scan.side_effect = cause
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = adapter(ban(), prompt="request prompt", holdback_len=5)
    window = StreamingBufferWindow(scanner, holdback_len=scanner.holdback_len)
    with pytest.raises(asyncio.CancelledError if cancelled else WindowScanError) as exc:
        window.feed("secret")
    real.scan.assert_called_once_with("request prompt", "secret")
    if not cancelled:
        assert exc.value.__cause__ is cause
        assert window.blocked


def test_construction_failure_propagates(monkeypatch):
    cause = RuntimeError("construction failed")
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(side_effect=cause))
    with pytest.raises(RuntimeError) as exc:
        adapter(ban())
    assert exc.value is cause
