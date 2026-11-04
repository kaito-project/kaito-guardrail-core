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

from guardrail_core.config import (
    BanSubstringsConfig,
    SensitiveConfig,
    build_scanner,
    parse_scanner_config,
)
from guardrail_core.native_scanners import BanSubstringsMatchType
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
    assert scanner.holdback_len == 256
    window = StreamingBufferWindow(scanner, holdback_len=scanner.holdback_len)
    first = window.feed("safe " + "secret"[:split])
    second = window.feed("secret"[split:] + " end")
    final = window.flush()
    if action == "block":
        assert second == WindowEmitResult(chunks=())
        assert final == WindowEmitResult(chunks=(), blocked=True)
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
    assert scanner.holdback_len == 256
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


@pytest.mark.parametrize("action", ["block", "redact"])
def test_original_config_mutation_does_not_change_scanning_or_holdback(action):
    substring = "x" * 300
    parsed = parse_scanner_config({**ban(action, substring), "match_type": "word"})
    config = parsed.config
    assert isinstance(config, BanSubstringsConfig)
    configs = [parsed]
    scanner = ScannerWindowAdapter(configs)

    config.substrings[:] = ["changed", "y" * 400]
    config.case_sensitive = True
    parsed.action_on_hit = "redact" if action == "block" else "block"
    parsed.config = SensitiveConfig(detectors=["email"])
    configs.clear()

    assert scanner.holdback_len == 300
    assert scanner.scan("X" * 300, flush=True) == (
        WindowScanResult(blocked=True)
        if action == "block"
        else WindowScanResult(sanitized_text="[REDACTED]")
    )
    assert scanner.scan("changed", flush=True) == WindowScanResult()
    assert scanner.scan("y" * 400, flush=True) == WindowScanResult()


def test_original_match_type_mutation_does_not_change_word_boundaries():
    parsed = parse_scanner_config({**ban(), "match_type": "word"})
    assert isinstance(parsed.config, BanSubstringsConfig)
    scanner = ScannerWindowAdapter([parsed])
    parsed.config.match_type = BanSubstringsMatchType.STR

    assert scanner.scan("secret") == WindowScanResult()
    assert scanner.scan("secretish", flush=True) == WindowScanResult()
    assert scanner.scan("secret", flush=True) == WindowScanResult(blocked=True)


@pytest.mark.parametrize(
    "raw",
    [
        ban(),
        {"type": "invisible_text"},
        {"type": "secrets"},
        {"type": "sensitive", "detectors": ["email"]},
    ],
)
def test_scanner_construction_receives_independent_config_snapshot(raw, monkeypatch):
    builder = Mock(wraps=build_scanner)
    monkeypatch.setattr(scanner_module, "build_scanner", builder)
    original = parse_scanner_config(raw)
    ScannerWindowAdapter([original])
    snapshot = builder.call_args.args[0]

    assert snapshot == original
    assert snapshot is not original
    assert snapshot.config is not original.config
    if isinstance(original.config, BanSubstringsConfig):
        assert snapshot.config.substrings is not original.config.substrings
        original.config.substrings.append("changed")
        assert snapshot.config.substrings == ["secret"]
    elif isinstance(original.config, SensitiveConfig):
        assert snapshot.config.detectors is not original.config.detectors
        original.config.detectors.append("phone")
        assert snapshot.config.detectors == ["email"]


@pytest.mark.parametrize(
    "config",
    [
        {**ban(), "contains_all": True},
        {"type": "regex", "patterns": ["secret"]},
        {"type": "json"},
        {"type": "reading_time", "max_time": 1},
        {"type": "token_limit", "limit": 10},
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


@pytest.mark.parametrize("holdback", [-1, 0, 4, 255])
def test_insufficient_holdback_is_rejected(holdback):
    with pytest.raises(StreamingPolicyError, match="at least 256"):
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


def test_case_sensitive_matching_is_preserved():
    scanner = adapter({**ban(), "case_sensitive": True})
    assert scanner.scan("SECRET") == WindowScanResult()
    assert scanner.scan("secret") == WindowScanResult(blocked=True)


def test_invalid_redaction_without_changes_is_rejected(monkeypatch):
    real = Mock()
    real.scan.return_value = ScanResult("secret", False, 1.0)
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = adapter(ban("redact"))
    assert scanner.scan("secret") == WindowScanResult(blocked=True)


@pytest.mark.parametrize("cancelled", [False, True])
def test_real_scanner_exceptions_and_prompt_are_preserved(monkeypatch, cancelled):
    cause = asyncio.CancelledError() if cancelled else RuntimeError("scanner failed")
    real = Mock()
    real.scan.side_effect = cause
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = adapter(ban(), prompt="request prompt", holdback_len=256)
    window = StreamingBufferWindow(scanner, holdback_len=scanner.holdback_len)
    with pytest.raises(asyncio.CancelledError if cancelled else WindowScanError) as exc:
        window.feed("secret" + " " * 256)
    real.scan.assert_called_once_with("request prompt", "secret" + " " * 256)
    if not cancelled:
        assert exc.value.__cause__ is cause
        assert window.blocked


def test_construction_failure_propagates(monkeypatch):
    cause = RuntimeError("construction failed")
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(side_effect=cause))
    with pytest.raises(RuntimeError) as exc:
        adapter(ban())
    assert exc.value is cause
