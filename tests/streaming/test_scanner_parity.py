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

"""Scanner/window regression scenarios recorded at kaito main dae02fe4d402.

Transport framing, refusal responses, and stream lifecycle remain consumer tests.
"""

from unittest.mock import Mock

import pytest

from guardrail_core.config import (
    BanSubstringsConfig,
    ParsedScannerConfig,
    parse_scanner_config,
)
from guardrail_core.scanning import ScanResult
from guardrail_core.streaming import scanners as scanner_module
from guardrail_core.streaming.scanners import ScannerWindowAdapter, StreamingPolicyError
from guardrail_core.streaming.window import StreamingBufferWindow, WindowScanResult


def ban(action="redact", substrings=("unsafe",), match_type="word", **kwargs):
    return {
        "type": "ban_substrings",
        "action": action,
        "substrings": list(substrings),
        "match_type": match_type,
        **kwargs,
    }


def make_adapter(*configs, **kwargs):
    return ScannerWindowAdapter(
        [parse_scanner_config(config) for config in configs], **kwargs
    )


def run_window(contents, *configs):
    scanner = make_adapter(*configs)
    window = StreamingBufferWindow(scanner, holdback_len=scanner.holdback_len)
    results = [window.feed(content) for content in contents]
    results.append(window.flush())
    return "".join(chunk for result in results for chunk in result.chunks), window


@pytest.mark.parametrize("action", ["block", "redact"])
def test_word_matching_uses_real_boundaries_and_final_flush(action):
    scanner = make_adapter(ban(action))
    assert scanner.scan("unsafe") == WindowScanResult()
    assert scanner.scan("unsafeish", flush=True) == WindowScanResult()
    assert scanner.scan("unsafe ", preceding_char="x") == WindowScanResult()
    assert scanner.scan("unsafe ", preceding_char="_") == WindowScanResult()
    assert scanner.scan("unsafe ", preceding_char="\u4f60") == WindowScanResult()
    expected = (
        WindowScanResult(blocked=True)
        if action == "block"
        else WindowScanResult(sanitized_text="[REDACTED]")
    )
    assert scanner.scan("unsafe", flush=True) == expected


@pytest.mark.parametrize("action", ["block", "redact"])
def test_word_match_waits_for_lookahead(action):
    prefix = "p" * 300
    text, window = run_window([prefix + " unsafe", "ish"], ban(action))
    assert text == prefix + " unsafeish"
    assert not window.blocked


@pytest.mark.parametrize("suffix", [" suffix", ""])
def test_word_redaction_at_right_boundary_or_eof(suffix):
    prefix = "p" * 300
    text, window = run_window([prefix + " unsafe", suffix], ban())
    assert text == prefix + " [REDACTED]" + suffix
    assert window.redacted


@pytest.mark.parametrize("action", ["block", "redact"])
def test_word_match_preserves_left_boundary_after_window_slides(action):
    prefix = "p" * 300
    text, window = run_window([prefix + "xunsafe "] + ["q"] * 300, ban(action))
    assert text == prefix + "xunsafe " + "q" * 300
    assert not window.blocked


@pytest.mark.parametrize("action", ["block", "redact"])
@pytest.mark.parametrize("suffix,matches", [(" suffix", False), ("x suffix", True)])
def test_word_match_with_nonword_suffix(action, suffix, matches):
    prefix = "p" * 300
    text, window = run_window([prefix + " unsafe-", suffix], ban(action, ("unsafe-",)))
    if not matches:
        assert text == prefix + " unsafe-" + suffix
        assert not window.blocked
    elif action == "block":
        assert window.blocked
        assert "unsafe-" not in text
    else:
        assert text == prefix + " [REDACTED]" + suffix


@pytest.mark.parametrize(
    "substrings,text,expected",
    [
        (("safe", "safety"), "safety", "[REDACTED]"),
        (("a.a", "a"), "a.a", "[REDACTED]"),
        (("x",), " x tail", " [REDACTED] tail"),
        (("X",), " x tail", " [REDACTED] tail"),
        ((".",), " a.b ", " a[REDACTED]b "),
        (
            ("unsafe", "prohibited"),
            "unsafe and prohibited",
            "[REDACTED] and [REDACTED]",
        ),
    ],
)
def test_word_redaction_merges_overlaps_and_preserves_adjacent_text(
    substrings, text, expected
):
    assert run_window([text], ban(substrings=substrings))[0] == expected


@pytest.mark.parametrize("case_sensitive", [False, True])
def test_word_case_sensitivity(case_sensitive):
    text, _ = run_window(["UNSAFE unsafe"], ban(case_sensitive=case_sensitive))
    assert text == ("UNSAFE [REDACTED]" if case_sensitive else "[REDACTED] [REDACTED]")


@pytest.mark.parametrize("match_type,holdback", [("word", 300), ("str", 299)])
@pytest.mark.parametrize("action", ["block", "redact"])
def test_long_substring_expands_default_holdback(match_type, holdback, action):
    raw = ban(action, ("x" * 300,), match_type)
    scanner = make_adapter(raw)
    assert scanner.holdback_len == holdback
    text, window = run_window(["x" * 270, "x" * 30, " suffix"], raw)
    if action == "block":
        assert window.blocked
        assert not text
    else:
        assert text == "[REDACTED] suffix"


@pytest.mark.parametrize("reverse", [False, True])
def test_all_redactors_run_before_blockers_regardless_of_configuration_order(reverse):
    configs = [
        {"type": "invisible_text", "action": "redact"},
        ban("block", match_type="str"),
    ]
    if reverse:
        configs.reverse()
    assert make_adapter(*configs).scan("un\u200bsafe").blocked


@pytest.mark.parametrize("reverse", [False, True])
def test_blocker_sees_redaction_replacement(reverse):
    configs = [ban(), ban("block", ("[REDACTED]",), "str")]
    if reverse:
        configs.reverse()
    assert make_adapter(*configs).scan("unsafe", flush=True).blocked


def test_redaction_runs_before_an_earlier_matching_blocker():
    scanner = make_adapter(ban("block", match_type="str"), ban(match_type="str"))
    assert scanner.scan("unsafe") == WindowScanResult(sanitized_text="[REDACTED]")


@pytest.mark.parametrize(
    "config",
    [
        {"type": "ban_substrings", "substrings": ["unsafe"]},
        {"type": "invisible_text"},
        {"type": "secrets"},
        {"type": "sensitive"},
    ],
)
def test_default_holdback_and_explicit_larger_window(config):
    assert make_adapter(config).holdback_len == 256
    assert make_adapter(config, holdback_len=512).holdback_len == 512
    with pytest.raises(StreamingPolicyError, match="at least 256"):
        make_adapter(config, holdback_len=255)


@pytest.mark.parametrize("match_type,minimum", [("word", 300), ("str", 299)])
def test_explicit_holdback_cannot_undercut_long_substring(match_type, minimum):
    config = ban(substrings=("x" * 300,), match_type=match_type)
    with pytest.raises(StreamingPolicyError, match=f"at least {minimum}"):
        make_adapter(config, holdback_len=minimum - 1)
    assert make_adapter(config, holdback_len=minimum).holdback_len == minimum


@pytest.mark.parametrize("invalid_default", ["mask", ""])
def test_invalid_default_action_fails_before_building(monkeypatch, invalid_default):
    builder = Mock()
    monkeypatch.setattr(scanner_module, "build_scanner", builder)
    with pytest.raises(StreamingPolicyError, match="default_action"):
        make_adapter(ban(), default_action=invalid_default)
    builder.assert_not_called()


@pytest.mark.parametrize("action", ["mask", ""])
def test_invalid_action_override_fails_before_building(monkeypatch, action):
    builder = Mock()
    monkeypatch.setattr(scanner_module, "build_scanner", builder)
    config = ParsedScannerConfig(
        "ban_substrings", BanSubstringsConfig(["unsafe"]), action_on_hit=action
    )
    with pytest.raises(StreamingPolicyError, match=r"scanner_configs\[0\].*action"):
        ScannerWindowAdapter([config])
    builder.assert_not_called()


def test_mismatched_typed_config_is_rejected(monkeypatch):
    builder = Mock()
    monkeypatch.setattr(scanner_module, "build_scanner", builder)
    with pytest.raises(StreamingPolicyError, match="does not match"):
        ScannerWindowAdapter(
            [ParsedScannerConfig("secrets", BanSubstringsConfig(["unsafe"]))]
        )
    builder.assert_not_called()


@pytest.mark.parametrize("action", ["block", "redact"])
def test_default_action_is_used_for_configs_without_an_override(action):
    scanner = ScannerWindowAdapter(
        [ParsedScannerConfig("ban_substrings", BanSubstringsConfig(["unsafe"]))],
        default_action=action,
    )
    result = scanner.scan("unsafe", flush=True)
    assert result == (
        WindowScanResult(blocked=True)
        if action == "block"
        else WindowScanResult(sanitized_text="[REDACTED]")
    )


@pytest.mark.parametrize("mode", ["partial", "hash"])
def test_non_all_secrets_redaction_is_rejected_before_construction(monkeypatch, mode):
    builder = Mock()
    monkeypatch.setattr(scanner_module, "build_scanner", builder)
    with pytest.raises(StreamingPolicyError, match="redact_mode"):
        make_adapter({"type": "secrets", "action": "redact", "redact_mode": mode})
    builder.assert_not_called()
    make_adapter({"type": "secrets", "action": "block", "redact_mode": mode})


@pytest.mark.parametrize("action", ["block", "redact"])
def test_secrets_split_across_release_boundary(action):
    prefix = "a" * 250
    text, window = run_window(
        [prefix + " AWS key: AKIA1234", "567890ABCDEF"],
        {"type": "secrets", "action": action},
    )
    assert "AKIA" not in text
    if action == "block":
        assert window.blocked
    else:
        assert text == prefix + " AWS key: ******"


@pytest.mark.parametrize(
    "contents,expected",
    [
        (["AWS key: AKIA1234", "567890ABCDEF"], "AWS key: ******"),
        (
            [
                "a" * 300 + " first AKIA1234567890ABCDEF",
                "b" * 300 + " second AKIAZYXWVUTSRQPONMLK",
            ],
            "a" * 300 + " first ******" + "b" * 300 + " second ******",
        ),
        (
            ["Keys: AKIA1234567890ABCDEF, " + "AIza" + "A" * 35],
            "Keys: ******, ******",
        ),
        (
            ["Keys: " + ", ".join(["AKIA1234567890ABCDEF"] * 20)],
            "Keys: " + ", ".join(["******"] * 20),
        ),
        (
            ["a" * 300 + " token = AKIA1234567890ABCDEF", " tail"],
            "a" * 300 + " token = ****** tail",
        ),
    ],
)
def test_secrets_all_redaction_and_window_rescanning(contents, expected):
    text, window = run_window(contents, {"type": "secrets", "action": "redact"})
    assert text == expected
    assert not window.blocked
    assert window.redacted


@pytest.mark.parametrize(
    "results",
    [
        [ScanResult("secret", False, 1.0)],
        [Mock(output=None, valid=False, score=1.0)],
        [ScanResult("secret unexpectedly expanded", False, 1.0)],
        [ScanResult("mutation", True, -1.0)],
        [ScanResult("******", False, 1.0), ScanResult("******", False, 1.0)],
        [ScanResult("******", False, 1.0), ScanResult("changed", True, -1.0)],
        [ScanResult("******", False, 1.0), Mock(output=None, valid=True, score=-1.0)],
    ],
)
def test_secrets_verification_fails_closed(monkeypatch, results):
    real = Mock()
    real.scan.side_effect = results
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = make_adapter({"type": "secrets", "action": "redact"})
    assert scanner.scan("secret") == WindowScanResult(blocked=True)


def test_secrets_redaction_scans_once_then_verifies(monkeypatch):
    real = Mock()
    real.scan.side_effect = [
        ScanResult("******", False, 1.0),
        ScanResult("******", True, -1.0),
    ]
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = make_adapter({"type": "secrets", "action": "redact"}, prompt="prompt")
    assert scanner.scan("secret") == WindowScanResult(sanitized_text="******")
    assert [call.args for call in real.scan.call_args_list] == [
        ("prompt", "secret"),
        ("prompt", "******"),
    ]


@pytest.mark.parametrize("action", ["block", "redact"])
def test_malformed_nonword_scanner_output_blocks_release(monkeypatch, action):
    real = Mock()
    real.scan.return_value = Mock(output=None, valid=True, score=-1.0)
    monkeypatch.setattr(scanner_module, "build_scanner", Mock(return_value=real))
    scanner = make_adapter({"type": "invisible_text", "action": action})
    assert scanner.scan("text") == WindowScanResult(blocked=True)


@pytest.mark.parametrize(
    "detector,pii,replacement,safe_text",
    [
        ("email", "alice@example.com", "<EMAIL>", "alice at example dot com"),
        ("phone", "+1 (206) 555-0100", "<PHONE>", "extension 42"),
        ("credit_card", "4111 1111 1111 1111", "<CREDIT_CARD>", "order 123456789012"),
        ("ip_address", "10.0.0.1", "<IP_ADDRESS>", "version 10.0.0"),
    ],
)
@pytest.mark.parametrize("action", ["block", "redact"])
def test_sensitive_detectors_and_safe_cases(
    detector, pii, replacement, safe_text, action
):
    raw = {"type": "sensitive", "action": action, "detectors": [detector]}
    text, window = run_window(["Value: " + pii], raw)
    if action == "block":
        assert window.blocked
        assert not text
    else:
        assert text == "Value: " + replacement
    safe, safe_window = run_window(["Value: " + safe_text], raw)
    assert safe == "Value: " + safe_text
    assert not safe_window.blocked


@pytest.mark.parametrize("prefix", ["", "a" * 250])
def test_sensitive_split_match_at_release_boundary(prefix):
    text, window = run_window(
        [prefix + " Contact alice@", "example.com now"],
        {"type": "sensitive", "action": "redact", "detectors": ["email"]},
    )
    assert text == prefix + " Contact <EMAIL> now"
    assert window.redacted


def test_sensitive_multiple_pii_and_block_after_redaction():
    raw = {"type": "sensitive", "action": "redact"}
    text, _ = run_window(
        [
            "Email alice@example.com, call +1 (206) 555-0100, "
            "use 4111 1111 1111 1111 from 10.0.0.1"
        ],
        raw,
    )
    assert text == "Email <EMAIL>, call <PHONE>, use <CREDIT_CARD> from <IP_ADDRESS>"
    assert (
        make_adapter(ban("block", ("<EMAIL>",), "str"), raw)
        .scan("Contact alice@example.com")
        .blocked
    )
