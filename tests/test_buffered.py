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

from dataclasses import replace
from unittest.mock import Mock, call

import pytest

from guardrail_core import buffered
from guardrail_core.buffered import PolicyScanResult, scan_policy_output
from guardrail_core.config import parse_scanner_config
from guardrail_core.scanning import ScanResult


def configs(*raw):
    return [parse_scanner_config(value) for value in raw]


def ban(action=None, text="secret"):
    raw = {"type": "ban_substrings", "substrings": [text]}
    if action is not None:
        raw["action"] = action
    return raw


@pytest.mark.parametrize("text", ["safe", "", " \n"])
def test_pass_preserves_complete_text(text):
    assert scan_policy_output(configs(ban()), "prompt", text) == PolicyScanResult(
        text, blocked=False
    )


def test_block_discards_output():
    assert scan_policy_output(configs(ban()), "", "a secret") == PolicyScanResult(
        None, blocked=True
    )


def test_redaction_uses_default_action_and_overrides():
    parsed = [
        replace(parse_scanner_config(ban()), action_on_hit=None),
        parse_scanner_config(ban("block", "forbidden")),
    ]
    assert scan_policy_output(
        parsed, "", "a secret", default_action="redact"
    ) == PolicyScanResult("a [REDACTED]", blocked=False)
    assert scan_policy_output(
        parsed, "", "forbidden", default_action="redact"
    ) == PolicyScanResult(None, blocked=True)


def test_block_before_redact_preserves_configured_order():
    assert scan_policy_output(
        configs(ban("block"), ban("redact")), "", "a secret"
    ).blocked
    assert scan_policy_output(
        configs(ban("redact"), ban("block")), "", "a secret"
    ) == PolicyScanResult("a [REDACTED]", blocked=False)


def test_duplicate_scanner_types_cannot_overwrite_a_block():
    assert scan_policy_output(
        configs(ban("block"), ban("block", "other")), "", "secret"
    ).blocked


def test_empty_redaction_is_allowed():
    assert scan_policy_output(
        configs({"type": "invisible_text", "action": "redact"}), "", "\u200b"
    ) == PolicyScanResult("", blocked=False)


@pytest.mark.parametrize(
    "raw,text",
    [
        ({"type": "regex", "patterns": ["secret"], "action": "redact"}, "secret"),
        ({"type": "json", "repair": True, "action": "redact"}, '{"a":1}'),
        ({"type": "reading_time", "max_time": 1}, "short"),
        ({"type": "token_limit", "limit": 10}, "short"),
        ({"type": "sensitive", "detectors": ["email"]}, "short"),
        (
            {"type": "ban_substrings", "substrings": ["a", "b"], "contains_all": True},
            "a",
        ),
    ],
)
def test_complete_scans_are_not_limited_to_streaming_capabilities(raw, text):
    assert not scan_policy_output(configs(raw), "", text).blocked


def test_real_secrets_redaction_is_verified():
    text = 'key = "' + "AKIA" + "Z" * 16 + '"'
    assert scan_policy_output(
        configs({"type": "secrets", "action": "redact"}), "", text
    ) == PolicyScanResult('key = "******"', blocked=False)


@pytest.mark.parametrize(
    "results",
    [
        [ScanResult("unchanged", False, 1.0)],
        [ScanResult("longer than input", False, 1.0)],
        [ScanResult("changed", True, -1.0)],
        [ScanResult("clean", False, 1.0), ScanResult("clean", False, 1.0)],
        [ScanResult("clean", False, 1.0), ScanResult("other", True, -1.0)],
    ],
)
def test_secrets_uses_shared_fail_closed_verification(monkeypatch, results):
    scanner = Mock(scan=Mock(side_effect=results))
    monkeypatch.setattr(buffered, "build_scanner", Mock(return_value=scanner))
    assert scan_policy_output(
        configs({"type": "secrets", "action": "redact"}), "", "unchanged"
    ) == PolicyScanResult(None, blocked=True)


@pytest.mark.parametrize("action", ["block", "redact"])
@pytest.mark.parametrize("output", [None, 42])
def test_malformed_scanner_output_blocks(monkeypatch, action, output):
    scanner = Mock(scan=Mock(return_value=ScanResult(output, True, -1.0)))
    monkeypatch.setattr(buffered, "build_scanner", Mock(return_value=scanner))
    assert scan_policy_output(configs(ban(action)), "", "text").blocked


def test_failed_redaction_without_changes_blocks(monkeypatch):
    scanner = Mock(scan=Mock(return_value=ScanResult("text", False, 1.0)))
    monkeypatch.setattr(buffered, "build_scanner", Mock(return_value=scanner))
    assert scan_policy_output(configs(ban("redact")), "", "text").blocked


def test_fresh_scanners_per_call_and_prompt_forwarding(monkeypatch):
    first = Mock(scan=Mock(return_value=ScanResult("first", True, -1.0)))
    second = Mock(scan=Mock(return_value=ScanResult("second", True, -1.0)))
    builder = Mock(side_effect=[first, second])
    monkeypatch.setattr(buffered, "build_scanner", builder)
    parsed = configs(ban())
    assert scan_policy_output(parsed, "prompt one", "first").output == "first"
    assert scan_policy_output(parsed, "prompt two", "second").output == "second"
    first.scan.assert_called_once_with("prompt one", "first")
    second.scan.assert_called_once_with("prompt two", "second")
    assert builder.call_args_list == [
        call(parsed[0], default_action="block"),
        call(parsed[0], default_action="block"),
    ]


@pytest.mark.parametrize("build_failure", [False, True])
def test_exceptions_propagate(monkeypatch, build_failure):
    failure = RuntimeError("scanner failed")
    builder = Mock(
        side_effect=failure if build_failure else None,
        return_value=Mock(scan=Mock(side_effect=failure)),
    )
    monkeypatch.setattr(buffered, "build_scanner", builder)
    with pytest.raises(RuntimeError) as exc:
        scan_policy_output(configs(ban()), "", "text")
    assert exc.value is failure


def test_later_construction_failure_is_not_hidden_by_earlier_block(monkeypatch):
    scanner = Mock(scan=Mock(return_value=ScanResult("secret", False, 1.0)))
    monkeypatch.setattr(
        buffered,
        "build_scanner",
        Mock(side_effect=[scanner, RuntimeError("construction failed")]),
    )
    with pytest.raises(RuntimeError, match="construction failed"):
        scan_policy_output(configs(ban(), ban()), "", "secret")
    scanner.scan.assert_not_called()


@pytest.mark.parametrize("action", ["", "ignore", None])
def test_invalid_default_action_fails_before_building(monkeypatch, action):
    builder = Mock()
    monkeypatch.setattr(buffered, "build_scanner", builder)
    with pytest.raises(ValueError, match="default_action"):
        scan_policy_output(configs(ban()), "", "text", default_action=action)
    builder.assert_not_called()


def test_empty_scanner_configuration_is_rejected():
    with pytest.raises(ValueError, match="scanner_configs"):
        scan_policy_output([], "", "text")


def test_non_string_output_is_rejected():
    with pytest.raises(TypeError, match="output"):
        scan_policy_output(configs(ban()), "", None)
