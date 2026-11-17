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

"""Apply a guardrail policy to complete output text.

Return a decision to allow the original text, redact it, or block it.
"""

from collections.abc import Sequence
from typing import NamedTuple

from guardrail_core.config import ParsedScannerConfig, SecretsConfig, build_scanner
from guardrail_core.scanning import _redact_secrets_and_verify, scan_output


class PolicyScanResult(NamedTuple):
    """A blocked result has no output; allowed output may be an empty string."""

    output: str | None
    blocked: bool


def scan_policy_output(
    scanner_configs: Sequence[ParsedScannerConfig],
    prompt: str,
    output: str,
    *,
    default_action: str = "block",
) -> PolicyScanResult:
    """Apply configured actions in order with fresh scanners for this output.

    All scanners are built before scanning. Redaction hits must produce changed
    text; Secrets redaction uses the same verification as the streaming adapter.
    Construction and scanning exceptions propagate for the caller to handle.
    """
    if not scanner_configs:
        raise ValueError("scanner_configs must not be empty")
    if default_action not in ("block", "redact"):
        raise ValueError("default_action must be 'block' or 'redact'")
    if not isinstance(output, str):
        raise TypeError("output must be a string")
    scanners = tuple(
        (
            parsed.config,
            build_scanner(parsed, default_action=default_action),
            default_action if parsed.action_on_hit is None else parsed.action_on_hit,
        )
        for parsed in scanner_configs
    )
    text = output
    for config, scanner, action in scanners:
        if action == "redact" and isinstance(config, SecretsConfig):
            sanitized = _redact_secrets_and_verify(scanner, prompt, text)
            if sanitized is None:
                return PolicyScanResult(None, blocked=True)
        else:
            sanitized, validity, _ = scan_output((scanner,), prompt, text)
            if not isinstance(sanitized, str):
                return PolicyScanResult(None, blocked=True)
            if not all(validity.values()) and (action == "block" or sanitized == text):
                return PolicyScanResult(None, blocked=True)
        text = sanitized
    return PolicyScanResult(text, blocked=False)
