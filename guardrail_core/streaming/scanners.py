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

"""Adapt bounded, context-independent configured scanners to a streaming window."""

from collections.abc import Sequence

from guardrail_core.config import (
    BanSubstringsConfig,
    InvisibleTextConfig,
    ParsedScannerConfig,
    build_scanner,
)
from guardrail_core.scanning import scan_output
from guardrail_core.streaming.window import WindowScanResult


class StreamingPolicyError(ValueError):
    """A configuration cannot preserve scanner semantics in a finite window."""


class ScannerWindowAdapter:
    """Run real scanners in order, applying each scanner's configured action.

    Only substring matching without contains_all and character-local invisible
    text scanning are supported. Other scanners need a richer streaming contract.
    Use holdback_len when constructing StreamingBufferWindow.
    """

    def __init__(
        self,
        scanner_configs: Sequence[ParsedScannerConfig],
        *,
        prompt: str = "",
        default_action: str = "block",
        holdback_len: int | None = None,
    ) -> None:
        configs = tuple(scanner_configs)
        if not configs:
            raise StreamingPolicyError("scanner_configs must not be empty")
        minimum_holdback = 0
        for index, parsed in enumerate(configs):
            config = parsed.config
            if isinstance(config, BanSubstringsConfig):
                if config.match_type != "str" or config.contains_all:
                    raise StreamingPolicyError(
                        f"scanner_configs[{index}]: streaming ban_substrings "
                        "requires match_type='str' and contains_all=False"
                    )
                if any(not value.strip() for value in config.substrings):
                    raise StreamingPolicyError(
                        f"scanner_configs[{index}]: whitespace-only substrings "
                        "cannot preserve scan_output's whitespace handling"
                    )
                if not config.case_sensitive and any(
                    not value.isascii() for value in config.substrings
                ):
                    raise StreamingPolicyError(
                        f"scanner_configs[{index}]: non-ASCII substrings require "
                        "case_sensitive=True for context-independent matching"
                    )
                lengths = [
                    len(value if config.case_sensitive else value.lower())
                    for value in config.substrings
                ]
                minimum_holdback = max(minimum_holdback, max(lengths) - 1)
            elif not isinstance(config, InvisibleTextConfig):
                raise StreamingPolicyError(
                    f"scanner_configs[{index}]: {parsed.type!r} has no bounded "
                    "streaming scanner contract"
                )
        if holdback_len is None:
            holdback_len = minimum_holdback
        if isinstance(holdback_len, bool) or not isinstance(holdback_len, int):
            raise TypeError("holdback_len must be an integer")
        if holdback_len < minimum_holdback:
            raise StreamingPolicyError(
                f"holdback_len must be at least {minimum_holdback} Unicode code points"
            )
        self.holdback_len = holdback_len
        self._prompt = prompt
        self._scanners = tuple(
            (
                build_scanner(config, default_action=default_action),
                config.action_on_hit or default_action,
            )
            for config in configs
        )

    def scan(
        self,
        text: str,
        *,
        flush: bool = False,
        preceding_char: str = "",
    ) -> WindowScanResult:
        del flush, preceding_char
        output = text
        for scanner, action in self._scanners:
            scanned, validity, _ = scan_output((scanner,), self._prompt, output)
            # scan_output intentionally skips whitespace-only output.
            if validity and not all(validity.values()):
                if action == "block":
                    return WindowScanResult(blocked=True)
                if scanned == output:
                    raise StreamingPolicyError(
                        "redact scanner reported a finding without sanitizing output"
                    )
            if scanned is None:
                raise StreamingPolicyError("scanner returned no output")
            output = scanned
        # Retained sanitized tails are scanned again; they must be a fixed point.
        if output != text:
            for scanner, action in self._scanners:
                rescanned, validity, _ = scan_output((scanner,), self._prompt, output)
                if validity and not all(validity.values()) and action == "block":
                    return WindowScanResult(blocked=True)
                if rescanned != output or (validity and not all(validity.values())):
                    raise StreamingPolicyError(
                        "scanner redaction is not stable under repeated window scans"
                    )
        return WindowScanResult(sanitized_text=output if output != text else None)
