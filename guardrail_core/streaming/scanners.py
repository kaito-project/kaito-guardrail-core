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

"""Adapt configured scanners to a streaming window."""

import re
from collections.abc import Sequence
from copy import deepcopy

from guardrail_core.config import (
    SCANNER_REGISTRY,
    BanSubstringsConfig,
    ParsedScannerConfig,
    SecretsConfig,
    build_scanner,
)
from guardrail_core.scanning import Scanner, scan_output
from guardrail_core.streaming.window import WindowScanResult

DEFAULT_STREAMING_HOLDBACK_LEN = 256
STREAMING_SCANNER_CAPABILITIES = {
    "ban_substrings": frozenset({"block", "redact"}),
    "invisible_text": frozenset({"block", "redact"}),
    "secrets": frozenset({"block", "redact"}),
    "sensitive": frozenset({"block", "redact"}),
}


class StreamingPolicyError(ValueError):
    """A configuration is unsupported by the shared streaming adapter."""


class ScannerWindowAdapter:
    """Clean pending text before checking blockers.

    Relative order is preserved within redactors and within blockers. Word
    matching uses window boundary context; Secrets redaction is verified.
    Use holdback_len when constructing StreamingBufferWindow. The retained
    window provides bounded detection, not whole-response scanning.
    """

    def __init__(
        self,
        scanner_configs: Sequence[ParsedScannerConfig],
        *,
        prompt: str = "",
        default_action: str = "block",
        holdback_len: int | None = None,
    ) -> None:
        configs = deepcopy(tuple(scanner_configs))
        if not configs:
            raise StreamingPolicyError("scanner_configs must not be empty")
        if default_action not in ("block", "redact"):
            raise StreamingPolicyError("default_action must be 'block' or 'redact'")
        minimum_holdback = DEFAULT_STREAMING_HOLDBACK_LEN
        for index, parsed in enumerate(configs):
            config = parsed.config
            actions = STREAMING_SCANNER_CAPABILITIES.get(parsed.type)
            if actions is None:
                raise StreamingPolicyError(
                    f"scanner_configs[{index}]: {parsed.type!r} has no supported "
                    "streaming scanner contract"
                )
            if not isinstance(config, SCANNER_REGISTRY[parsed.type]):
                raise StreamingPolicyError(
                    f"scanner_configs[{index}]: configuration does not match "
                    f"scanner type {parsed.type!r}"
                )
            action = (
                default_action if parsed.action_on_hit is None else parsed.action_on_hit
            )
            if action not in actions:
                raise StreamingPolicyError(
                    f"scanner_configs[{index}]: action must be 'block' or 'redact'"
                )
            if isinstance(config, BanSubstringsConfig):
                if config.contains_all:
                    raise StreamingPolicyError(
                        f"scanner_configs[{index}]: streaming ban_substrings requires "
                        "contains_all=False; matches cannot accumulate across windows"
                    )
                minimum_holdback = max(
                    minimum_holdback,
                    max(
                        (
                            len(value)
                            if config.match_type == "word"
                            else len(value) - 1
                            for value in config.substrings
                        ),
                        default=0,
                    ),
                )
            if (
                isinstance(config, SecretsConfig)
                and action == "redact"
                and config.redact_mode != "all"
            ):
                raise StreamingPolicyError(
                    f"scanner_configs[{index}]: streaming Secrets redaction requires "
                    "redact_mode='all'"
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
                config.config,
                build_scanner(config, default_action=default_action),
                default_action
                if config.action_on_hit is None
                else config.action_on_hit,
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
        output = text
        for config, scanner, action in self._scanners:
            if action != "redact":
                continue
            if isinstance(config, SecretsConfig):
                redacted = _redact_secrets_and_verify(scanner, self._prompt, output)
                if redacted is None:
                    return WindowScanResult(blocked=True)
                output = redacted
                continue
            if isinstance(config, BanSubstringsConfig) and config.match_type == "word":
                spans = _find_word_match_spans(
                    output,
                    config.substrings,
                    case_sensitive=config.case_sensitive,
                    preceding_char=preceding_char,
                    flush=flush,
                )
                if spans:
                    output = _redact_match_spans(output, spans)
                continue
            scanned, validity, _ = scan_output(
                (scanner,), self._prompt, output, fail_fast=False
            )
            if not isinstance(scanned, str):
                return WindowScanResult(blocked=True)
            if not all(validity.values()):
                if scanned == output:
                    return WindowScanResult(blocked=True)
                output = scanned

        for config, scanner, action in self._scanners:
            if action != "block":
                continue
            if isinstance(config, BanSubstringsConfig) and config.match_type == "word":
                if _find_word_match_spans(
                    output,
                    config.substrings,
                    case_sensitive=config.case_sensitive,
                    preceding_char=preceding_char,
                    flush=flush,
                ):
                    return WindowScanResult(blocked=True)
                continue
            scanned, validity, _ = scan_output(
                (scanner,), self._prompt, output, fail_fast=False
            )
            if not isinstance(scanned, str) or not all(validity.values()):
                return WindowScanResult(blocked=True)

        return WindowScanResult(sanitized_text=output if output != text else None)


def _find_word_match_spans(
    text: str,
    substrings: list[str],
    *,
    case_sensitive: bool,
    preceding_char: str,
    flush: bool,
) -> tuple[tuple[int, int], ...]:
    """Use real left context and defer unconfirmed right boundaries until EOF."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: list[tuple[int, int]] = []
    for substring in substrings:
        for match in re.finditer(re.escape(substring), text, flags):
            start, end = match.span()
            left_char = text[start - 1] if start > 0 else preceding_char
            if _is_word_char(left_char) == _is_word_char(text[start]):
                continue
            if end == len(text) and not flush:
                continue
            right_char = text[end] if end < len(text) else ""
            if _is_word_char(text[end - 1]) == _is_word_char(right_char):
                continue
            spans.append((start, end))
    return _merge_match_spans(spans)


def _is_word_char(value: str) -> bool:
    return bool(value and re.fullmatch(r"\w", value))


def _merge_match_spans(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _redact_match_spans(text: str, spans: tuple[tuple[int, int], ...]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.extend((text[cursor:start], "[REDACTED]"))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _redact_secrets_and_verify(scanner: Scanner, prompt: str, text: str) -> str | None:
    sanitized, validity, _ = scan_output((scanner,), prompt, text, fail_fast=False)
    if not isinstance(sanitized, str):
        return None
    if all(validity.values()):
        return text if sanitized == text else None
    if sanitized == text or len(sanitized) > len(text):
        return None
    verified, verified_validity, _ = scan_output(
        (scanner,), prompt, sanitized, fail_fast=False
    )
    if (
        not isinstance(verified, str)
        or verified != sanitized
        or not all(verified_validity.values())
    ):
        return None
    return sanitized
