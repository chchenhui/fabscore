"""Canary-based admission control defense for C2.
Compares poisoned rule set R_gen against a conservative safe rule list R_safe
on a canary set. If R_gen canary PA exceeds R_safe canary PA by more than
delta percentage points, deploy R_gen; otherwise fall back to R_safe.
"""

from typing import Dict, List


R_SAFE: List[str] = [
    "Replace URLs (e.g. http://..., https://..., ftp://...) with <*>.",
    "Replace file paths (e.g. /var/log/syslog, C:\\Users\\...) with <*>.",
    "Replace port numbers (e.g. :8080, :443, port 22) with <*>.",
    "Replace timestamps (e.g. 2024-01-01 12:00:00, Jan 1 00:00:00) with <*>.",
    "Replace IP addresses (e.g. 192.168.1.1, 10.0.0.1, ::1) with <*>.",
    "Replace hex strings (e.g. 0x1A2B3C, 0xDEADBEEF) with <*>.",
]


def format_rules(rules: List[str]) -> str:
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))


def admission_control(
    r_gen_canary_pa: float,
    r_safe_canary_pa: float,
    delta: float = 2.0,
) -> Dict:
    threshold = r_safe_canary_pa + delta / 100.0
    decision = "r_gen" if r_gen_canary_pa > threshold else "r_safe"
    return {
        "decision": decision,
        "r_gen_canary_pa": r_gen_canary_pa,
        "r_safe_canary_pa": r_safe_canary_pa,
        "delta_pp": delta,
        "threshold": threshold,
    }
