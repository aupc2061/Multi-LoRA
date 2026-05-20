from __future__ import annotations

SD35_BASELINE_METHODS = ("merge", "switch")
SD35_NULLMASK_METHODS = ("sd35_mask_only", "sd35_mask_nullproj")
SD35_SUPPORTED_METHODS = SD35_BASELINE_METHODS + SD35_NULLMASK_METHODS


def validate_methods(methods: list[str]) -> list[str]:
    invalid = [method for method in methods if method not in SD35_SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported SD3.5 null-mask methods: {', '.join(invalid)}")
    return methods
