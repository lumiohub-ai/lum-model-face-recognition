"""Per-camera ChArUco board specification and validation."""

from typing import Any, Mapping

DEFAULT_CHARUCO_BOARD_SPEC = {
    "squares_x": 5,
    "squares_y": 7,
    "square_length_m": 0.04,
    "marker_length_m": 0.02,
    "dictionary": "DICT_4X4_50",
}

# Marker capacities for the stable dictionary keys exposed by the dashboard.
CHARUCO_DICTIONARY_CAPACITIES = {
    "DICT_4X4_50": 50,
    "DICT_5X5_100": 100,
    "DICT_6X6_250": 250,
    "DICT_7X7_1000": 1000,
}


def normalize_charuco_board_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge a stored spec with legacy defaults and validate the result."""
    if spec is None:
        return dict(DEFAULT_CHARUCO_BOARD_SPEC)
    if not isinstance(spec, Mapping):
        raise ValueError("charuco_board_spec must be an object")

    normalized = dict(DEFAULT_CHARUCO_BOARD_SPEC)
    normalized.update(spec)
    _validate_charuco_board_spec(normalized)
    return {
        "squares_x": int(normalized["squares_x"]),
        "squares_y": int(normalized["squares_y"]),
        "square_length_m": float(normalized["square_length_m"]),
        "marker_length_m": float(normalized["marker_length_m"]),
        "dictionary": normalized["dictionary"],
    }


def validate_charuco_board_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and return a normalized ChArUco board specification."""
    return normalize_charuco_board_spec(spec)


def required_marker_count(squares_x: int, squares_y: int) -> int:
    """Return the number of ArUco markers required by a ChArUco board."""
    return (squares_x * squares_y) // 2


def _validate_charuco_board_spec(spec: Mapping[str, Any]) -> None:
    try:
        squares_x = int(spec["squares_x"])
        squares_y = int(spec["squares_y"])
        square_length_m = float(spec["square_length_m"])
        marker_length_m = float(spec["marker_length_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ChArUco board dimensions and lengths are required") from exc

    if squares_x < 2 or squares_y < 2:
        raise ValueError("ChArUco board must have at least 2 squares in each direction")
    if square_length_m <= 0 or marker_length_m <= 0:
        raise ValueError("ChArUco lengths must be positive meters")
    if marker_length_m >= square_length_m:
        raise ValueError("ChArUco marker length must be smaller than square length")

    dictionary = spec.get("dictionary")
    capacity = CHARUCO_DICTIONARY_CAPACITIES.get(dictionary)
    if capacity is None:
        raise ValueError(f"Unsupported ChArUco dictionary: {dictionary!r}")
    required = required_marker_count(squares_x, squares_y)
    if required > capacity:
        raise ValueError(
            f"ChArUco board requires {required} markers, but {dictionary} supports "
            f"only {capacity}"
        )


__all__ = [
    "CHARUCO_DICTIONARY_CAPACITIES",
    "DEFAULT_CHARUCO_BOARD_SPEC",
    "normalize_charuco_board_spec",
    "required_marker_count",
    "validate_charuco_board_spec",
]
