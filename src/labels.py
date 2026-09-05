"""Canonical label set for the AHC Visual Intelligence Hackathon.

These twelve strings are the contract with the scorer. They are copied
verbatim from train/*/ground_truth.csv and test/ground_truth.csv; a single
character of drift silently zeroes a class. Never retype them by hand.
"""

CLASSES = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]

NORMAL = "normal"
ANOMALY_CLASSES = [c for c in CLASSES if c != NORMAL]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Groups the audit showed will dominate confusion. Stage A predicts the group,
# the VLM arbitrates within it -- never ask the VLM to pick from 12 flat options.
CONFUSABLE_GROUPS = [
    ["fire", "smoke"],
    ["traffic_congestion", "vehicle_blocking_traffic", "stalled_or_broken_down_vehicle"],
    ["normal", "loitering_or_suspicious_presence"],
    ["traffic_accident", "stalled_or_broken_down_vehicle"],
    # Measured, not assumed: on T025 the temporal head assigns 98.2% of its
    # class mass to wrong_way_driving for six intervals whose ground truth is
    # traffic_accident, and traffic_accident does not appear in its top four.
    # Without this pair the shortlist handed to the VLM would not contain the
    # right answer at all, so Stage B could never recover the error.
    ["wrong_way_driving", "traffic_accident", "vehicle_blocking_traffic"],
]


def is_valid(class_name: str) -> bool:
    return class_name in CLASS_TO_IDX


def validate_series(values) -> list[str]:
    """Return the sorted set of invalid class strings found in `values`."""
    return sorted({v for v in values if not is_valid(v)})


def group_of(class_name: str) -> list[str]:
    """The confusable shortlist a class belongs to, or just the class itself."""
    for g in CONFUSABLE_GROUPS:
        if class_name in g:
            return g
    return [class_name]


if __name__ == "__main__":
    for i, c in enumerate(CLASSES):
        print(f"{i:2d}  {c}")
    print(f"\n{len(CLASSES)} classes, {len(ANOMALY_CLASSES)} anomaly classes")
