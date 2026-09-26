"""
tests/test_traffic_scheduler.py — Unit tests for the Greek high-traffic upload scheduler.

Verifies:
  1. High-traffic slots generate the correct Europe/Athens local times and UTC translations.
  2. Day traffic weighting and pool filtering (weekday vs weekend slots).
  3. Lead time safety buffers and spacing between consecutive short releases.
  4. Integration helper functions get_optimal_schedule_time and get_optimal_schedule_slot.
"""

from __future__ import annotations

from datetime import datetime, timezone

from shorts_engine.services.traffic_scheduler import (
    HIGH_TRAFFIC_POOLS,
    ScheduledSlot,
    get_greek_high_traffic_slots,
    get_optimal_schedule_slot,
    get_optimal_schedule_time,
)


def test_get_greek_high_traffic_slots_count() -> None:
    fixed_start = datetime(2026, 9, 26, 8, 0, 0, tzinfo=timezone.utc)
    slots = get_greek_high_traffic_slots(count=5, start_after_utc=fixed_start)
    assert len(slots) == 5

    for s in slots:
        assert isinstance(s, ScheduledSlot)
        assert s.utc_datetime.tzinfo == timezone.utc
        assert s.local_datetime.tzinfo is not None
        assert s.day_name_greek in {
            "Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"
        }
        assert len(s.pool_name) > 0
        assert len(s.traffic_tier) > 0


def test_slot_conversion_accuracy() -> None:
    fixed_start = datetime(2026, 9, 26, 8, 0, 0, tzinfo=timezone.utc)
    slots = get_greek_high_traffic_slots(count=3, start_after_utc=fixed_start)

    for slot in slots:
        # local_datetime astimezoned to UTC must precisely match utc_datetime
        assert slot.local_datetime.astimezone(timezone.utc) == slot.utc_datetime

        # Local time must be in Athens timezone
        assert slot.local_datetime.tzname() in ("EEST", "EET")

        # Local time must correspond to one of the configured high traffic hours
        valid_times = {(p.hour, p.minute) for p in HIGH_TRAFFIC_POOLS}
        assert (slot.local_datetime.hour, slot.local_datetime.minute) in valid_times


def test_slots_chronologically_sorted_and_spaced() -> None:
    fixed_start = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    slots = get_greek_high_traffic_slots(count=4, start_after_utc=fixed_start, min_buffer_minutes=30)

    for i in range(1, len(slots)):
        prev_utc = slots[i - 1].utc_datetime
        curr_utc = slots[i].utc_datetime
        assert curr_utc > prev_utc

        # Spacing between scheduled shorts must be at least 3.5 hours
        gap_hours = (curr_utc - prev_utc).total_seconds() / 3600.0
        assert gap_hours >= 3.5


def test_weekend_vs_weekday_pool_filtering() -> None:
    # 2026-09-26 is Saturday, 2026-09-27 is Sunday, 2026-09-28 is Monday
    fixed_start = datetime(2026, 9, 26, 0, 0, 0, tzinfo=timezone.utc)
    slots = get_greek_high_traffic_slots(count=8, start_after_utc=fixed_start)

    for slot in slots:
        weekday = slot.local_datetime.weekday()
        is_weekend = weekday in (5, 6)

        if "Weekend Midday" in slot.pool_name:
            assert is_weekend, f"Weekend Midday slot scheduled on weekday {weekday}"
        if "Afternoon Break" in slot.pool_name:
            assert not is_weekend, f"Afternoon Break slot scheduled on weekend {weekday}"


def test_get_optimal_schedule_time_and_slot_match() -> None:
    opt_time = get_optimal_schedule_time(0)
    opt_slot = get_optimal_schedule_slot(0)

    assert isinstance(opt_time, datetime)
    assert opt_time.tzinfo == timezone.utc
    assert opt_slot.utc_datetime == opt_time
    assert "(Ώρα Ελλάδας)" in opt_slot.display_str
    assert "UTC" in opt_slot.display_str
