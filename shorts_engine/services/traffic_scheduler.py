"""
services/traffic_scheduler.py — High-traffic upload scheduler for Greece.

Computes optimal YouTube Shorts upload and publish time slots based on
Greek social media consumption patterns and peak traffic windows (Europe/Athens).
Converts all local Greek peak times to UTC as required by YouTube Data API v3.
"""

from __future__ import annotations

import logging
import zoneinfo
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

logger = logging.getLogger(__name__)

# Primary timezone for Greece
GREECE_TZ = zoneinfo.ZoneInfo("Europe/Athens")

# Greek day names for display
GREEK_DAY_NAMES: dict[int, str] = {
    0: "Δευτέρα",
    1: "Τρίτη",
    2: "Τετάρτη",
    3: "Πέμπτη",
    4: "Παρασκευή",
    5: "Σάββατο",
    6: "Κυριακή",
}

# Day traffic ratings in Greece (0=Monday ... 6=Sunday)
DAY_TRAFFIC_WEIGHTS: dict[int, float] = {
    4: 1.00,  # Παρασκευή (Peak viral reach & weekend anticipation)
    3: 0.95,  # Πέμπτη (High mid-week evening engagement)
    6: 0.92,  # Κυριακή (High evening home leisure scroll)
    2: 0.88,  # Τετάρτη (Mid-week peak for news & political debate)
    5: 0.85,  # Σάββατο (Midday & late-night peak)
    1: 0.78,  # Τρίτη (Steady weekday consumption)
    0: 0.72,  # Δευτέρα (Catch-up day)
}


@dataclass(frozen=True)
class HighTrafficSlotDefinition:
    """Definition of a recurring peak hour window in Greek local time."""
    hour: int
    minute: int
    pool_name: str
    traffic_tier: str
    is_weekend_only: bool = False
    is_weekday_only: bool = False


# High traffic hour pools in Greece (Europe/Athens local time):
# 1. Prime Evening Peak (19:30 - 21:45): The highest retention and algorithmic surge window in Greece.
# 2. Afternoon Break Peak (14:30 - 15:45): Office/university break and commuter scroll.
# 3. Weekend Midday Peak (12:00 - 13:30): Saturday/Sunday morning coffee & casual browse.
HIGH_TRAFFIC_POOLS: list[HighTrafficSlotDefinition] = [
    # Core Prime Evening (Top priority in Greece)
    HighTrafficSlotDefinition(
        hour=20,
        minute=45,
        pool_name="Prime Evening Peak (Core)",
        traffic_tier="🔥 Peak Viral (10/10)",
    ),
    HighTrafficSlotDefinition(
        hour=19,
        minute=30,
        pool_name="Early Evening Prime (News & Commute)",
        traffic_tier="🔥 Very High (9.5/10)",
    ),
    HighTrafficSlotDefinition(
        hour=21,
        minute=30,
        pool_name="Late Night Prime",
        traffic_tier="⚡ High (8.8/10)",
    ),
    # Afternoon Siesta / Lunch break
    HighTrafficSlotDefinition(
        hour=15,
        minute=15,
        pool_name="Afternoon Break / Siesta",
        traffic_tier="⚡ High (8.5/10)",
        is_weekday_only=True,
    ),
    # Weekend midday coffee browse
    HighTrafficSlotDefinition(
        hour=12,
        minute=15,
        pool_name="Weekend Midday Coffee",
        traffic_tier="⚡ High (8.7/10)",
        is_weekend_only=True,
    ),
]


@dataclass(frozen=True)
class ScheduledSlot:
    """A fully resolved publishing time slot."""
    utc_datetime: datetime
    local_datetime: datetime
    pool_name: str
    day_name_greek: str
    traffic_tier: str

    @property
    def display_str(self) -> str:
        """Formatted human-readable string for Streamlit UI."""
        date_fmt = self.local_datetime.strftime("%d/%m/%Y %H:%M")
        utc_fmt = self.utc_datetime.strftime("%H:%M UTC")
        return (
            f"📅 {self.day_name_greek} {date_fmt} (Ώρα Ελλάδας) [{utc_fmt}] — {self.pool_name} ({self.traffic_tier})"
        )


def get_greek_high_traffic_slots(
    count: int = 5,
    start_after_utc: datetime | None = None,
    min_buffer_minutes: int = 45,
) -> list[ScheduledSlot]:
    """
    Generate the next `count` optimal publishing slots tailored for Greek audience traffic.

    Sequences slots across the best high-traffic hours and days in Europe/Athens time,
    ensuring each video has at least `min_buffer_minutes` of lead time from now.

    Args:
        count: Number of consecutive slots to reserve (e.g. for batch of shorts).
        start_after_utc: Earliest threshold (defaults to current time in UTC).
        min_buffer_minutes: Minimum minutes ahead of threshold to avoid publishing in past.

    Returns:
        List of ScheduledSlot objects sorted chronologically.
    """
    if start_after_utc is None:
        start_after_utc = datetime.now(timezone.utc)
    elif start_after_utc.tzinfo is None:
        start_after_utc = start_after_utc.replace(tzinfo=timezone.utc)

    # Convert threshold to Greek local time with minimum safety buffer
    buffer_delta = timedelta(minutes=max(15, min_buffer_minutes))
    earliest_allowed_utc = start_after_utc + buffer_delta
    earliest_allowed_local = earliest_allowed_utc.astimezone(GREECE_TZ)

    candidate_slots: list[ScheduledSlot] = []

    # Search through the next 14 days
    current_date = earliest_allowed_local.date()
    days_checked = 0

    while len(candidate_slots) < count * 3 and days_checked < 14:
        day_date = current_date + timedelta(days=days_checked)
        weekday = day_date.weekday()
        is_weekend = weekday in (5, 6)

        # Evaluate candidate time slots for this day
        day_candidates: list[tuple[datetime, HighTrafficSlotDefinition]] = []
        for slot_def in HIGH_TRAFFIC_POOLS:
            if slot_def.is_weekend_only and not is_weekend:
                continue
            if slot_def.is_weekday_only and is_weekend:
                continue

            local_dt = datetime.combine(
                day_date,
                time(slot_def.hour, slot_def.minute),
                tzinfo=GREECE_TZ,
            )

            if local_dt >= earliest_allowed_local:
                day_candidates.append((local_dt, slot_def))

        # Sort candidate slots of the day chronologically
        day_candidates.sort(key=lambda item: item[0])

        for local_dt, slot_def in day_candidates:
            utc_dt = local_dt.astimezone(timezone.utc)
            candidate_slots.append(
                ScheduledSlot(
                    utc_datetime=utc_dt,
                    local_datetime=local_dt,
                    pool_name=slot_def.pool_name,
                    day_name_greek=GREEK_DAY_NAMES.get(weekday, ""),
                    traffic_tier=slot_def.traffic_tier,
                )
            )

        days_checked += 1

    # Select slots with spacing: if multiple videos, space them across optimal slots
    # (avoid dumping multiple videos in the exact same hour on the same day)
    selected_slots: list[ScheduledSlot] = []
    used_days_count: dict[str, int] = {}

    for slot in candidate_slots:
        day_key = slot.local_datetime.strftime("%Y-%m-%d")
        # Max 2 shorts per day, spaced at least 4 hours apart
        if used_days_count.get(day_key, 0) < 2:
            if selected_slots:
                prev_slot = selected_slots[-1]
                time_gap_hours = (slot.utc_datetime - prev_slot.utc_datetime).total_seconds() / 3600.0
                if time_gap_hours < 3.5:
                    continue

            selected_slots.append(slot)
            used_days_count[day_key] = used_days_count.get(day_key, 0) + 1

            if len(selected_slots) >= count:
                break

    # Fallback if candidates were insufficient
    while len(selected_slots) < count:
        last_dt = selected_slots[-1].local_datetime if selected_slots else earliest_allowed_local
        next_local = last_dt + timedelta(days=1)
        next_local = next_local.replace(hour=20, minute=45, second=0, microsecond=0)
        utc_dt = next_local.astimezone(timezone.utc)
        selected_slots.append(
            ScheduledSlot(
                utc_datetime=utc_dt,
                local_datetime=next_local,
                pool_name="Prime Evening Peak (Core)",
                day_name_greek=GREEK_DAY_NAMES.get(next_local.weekday(), ""),
                traffic_tier="🔥 Peak Viral (10/10)",
            )
        )

    return selected_slots[:count]


def get_optimal_schedule_time(slot_index: int = 0) -> datetime:
    """
    Return the optimal UTC datetime for scheduling a Short targeting the Greek audience.

    Drop-in replacement for the legacy hardcoded function.
    For slot_index=0, picks the next high-traffic Greek window.
    For slot_index=1, 2, ... spaces them intelligently across subsequent peak slots.

    Returns:
        timezone-aware datetime in UTC.
    """
    needed_count = max(1, slot_index + 1)
    slots = get_greek_high_traffic_slots(count=needed_count)
    chosen_slot = slots[min(slot_index, len(slots) - 1)]
    return chosen_slot.utc_datetime


def get_optimal_schedule_slot(slot_index: int = 0) -> ScheduledSlot:
    """
    Return the full ScheduledSlot object with both Greek local time and UTC.
    """
    needed_count = max(1, slot_index + 1)
    slots = get_greek_high_traffic_slots(count=needed_count)
    return slots[min(slot_index, len(slots) - 1)]
