import csv
from pathlib import Path

import pytest


PREFIX = "20260101_120000_Public_libraries_are_useful_01AAAAAAAAAAAAAAAAAAAAAAAA"


def synthetic_rows():
    """Four synthetic adults, three distinct dyadic rounds, no personal data."""
    rows = []
    players = ["A0001", "B0002", "C0003", "D0004"]

    def add(event, player, text, partner="", round_number="", field="", slider=""):
        rows.append(dict(event_order=len(rows) + 1, event_type=event, worker_id=player,
                         empirica_id=player, sender_id=player, recipient_id=partner,
                         text=text, chat_round_order=round_number, field=field,
                         sliderValue=slider, validity="VALID"))

    for player in players:
        add("Initial Opinion", player, f"INITIAL_{player}: Libraries are useful.", slider=5)
    for round_number, pairs in enumerate(((('A0001', 'B0002'), ('C0003', 'D0004')),
                                         (('A0001', 'C0003'), ('B0002', 'D0004')),
                                         (('A0001', 'D0004'), ('B0002', 'C0003'))), 1):
        for a, b in pairs:
            for player, partner in ((a, b), (b, a)):
                add("tweet", player, f"HUMAN_TWEET_{round_number}_{player}", partner, round_number)
            for turn, (player, partner) in enumerate(((a, b), (b, a), (a, b), (b, a))):
                text = f"HUMAN_CHAT_{round_number}_{turn}_{player}"
                add("message_sent", player, text, partner, round_number)
                add("message_recieved", partner, text, player, round_number)
    for player in players:
        add("Post Opinion", player, f"HUMAN_POST_{player}", slider=5)
        add("exit_survey", player, "30", field="age")
        add("exit_survey", player, "['Synthetic']", field="ethnicity")
    return rows


def write_session(path: Path, rows=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = synthetic_rows() if rows is None else rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def session(tmp_path):
    return write_session(tmp_path / f"{PREFIX}.csv")
