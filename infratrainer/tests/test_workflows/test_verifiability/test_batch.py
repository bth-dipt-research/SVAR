from workflows.verifiability.utils import get_all_remote_records
import argilla as rg

def test_get_all_remote_records_are_empty(users):
    records = get_all_remote_records(users)
    assert len(records) == 0

def test_get_all_remote_records(users, create_batch):
    # 4 users, 10 records, overlap of 3 should result in 30 records

    records = get_all_remote_records(users)
    assert len(records) == 30
