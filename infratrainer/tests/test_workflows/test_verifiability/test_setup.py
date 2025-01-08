from workflows.verifiability.utils import create_or_get_user, delete_user, create_or_get_workspace
from constants import TEST_PREFIX
import pytest
import argilla as rg

def test_create_user(argilla_client):
    username = f'{TEST_PREFIX}user'
    user = create_or_get_user(username, "annotator", "abcdefghi")
    assert user.is_annotator
    assert user.username == username
    ws = create_or_get_workspace(username)
    assert username == ws.name
    assert len(ws.users) == 1
    assert ws.users[0].username == user.username

def test_get_user(argilla_client):
    user = create_or_get_user("owner")
    assert user.is_owner
    assert user.username == "owner"

def test_delete_user(argilla_client):
    assert delete_user(f'{TEST_PREFIX}user')

def test_create_workspace(argilla_client):
    wsname = f'{TEST_PREFIX}new'
    ws = create_or_get_workspace(wsname)
    assert ws.name == wsname

def test_create_invalid_workspace_name(argilla_client):
    wsname = f'{TEST_PREFIX}NEW' #ws names must be lowercase
    with pytest.raises(Exception):
        create_or_get_workspace(wsname)

def test_get_workspace(argilla_client):
    wsname = f'{TEST_PREFIX}ws'
    ws = create_or_get_workspace(wsname)
    assert ws.name == wsname

