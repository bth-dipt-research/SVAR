import pytest
import argilla as rg
from argilla.client.feedback.utils import assign_records
from argilla.client.feedback.utils import assign_workspaces
import os

from constants import TEST_PREFIX

def teardown_client():
    try:
        users = rg.User.list()
        for user in users:
            if user.username.startswith(TEST_PREFIX):
                user.delete()

        workspaces = rg.Workspace.list()
        for workspace in workspaces:
            if workspace.name.startswith(TEST_PREFIX):
                datasets = rg.FeedbackDataset.list(workspace=workspace.name)
                for dataset in datasets:
                    rg.FeedbackDataset.from_argilla(name=dataset.name,workspace=workspace.name).delete()
                workspace.delete()
    except ValueError as e:
        print(e)

@pytest.fixture(scope="module")
def argilla_client():
    api_url = os.getenv("ARGILLA_URL", "http://localhost:6900")
    api_key = os.getenv("ARGILLA_API_KEY", "your_api_key")

    rg.init(
        api_url = api_url,
        api_key = api_key,
        workspace = "admin"
    )

    teardown_client()

    try:
        wsname = f'{TEST_PREFIX}ws'
        rg.Workspace.create(wsname)
        rg.set_workspace(wsname)
    except ValueError as e:
        print(e)

    yield

    teardown_client()

@pytest.fixture(scope="module")
def users(argilla_client):
    usernames = [
        f'{TEST_PREFIX}username1',
        f'{TEST_PREFIX}username2',
        f'{TEST_PREFIX}username3',
        f'{TEST_PREFIX}username4'
    ]

    users = []
    for username in usernames:
        ws = rg.Workspace.create(username)

        users.append(
            rg.User.create(
                username=username,
                password=username,
                first_name=username,
                role="annotator",
                workspaces=[ws.name]
            )
        )

    return users

@pytest.fixture(scope="module")
def create_batch(argilla_client, users):
     records = []
     for i in range(0,10):
         records.append(
             rg.FeedbackRecord(
                 fields = {
                     "text": f'sometext{i}'
                 },
                 metadata = {"id": f'{i}'}
             )
         )

     assignments = assign_records(
         users=users,
         records=records,
         overlap=3,
         shuffle=False
     )

     assign_workspaces(
         assignments=assignments,
         workspace_type="individual"
     )

     for username, records in assignments.items():
         dataset = rg.FeedbackDataset(
             fields = [
                 rg.TextField(name="text", title="This is some text")
             ],
             questions = [
                 rg.LabelQuestion(
                     name = "label",
                     title = "Choose a label",
                     labels = ["one", "two", "three"],
                     required = True
                 )
             ],
             allow_extra_metadata = True
         )

         dataset.add_records(records)
         dataset.push_to_argilla(name=f'{TEST_PREFIX}batch', workspace=username)
