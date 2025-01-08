import pandas as pd
import argilla as rg
import re

def _clean(x):
    if isinstance(x, str):
        return re.sub(r'K\d{3,}', '', x).strip()
    return x

def read_data(filename):
    """Returns cleaned requirements"""

    df = pd.read_excel(filename, header=None)
    print("Loaded {} requirements.".format(len(df)))

    #Clean up the sentences which start with the requirement ID.
    df[1] = df[1].map(_clean)

    #Check if the requirement is a string
    mask = [isinstance(s,str) for s in df[1]]
    string = df[mask]
    notstring = df[[not x for x in mask]]

    #Check if the string is not empty
    mask = [len(s.strip()) > 0 for s in string[1]]
    valid = string[mask]
    empty = string[[not x for x in mask]]

    invalid = pd.concat([notstring, empty], axis=0)
    if(len(invalid) > 0):
        print(f'Found {len(invalid)} invalid requirements:')
        for index, r in invalid.iterrows():
            print(f'{r[0]}\n')

    mask = valid.duplicated(subset=[0])
    duplicates = valid[mask]
    uniques = valid[[not x for x in mask]]

    if len(duplicates) > 0:
        print(f'Found {len(duplicates)} duplicated requirement IDs:')
        print(duplicates[0].str.cat(sep=', '))

    print(f'Found {len(uniques)} complete and unique requirements.')

    return uniques

def create_or_get_user(username, role=None, password=None):
    try:
        user = rg.User.from_name(username)
    except Exception:
        ws = create_or_get_workspace(username)

        user = rg.User.create(
            username=username,
            password=password,
            first_name=username,
            role=role,
            workspaces=[ws.name]
        )

    return user

def delete_user(name):
    try:
        rg.User.from_name(name).delete()
    except ValueError as e:
        print(e)
        return False
    return True

def create_or_get_workspace(name):
    try:
        workspace = rg.Workspace.from_name(name)
    except Exception:
        workspace = rg.Workspace.create(name)
    return workspace

def add_user_to_workspace(workspace, user):
    try:
        workspace.add_user(user.id)
    except ValueError:
        print(f"User {user.username} already belongs to {workspace.name}.")


def get_all_remote_records(users):
    records = []
    for user in users:
        datasets = rg.FeedbackDataset.list(workspace=user.username)
        for dataset in datasets:
            for record in dataset.records:
                records.append(record)
    return records


def create_batch(size, annotators, dataset):
    return

