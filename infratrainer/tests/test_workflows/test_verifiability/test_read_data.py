from workflows.verifiability.utils import read_data

data_file = "../data/TRVInfra_all_only_complete_requirements_english.xlsx"

def test_read_data():
    df = read_data(data_file)
    assert len(df) == 17500

def is_string(s):
    assert isinstance(s, str)
    return s

def test_requirement_is_string():
    df = read_data(data_file)
    df.apply(lambda r : is_string(r[1]), axis=1)

def length_is_positive(s):
    assert len(s) > 0
    return s

def test_requirement_not_empy():
    df = read_data(data_file)
    df.apply(lambda r : length_is_positive(r[1]), axis=1)

def test_unique_ids():
    df = read_data(data_file)
    assert not df.duplicated(subset=[0]).any()

