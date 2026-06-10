
from localllm.secrets import read_hf_token


def test_read_hf_token_from_project_file():
    token = read_hf_token()
    assert token is not None
    assert token.startswith("hf_")
