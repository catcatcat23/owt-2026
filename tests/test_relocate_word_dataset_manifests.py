from pathlib import Path

import pytest

from tools.relocate_word_dataset_manifests import relocate_path


def test_relocate_path_preserves_relative_suffix():
    assert relocate_path(
        "/old/WORD/Training/image/case/slice.jpg",
        Path("/old/WORD"),
        Path("/new/WORD"),
    ) == "/new/WORD/Training/image/case/slice.jpg"


def test_relocate_path_rejects_unrelated_path():
    with pytest.raises(ValueError, match="not below old root"):
        relocate_path("/other/slice.jpg", Path("/old/WORD"), Path("/new/WORD"))
