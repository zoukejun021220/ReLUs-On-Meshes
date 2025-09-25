from __future__ import annotations

from viz_tool.viewer import _read_seed_file


def test_read_seed_file_with_labels(tmp_path):
    seed_txt = tmp_path / "seeds.txt"
    seed_txt.write_text(
        """
        # comment line
        1
        2 custom
        3 custom label
        4, ignored
        5 another // inline comment
        6
        """
    )

    indices, labels = _read_seed_file(str(seed_txt), "seed")

    assert indices == [1, 2, 3, 4, 5, 6]
    assert labels == ["seed", "custom", "custom label", "ignored", "another", "seed"]
