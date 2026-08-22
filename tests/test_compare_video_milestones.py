import json

from PIL import Image

from scripts.compare_video_milestones import build_comparison


def _gif(path, color):
    frames = [Image.new("RGB", (16, 16), (color + i, 0, 0)) for i in range(4)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def test_comparison_writes_labeled_gif_strip_and_provenance(tmp_path):
    a, b = tmp_path / "a.gif", tmp_path / "b.gif"
    _gif(a, 10)
    _gif(b, 30)
    result = build_comparison([("baseline", a), ("ablation", b)], tmp_path / "out")
    assert all((tmp_path / "out" / name).exists() for name in ("comparison.gif", "strip.png", "manifest.json"))
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["frame_count"] == 4
    assert [row["label"] for row in manifest["inputs"]] == ["baseline", "ablation"]
    assert all(len(row["sha256"]) == 64 for row in manifest["inputs"])
    assert result["strip"].endswith("strip.png")
