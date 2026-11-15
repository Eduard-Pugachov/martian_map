import importlib.util

import pytest

from martianmap.config import load_config, repo_path

if not repo_path(load_config()["app"]["bundle"]).exists():
    pytest.skip("app bundle not built (scripts/export_app_data.py)", allow_module_level=True)

spec = importlib.util.spec_from_file_location("app", repo_path("app/app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_demo_pair_both_modes_render_and_score() -> None:
    pts, image, table, msg = app.demo_pair(app.MODE_FREE, app.OVERLAYS[2], True)
    assert image.shape == (*app.SLOPE.shape, 3) and image.dtype.name == "uint8"
    rows = dict(zip(table["Route"], table["True hazard cells crossed"]))
    assert rows[app.NAMES["T"]] == 0 and rows[app.NAMES[app.RISK_KEY]] == 0 and rows[app.NAMES["P"]] > 0
    _, table, _ = app.update(app.MODE_DTM, app.OVERLAYS[0], pts, False)
    assert list(table["Route"]) == [app.NAMES["T"]]


def test_click_sets_start_then_goal_and_blocked_point_is_reported() -> None:
    class Evt:
        index = (10, 20)

    pts, *_, msg = app.on_click(app.MODE_DTM, app.OVERLAYS[0], {"start": None, "goal": None}, False, Evt())
    assert pts == {"start": (20, 10), "goal": None} and "goal" in msg
    blocked = tuple(int(v) for v in app.np.argwhere(~app.np.isfinite(app.COSTS["T"]))[0])
    _, table, msg = app.update(app.MODE_DTM, app.OVERLAYS[0], {"start": blocked, "goal": (600, 600)}, False)
    assert table.empty and msg.startswith("No route")
