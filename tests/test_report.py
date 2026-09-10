from src.report import metrics_table, winner

METRICS = {
    "naive_168h": {
        "mean": {"wape": 0.0512, "mae": 2100.0, "rmse": 2900.0, "pinball": 900.0, "coverage": 0.81},
        "std": {"wape": 0.004, "mae": 90.0, "rmse": 120.0, "pinball": 40.0, "coverage": 0.03},
    },
    "lightgbm_l1": {
        "mean": {"wape": 0.0388, "mae": 1600.0, "rmse": 2200.0, "pinball": 700.0, "coverage": 0.77},
        "std": {"wape": 0.003, "mae": 70.0, "rmse": 100.0, "pinball": 30.0, "coverage": 0.04},
    },
}


def test_winner_is_the_lowest_wape():
    name, value = winner(METRICS)
    assert name == "lightgbm_l1"
    assert value == 0.0388


def test_table_has_one_row_per_model_and_shows_dispersion():
    table = metrics_table(METRICS)
    lines = [line for line in table.splitlines() if line.startswith("|")]
    assert len(lines) == 4, "header, separator, two models"
    assert "naive_168h" in table
    assert "±" in table, "the standard deviation must be visible"


def test_table_is_sorted_by_wape():
    table = metrics_table(METRICS)
    assert table.index("lightgbm_l1") < table.index("naive_168h")
