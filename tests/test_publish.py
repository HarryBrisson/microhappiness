"""aggregation_spec.json is shared by the happiness, outcomes and ATUS writers; each must keep its own layers."""

import json

from microhappiness import publish


def test_happiness_metrics_list_only_happiness_layers(tmp_path):
    # outcomes/ATUS layers already in the spec, as when happiness is rebuilt after them
    (tmp_path / "aggregation_spec.json").write_text(json.dumps({"layers": {
        "outcomes_tract": {"file": "outcomes_tract.csv", "kind": "polygon_values", "id_field": "geoid"},
        "atus_tract": {"file": "atus_tract.csv", "kind": "polygon_values", "id_field": "geoid"},
    }, "metrics": {}}))
    for geo in ("tract", "zcta"):
        publish.write_spec(tmp_path, [geo], vintage="2022", model_key="m5", gss_years="1972-2024")
    spec = json.loads((tmp_path / "aggregation_spec.json").read_text())
    assert set(spec["layers"]) == {"outcomes_tract", "atus_tract", "tract", "zcta"}
    for mid in publish.METRICS:
        assert spec["metrics"][mid]["layer"] == ["tract", "zcta"]
