from __future__ import annotations

import math

import pyproj
from shapely import ops as so

import disaster_embedding_pipeline as pipeline
import export_fire_lava_visual_pair as exporter
import fire_lava_ecs_rectangle as rectangles


def test_visual_cases_include_requested_2018_events_and_short_folders() -> None:
    by_key = {str(case["event_key"]): case for case in exporter.CASES}

    assert set(by_key) == {
        "thomas_fire_2017",
        "la_palma_lava_2021",
        "walker_fire_2019",
        "shovel_creek_fire_2021",
        "kilauea_lava_2018",
        "rhea_fire_2018",
        "woolsey_fire_2018",
    }
    assert by_key["walker_fire_2019"]["source_kind"] == "heldout_mtbs_sealed"
    assert by_key["walker_fire_2019"]["fire_id"] == exporter.WALKER_FIRE_ID
    assert by_key["shovel_creek_fire_2021"]["source_kind"] == ("heldout_mtbs_sealed")
    assert by_key["shovel_creek_fire_2021"]["fire_id"] == exporter.SHOVEL_CREEK_FIRE_ID
    assert by_key["kilauea_lava_2018"]["output_key"] == "kil"
    assert str(by_key["kilauea_lava_2018"]["label"]).startswith("Kīlauea")
    assert by_key["rhea_fire_2018"]["output_key"] == "rh"
    assert by_key["woolsey_fire_2018"]["output_key"] == "woo"
    for event_key in ("kilauea_lava_2018", "rhea_fire_2018", "woolsey_fire_2018"):
        assert by_key[event_key]["post_window"] == {
            "variant_key": "post_window_2018_2020",
            "pre_year": 2018,
            "post_year": 2020,
        }


def test_walker_source_is_sealed_and_official_mtbs() -> None:
    config, source = exporter.walker_source()

    assert config == {
        "label": "Walker Fire (California, 2019; held-out MTBS reserve)",
        "pre_year": 2018,
        "post_year": 2020,
    }
    assert source["sealed"]["rows"] == 284
    assert source["sealed"]["ecs_column"] == "ecs"
    assert source["sealed"]["reference_column"] == "mtbs_reference"
    assert source["heldout_result"]["replacement"] is True
    assert source["heldout_result"]["rho_ecs_reference"] > 0.9


def test_shovel_creek_source_is_sealed_and_official_mtbs() -> None:
    config, source = exporter.shovel_creek_source()

    assert config == {
        "label": "Shovel Creek Fire (Idaho, 2021; held-out MTBS primary)",
        "pre_year": 2020,
        "post_year": 2022,
    }
    assert source["sealed"]["rows"] == 589
    assert source["sealed"]["ecs_column"] == "ecs"
    assert source["sealed"]["reference_column"] == "mtbs_reference"
    assert source["heldout_result"]["replacement"] is False
    assert source["heldout_result"]["replacement_for"] is None
    assert 0.4 < source["heldout_result"]["rho_ecs_reference"] < 0.5


def test_new_heldout_2018_sources_are_sealed_and_official_mtbs() -> None:
    cases = (
        (exporter.RHEA_FIRE_ID, "RHEA", "Rhea Fire", 1240),
        (exporter.WOOLSEY_FIRE_ID, "WOOLSEY", "Woolsey Fire", 434),
    )
    for fire_id, expected_name, label, expected_rows in cases:
        config, source = exporter.heldout_source(
            fire_id=fire_id,
            expected_name=expected_name,
            expected_event_year=2018,
            label=label,
        )

        assert config["pre_year"] == 2017
        assert config["post_year"] == 2019
        assert source["sealed"]["rows"] == expected_rows
        assert source["sealed"]["ecs_column"] == "ecs"
        assert source["sealed"]["reference_column"] == "mtbs_reference"


def test_thomas_rectangle_is_three_diameters_by_two_diameters() -> None:
    config = pipeline._event_configurations()["thomas_fire_2017"]
    source = exporter.ROOT / "outputs" / "thomas_fire_2017" / "joined_inner.parquet"
    spec = rectangles.build_spec(
        event_key="thomas_fire_2017",
        config=config,
        join_path=source,
        source_kind="pipeline_join",
    )
    local_crs = rectangles._local_crs(spec.center_lon, spec.center_lat)
    to_local = pyproj.Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )
    local_rectangle = so.transform(
        to_local.transform,
        rectangles.rectangle_geometry(spec),
    )

    minimum_x, minimum_y, maximum_x, maximum_y = local_rectangle.bounds
    assert math.isclose(minimum_x, -3.0 * spec.radius_m, abs_tol=1e-4)
    assert math.isclose(maximum_x, 3.0 * spec.radius_m, abs_tol=1e-4)
    assert math.isclose(minimum_y, -2.0 * spec.radius_m, abs_tol=1e-4)
    assert math.isclose(maximum_y, 2.0 * spec.radius_m, abs_tol=1e-4)
    assert set(spec.original_h3_ids) <= set(rectangles.target_h3_ids(spec))


def test_walker_reference_circle_encloses_original_h3_support() -> None:
    config, _ = exporter.walker_source()
    source = exporter._walker_paths()["sealed"]
    spec = rectangles.build_spec(
        event_key="walker_fire_2019",
        config=config,
        join_path=source,
        source_kind="heldout_mtbs_sealed",
    )
    local_crs = rectangles._local_crs(spec.center_lon, spec.center_lat)
    to_local = pyproj.Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    maximum_distance = max(
        math.hypot(x_value, y_value)
        for h3_index in spec.original_h3_ids
        for x_value, y_value in so.transform(
            to_local.transform,
            rectangles._h3_polygon(h3_index),
        ).exterior.coords
    )
    assert math.isclose(maximum_distance, spec.radius_m, abs_tol=1e-6)
    assert set(spec.original_h3_ids) <= set(rectangles.target_h3_ids(spec))


def test_shovel_creek_reference_circle_encloses_original_h3_support() -> None:
    config, _ = exporter.shovel_creek_source()
    source = exporter._shovel_creek_paths()["sealed"]
    spec = rectangles.build_spec(
        event_key="shovel_creek_fire_2021",
        config=config,
        join_path=source,
        source_kind="heldout_mtbs_sealed",
    )
    local_crs = rectangles._local_crs(spec.center_lon, spec.center_lat)
    to_local = pyproj.Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    maximum_distance = max(
        math.hypot(x_value, y_value)
        for h3_index in spec.original_h3_ids
        for x_value, y_value in so.transform(
            to_local.transform,
            rectangles._h3_polygon(h3_index),
        ).exterior.coords
    )
    target_h3 = rectangles.target_h3_ids(spec)
    assert math.isclose(maximum_distance, spec.radius_m, abs_tol=1e-6)
    assert set(spec.original_h3_ids) <= set(target_h3)
    assert len(target_h3) <= rectangles.MAX_TARGET_H3


def test_new_2018_expanded_supports_fit_the_extraction_limit() -> None:
    cases = (
        (
            "kilauea_lava_2018",
            pipeline._event_configurations()["kilauea_lava_2018"],
            exporter.ROOT / "outputs" / "kilauea_lava_2018" / "joined_inner.parquet",
            "pipeline_join",
        ),
        (
            "rhea_fire_2018",
            exporter.heldout_source(
                fire_id=exporter.RHEA_FIRE_ID,
                expected_name="RHEA",
                expected_event_year=2018,
                label="Rhea Fire",
            )[0],
            exporter._heldout_paths(exporter.RHEA_FIRE_ID)["sealed"],
            "heldout_mtbs_sealed",
        ),
        (
            "woolsey_fire_2018",
            exporter.heldout_source(
                fire_id=exporter.WOOLSEY_FIRE_ID,
                expected_name="WOOLSEY",
                expected_event_year=2018,
                label="Woolsey Fire",
            )[0],
            exporter._heldout_paths(exporter.WOOLSEY_FIRE_ID)["sealed"],
            "heldout_mtbs_sealed",
        ),
    )
    for event_key, config, join_path, source_kind in cases:
        spec = rectangles.build_spec(
            event_key=event_key,
            config=config,
            join_path=join_path,
            source_kind=source_kind,
        )
        target_h3 = rectangles.target_h3_ids(spec)

        assert set(spec.original_h3_ids) <= set(target_h3)
        assert len(target_h3) <= rectangles.MAX_TARGET_H3
