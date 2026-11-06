"""Protect the shared convergence reference and original experiment selection."""
import pytest

from reproduction.tables import convergence_rows, prediction_error_rows, LLAMA_MODELS


def test_shared_reference_changes_model_delta_but_preserves_original_family_values():
    models = {"human": "human", "base": "base-model", "SFT": "sft-model"}
    trajectories = {
        "human": {"public": ([1.4, 1.2], [0., 0.]), "private": ([1.5, 1.3], [0., 0.])},
        "base-model": {"public": ([1.6, 1.0], [0., 0.]), "private": ([1.7, 0.8], [0., 0.])},
        "sft-model": {"public": ([1.3, 1.1], [0., 0.]), "private": ([1.4, 1.2], [0., 0.])},
    }
    paper, original = convergence_rows("Qwen", "group_test", models, trajectories, {"public": 1.25, "private": 1.35})
    # The table must change the subtraction, not simply replace the Human cell.
    assert len(paper) == 4
    actual = {(row["channel"], row["role"]): row["mean_delta_sd"] for row in paper}
    assert actual == pytest.approx({("public", "base"): -0.25, ("public", "SFT"): -0.15,
                                    ("private", "base"): -0.55, ("private", "SFT"): -0.15})
    assert len(original) == 6
    base = next(row for row in original if row["channel"] == "public" and row["role"] == "base")
    assert base["raw_initial_mean_sd"] == 1.6
    assert base["family_human_initial_mean_sd"] == 1.4
    assert base["mean_delta_sd"] == pytest.approx(-0.4)
    human = next(row for row in original if row["channel"] == "public" and row["role"] == "human")
    assert human["mean_delta_sd"] == pytest.approx(-0.2)


def test_error_rows_select_ruixuan_fds_and_keep_public_private_separate():
    models = {"human": "human", "base": "Llama-3.1-8B-Instruct", "SFT": LLAMA_MODELS["group_test"]}
    def row(model, version, public, private):
        return {"model": model, "simulation_version": version,
                "public_stance_alignment_abs": public[0], "public_stance_change_error_abs": public[1], "public_opinion_convergence_error": public[2],
                "private_stance_alignment_abs": private[0], "private_stance_change_error_abs": private[1], "private_opinion_convergence_error": private[2]}
    raw = [row(models["base"], "v0", (1.6, 1.7, .8), (1.8, 1.9, .9)),
           row(models["SFT"], "v0", (1.1, 1.2, .6), (1., .9, .5)),
           row(models["SFT"], "v1_5", (4., 4., 4.), (4., 4., 4.)),
           row("ft:Llama-3.1-8B-Instruct-SFT-20260613:leo-group-3epochs", "v0", (3., 3., 3.), (3., 3., 3.))]
    errors = prediction_error_rows("Llama", "group_test", models, raw)
    assert errors["public"][1]["public_opinion_convergence_error"] == .6
    assert errors["private"][1]["private_opinion_convergence_error"] == .5
    assert errors["public"][1]["model"] == LLAMA_MODELS["group_test"]
    with pytest.raises(ValueError, match="Expected one Full Dynamics row"):
        prediction_error_rows("Llama", "group_test", models, raw + [raw[1]])


@pytest.mark.parametrize("mutation", ["missing_reference", "duplicate_reference", "extra_reference", "missing_output", "duplicate_output", "extra_output"])
def test_verification_rejects_incomplete_or_duplicate_table_cells(mutation):
    import json
    from reproduction.run import HERE
    from reproduction.tables import verify_cross_model_tables

    expected = json.loads((HERE / "reference/cross_model_table_values.json").read_text())
    grouped = {}
    for item in expected:
        identity = tuple(item[key] for key in ("family", "split", "channel", "role"))
        kind = "convergence" if item["metric"] == "mean_delta_sd" else item["channel"]
        row = grouped.setdefault((kind, identity), {key: item[key] for key in ("family", "split", "channel", "role")})
        row[item["metric"]] = item["source_value"]
    convergence = [row for (kind, _), row in grouped.items() if kind == "convergence"]
    errors = {channel: [row for (kind, _), row in grouped.items() if kind == channel] for channel in ("public", "private")}
    assert len(verify_cross_model_tables(convergence, errors, expected)) == 100
    if mutation == "missing_reference":
        expected.pop()
    elif mutation == "duplicate_reference":
        expected.append(dict(expected[0]))
    elif mutation == "extra_reference":
        expected.append({**expected[0], "family": "Other"})
    elif mutation == "missing_output":
        convergence.pop()
    elif mutation == "duplicate_output":
        errors["public"].append(dict(errors["public"][0]))
    else:
        errors["public"].append({**errors["public"][0], "family": "Other"})
    with pytest.raises(ValueError):
        verify_cross_model_tables(convergence, errors, expected)
