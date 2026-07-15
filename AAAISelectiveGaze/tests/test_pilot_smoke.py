import json

from AAAISelectiveGaze.scripts.evaluate_selective import main


def test_selective_evaluation_synthetic_smoke_writes_required_artifacts(tmp_path):
    output = tmp_path / "metrics"
    assert main(["--synthetic-smoke", "--output-dir", str(output)]) == 0

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "ok"
    assert metrics["synthetic_smoke"] is True
    assert metrics["num_in_frame_samples"] > 20
    assert (output / "per_sample.csv").is_file()
    assert (output / "coverage_risk.csv").is_file()
    assert metrics["metrics"]["final_plus_disagreement"]["failure_auroc"] > 0.5
