"""Publish the current frozen benchmark state without promoting unfinished runs."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "research/frozen_benchmark_status.json"
START = "<!-- FROZEN_BENCHMARK_PROGRESS_START -->"
END = "<!-- FROZEN_BENCHMARK_PROGRESS_END -->"

def publish():
    if not STATUS.exists():
        return
    status = json.loads(STATUS.read_text())
    policy_path = ROOT / "configs/benchmarks/execution_policy.json"
    campaign_path = ROOT / "research/benchmark_campaign_status.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    if policy.get("benchmark_execution_approved") is True and campaign_path.exists():
        current = json.loads(campaign_path.read_text())
        status = dict(status, updated_utc=current["updated_utc"], stages={
            "Execution": "Released by the user. Physical GPU0 only. Exact runnable jobs and dependent stages are in the maintained queue.",
            "Native ARGformer": current["progress_note"],
            "Native follow-on": "Four complete embedding stages and native retrieval follow genuine contrastive completion through the finite chain. No query truth or ancestry scoring in this chain.",
            "Fresh American data": "20 TRAIN / 5 validation / 10 TEST planned. Fresh shared simulation and tsinfer/tsdate adapters prepared. CPU dispatch and remaining method integrations are explicit.",
            "Frozen TreeLoom": "Architecture unchanged. Full 30-pass, 20-parent training remains pending fresh preparation, target joins and full-size execution integration.",
            "European comparisons": "Paper recipes and cohort reports encoded. Native sampling, bounded European execution and missing ARGMix settings remain pending.",
            "Native FLARE": "Previously completed five 30 Mb validation predictions and scores retained below.",
        }, maintained_campaign=current)
    lines = [START, "## Frozen benchmark execution", "",
             "Updated UTC: " + status["updated_utc"] + ". Physical GPU 0 only.", "",
             "| Stage | Current verified status |",
             "|---|---|"]
    lines += ["| " + name + " | " + value.replace("|", "/") + " |"
              for name, value in status["stages"].items()]
    accepted = status.get("FLARE_all5_accepted_result_ref")
    if accepted:
        result_path = ROOT / accepted["path"]
        assert hashlib.sha256(result_path.read_bytes()).hexdigest() == accepted["sha256"]
        result = json.loads(result_path.read_text())
        assert result["status"] == "completed_FLARE_original5VAL_descriptive_completion_v1"
        assert result["SD_ddof"] == 1 and len(result["per_parent"]) == 5
        rates = result["rates_percent"]
        def value(key):
            row = rates[key]
            return f'{row["equal_parent_mean"]:.4f} ± {row["sample_SD"]:.4f}'
        lines += ["", "**Native FLARE, five American validation simulations:** accuracy " +
                  value("accuracy") + "% and macro F1 " + value("macro_F1") + "%.", "",
                  "| Population | Precision % | Recall % | F1 % |",
                  "|---|---:|---:|---:|"]
        lines += ["| " + population + " | " +
                  " | ".join(value(population + "_" + key)
                             for key in ("precision", "recall", "F1")) + " |"
                  for population in ("EUR", "EAS", "AFR")]
        lines += ["", "Equal weight per simulation, 10,000 fixed focal records each. "
                  "SD is across five simulation parents, with ddof 1. "
                  "VAL000 was previously exposed. This is a common-data development "
                  "comparison, with no exact publication-table reproduction claimed.",
                  "[Complete aggregate](" + accepted["path"] + ")."]
    lines += ["", "The substantial TreeLoom schedule is 20 full 30 Mb TRAIN parents "
              "and 30 passes, or 600 complete-parent updates from fresh weights. "
              "A disposable one-update check is not representative training. "
              "No overall top-1 claim is established.", END]
    block = "\n".join(lines)
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                      lambda _: block, text, count=1, flags=re.S)
    else:
        marker = "## Latest status"
        assert marker in text
        text = text.replace(marker, block + "\n\n" + marker, 1)
    readme.write_text(text)
    detail = ROOT / "research/current_scoreboard.md"
    if detail.exists():
        text = detail.read_text()
        adapted = block.replace("](" + "experiments/", "](../experiments/")
        if START in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                          lambda _: adapted, text, count=1, flags=re.S)
        else:
            text += "\n" + adapted + "\n"
        detail.write_text(text)
    state = ROOT / "research/current_scoreboard.json"
    if state.exists():
        value = json.loads(state.read_text())
        value["frozen_benchmark_execution"] = status
        state.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    publish()
