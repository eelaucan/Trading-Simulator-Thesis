"""Filesystem export helpers for completed human experiment sessions."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from simulator.env import TradingEnvironment
from simulator.metrics import SimulationMetrics
from ui.session import SessionMetadata, SessionStatus


def export_session_results(
    *,
    metadata: SessionMetadata,
    status: SessionStatus,
    env: TradingEnvironment,
    metrics: SimulationMetrics,
    output_root: str | Path = "output/sessions",
) -> Path:
    """Write session metadata, metrics, and simulator logs to disk."""
    export_dir = _build_export_dir(
        output_root=Path(output_root),
        participant_id=metadata.participant_id,
        started_at_iso=metadata.started_at.strftime("%Y%m%dT%H%M%S"),
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = export_dir / "session_metadata.json"
    metrics_path = export_dir / "metrics.json"
    action_log_path = export_dir / "action_log.csv"
    validation_log_path = export_dir / "validation_log.csv"
    execution_log_path = export_dir / "execution_log.csv"
    batch_log_csv_path = export_dir / "batch_log.csv"
    batch_log_jsonl_path = export_dir / "batch_log.jsonl"
    weekly_returns_path = export_dir / "weekly_returns.csv"
    manifest_path = export_dir / "manifest.json"

    metadata_payload = metadata.to_dict()
    metadata_payload["status"] = status.value
    _write_json(metadata_path, metadata_payload)
    _write_json(metrics_path, _metrics_to_json(metrics))

    env.logger.to_action_dataframe(include_internal=True).to_csv(action_log_path, index=False)
    metrics.validation_log_df.to_csv(validation_log_path, index=False)
    metrics.execution_log_df.to_csv(execution_log_path, index=False)
    env.logger.to_batch_dataframe().to_csv(batch_log_csv_path, index=False)
    env.logger.export_jsonl(batch_log_jsonl_path)
    pd.DataFrame(
        {
            "week_index": list(metrics.weekly_returns.index),
            "weekly_return": metrics.weekly_returns.tolist(),
        }
    ).to_csv(weekly_returns_path, index=False)

    manifest_payload = {
        "participant_id": metadata.participant_id,
        "episode_name": metadata.episode_name,
        "condition": metadata.condition,
        "started_at": metadata.started_at.isoformat(),
        "decision_start_week": metadata.decision_start_week,
        "visible_history_weeks_at_start": metadata.visible_history_weeks_at_start,
        "finished_at": metadata.finished_at.isoformat() if metadata.finished_at else None,
        "status": status.value,
        "files": {
            "session_metadata": metadata_path.name,
            "metrics": metrics_path.name,
            "action_log": action_log_path.name,
            "validation_log": validation_log_path.name,
            "execution_log": execution_log_path.name,
            "batch_log_csv": batch_log_csv_path.name,
            "batch_log_jsonl": batch_log_jsonl_path.name,
            "weekly_returns": weekly_returns_path.name,
        },
    }
    _write_json(manifest_path, manifest_payload)
    return export_dir


def _build_export_dir(
    *,
    output_root: Path,
    participant_id: str,
    started_at_iso: str,
) -> Path:
    safe_participant_id = _slugify(participant_id)
    return output_root.expanduser() / f"{safe_participant_id}_{started_at_iso}"


def _metrics_to_json(metrics: SimulationMetrics) -> dict[str, Any]:
    return {
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "realized_vol": metrics.realized_vol,
        "avg_weekly_turnover": metrics.avg_weekly_turnover,
        "avg_hhi": metrics.avg_hhi,
        "max_hhi": metrics.max_hhi,
        "blow_up_flag": metrics.blow_up_flag,
        "sharpe_ratio": metrics.sharpe_ratio,
        "n_invalid_attempts": metrics.n_invalid_attempts,
        "n_clipped_trades": metrics.n_clipped_trades,
        "n_stop_triggers": metrics.n_stop_triggers,
        "n_gap_adjustments": metrics.n_gap_adjustments,
        "vol_rule_activation_week": metrics.vol_rule_activation_week,
        "weekly_returns": metrics.weekly_returns.tolist(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "session"
