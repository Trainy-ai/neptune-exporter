#!/usr/bin/env python3
from __future__ import annotations

import gc
import logging
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import pyarrow as pa

from neptune_exporter.types import ProjectId, TargetExperimentId, TargetRunId
from neptune_exporter.loaders.loader import DataLoader


class PlutoLoader(DataLoader):
    """Loads Neptune-exported parquet data into Pluto (pluto-ml), memory-hardened."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
        skip_runtime_attachments: bool = True,
    ) -> None:
        self._logger = logging.getLogger(__name__)

        self.api_key = api_key
        self.host = host
        self.name_prefix = name_prefix
        self._skip_runtime_attachments = skip_runtime_attachments

        # Smaller defaults; Pluto tends to buffer internally.
        self._batch_rows: int = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_BATCH_ROWS", "10000"))
        # Hard cap to prevent “I set 200000 by accident” scenarios.
        self._batch_rows = max(1000, min(self._batch_rows, 20000))

        # Default to downsampling; lossless metrics with 900k steps is unfeasible
        self._log_every: int = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_LOG_EVERY", "50"))
        if self._log_every <= 0:
            self._log_every = 1

        # Flush frequently if Pluto supports it
        self._flush_every_steps: int = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_FLUSH_EVERY_N_STEPS", "50"))
        if self._flush_every_steps <= 0:
            self._flush_every_steps = 50

        # Skip toggles (use these to isolate the culprit quickly)
        self._skip_params = os.getenv("NEPTUNE_EXPORTER_PLUTO_SKIP_PARAMS", "0") == "1"
        self._skip_metrics = os.getenv("NEPTUNE_EXPORTER_PLUTO_SKIP_METRICS", "0") == "1"
        self._skip_files = os.getenv("NEPTUNE_EXPORTER_PLUTO_SKIP_FILES", "0") == "1"
        self._skip_text = os.getenv("NEPTUNE_EXPORTER_PLUTO_SKIP_TEXT", "0") == "1"
        self._skip_hist = os.getenv("NEPTUNE_EXPORTER_PLUTO_SKIP_HIST", "0") == "1"

        # Metrics streaming: buffer size before flushing (bounds memory)
        self._metrics_stream_buffer_steps: int = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_METRICS_STREAM_BUFFER_STEPS", "1000"))
        if self._metrics_stream_buffer_steps < 100:
            self._metrics_stream_buffer_steps = 100

        # Optional hard cap to prevent pathological runs (0 = disabled)
        self._max_files_per_run: int = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_MAX_FILES_PER_RUN", "0"))
        if self._max_files_per_run < 0:
            self._max_files_per_run = 0

        # Base dir
        self._pluto_base_dir = Path(os.getenv("NEPTUNE_EXPORTER_PLUTO_BASE_DIR", ".")).resolve()
        self._pluto_base_dir.mkdir(parents=True, exist_ok=True)

        # Duplicate cache
        cache_default = self._pluto_base_dir / ".neptune_exporter_pluto_loaded_runs.txt"
        self._loaded_cache_path = Path(os.getenv("NEPTUNE_EXPORTER_PLUTO_LOADED_CACHE", str(cache_default))).resolve()

        self._loaded_run_keys: set[str] = set()
        try:
            if self._loaded_cache_path.exists():
                self._loaded_run_keys = set(
                    line.strip()
                    for line in self._loaded_cache_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        except Exception:
            self._logger.debug("Failed reading local Pluto loaded-run cache", exc_info=True)

        # Import Pluto
        try:
            import pluto  # type: ignore

            self._pluto = pluto
            self._client = None
            if hasattr(pluto, "Client"):
                try:
                    self._client = pluto.Client(api_key=api_key, host=host)
                except Exception:
                    self._logger.debug("pluto.Client failed; will rely on pluto.init()", exc_info=True)
            else:
                self._client = pluto
        except Exception:
            self._pluto = None
            self._client = None
            self._logger.debug("pluto SDK not available", exc_info=True)

        if show_client_logs:
            logging.getLogger("pluto").setLevel(logging.INFO)
        else:
            logging.getLogger("pluto").setLevel(logging.ERROR)

        self._ops: dict[str, object] = {}
        self._run_id_to_key: dict[str, str] = {}

    def _ensure_pluto(self):
        if self._pluto is None:
            raise RuntimeError("Pluto SDK not available. Install `pluto-ml`.")
        return self._pluto

    def _run_key(self, project_id: ProjectId, run_name: str) -> str:
        return f"{project_id}::{run_name}"

    def _mark_run_loaded(self, key: str) -> None:
        self._loaded_run_keys.add(key)
        try:
            self._loaded_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._loaded_cache_path.open("a", encoding="utf-8") as f:
                f.write(key + "\n")
        except Exception:
            self._logger.debug("Failed writing local Pluto loaded-run cache", exc_info=True)

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.\s/]", "_", str(attribute_path))
        if len(sanitized) > 250:
            sanitized = sanitized[:250]
        return sanitized

    def _convert_step_to_int_optional(self, step: Optional[Decimal], step_multiplier: int) -> Optional[int]:
        if step is None or pd.isna(step):
            return None
        return int(float(step) * step_multiplier)

    def _ensure_runtime_dirs(self, pluto_project: str, run_name: str) -> None:
        if not self._skip_runtime_attachments:
            return
        for p in (
            self._pluto_base_dir / ".pluto" / pluto_project / run_name / "files" / "runtime",
            self._pluto_base_dir / ".pluto" / ".pluto" / pluto_project / run_name / "files" / "runtime",
        ):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _maybe_flush(self, op: object) -> None:
        try:
            if hasattr(op, "flush"):
                op.flush()
        except Exception:
            self._logger.debug("op.flush failed (non-fatal)", exc_info=True)

    def create_experiment(self, project_id: ProjectId, experiment_name: str) -> TargetExperimentId:
        # Allow overriding project name via env var (useful for multiple imports)
        override = os.getenv("NEPTUNE_EXPORTER_PLUTO_PROJECT_NAME")
        if override:
            target_name = override
        else:
            target_name = str(project_id)
            if self.name_prefix:
                target_name = f"{self.name_prefix}/{target_name}"

        self._logger.info("Pluto client not required; using project name as id")
        self._logger.info("Using Pluto project/experiment %s", target_name)
        return TargetExperimentId(target_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        key = self._run_key(project_id, run_name)
        if key in self._loaded_run_keys:
            return TargetRunId(run_name)

        client = self._client
        if client is None:
            return None

        try:
            if hasattr(client, "find_run"):
                found = client.find_run(project=str(experiment_id), name=run_name)
                if found:
                    rid = getattr(found, "id", str(found))
                    return TargetRunId(str(rid))
            if hasattr(client, "list_runs"):
                for r in client.list_runs(project=str(experiment_id)):
                    if getattr(r, "name", None) == run_name:
                        rid = getattr(r, "id", run_name)
                        return TargetRunId(str(rid))
        except Exception:
            self._logger.debug("Pluto client run lookup failed (non-fatal)", exc_info=True)

        return None

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        pluto = self._ensure_pluto()

        key = self._run_key(project_id, run_name)
        if key in self._loaded_run_keys:
            self._logger.info("Run '%s' already loaded (cache hit); skipping", run_name)
            return TargetRunId(run_name)

        already = self.find_run(project_id, run_name, experiment_id)
        if already is not None:
            self._logger.info("Run '%s' already loaded (API hit); skipping", run_name)
            self._mark_run_loaded(key)
            return already

        tags = ["import:neptune", f"import_project:{project_id}"]
        pluto_project = str(experiment_id) if experiment_id else str(project_id)

        self._ensure_runtime_dirs(pluto_project, run_name)

        op = pluto.init(
            dir=str(self._pluto_base_dir),
            project=pluto_project,
            name=run_name,
            config=None,
            tags=tags,
        )

        run_id = getattr(getattr(op, "settings", None), "_op_id", None) or run_name
        run_id_str = str(run_id)

        self._ops[run_id_str] = op
        self._run_id_to_key[run_id_str] = key

        self._logger.info("Created Pluto Op run '%s' with id %s", run_name, run_id_str)
        return TargetRunId(run_id_str)

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        run_id_str = str(run_id)
        op = self._ops.get(run_id_str)

        if op is None:
            if any(k.endswith(f"::{run_id_str}") for k in self._loaded_run_keys):
                self._logger.info("Run '%s' already loaded; skipping upload", run_id_str)
                return
            raise RuntimeError(f"No active Pluto Op for run {run_id_str}")

        pluto = self._ensure_pluto()

        any_logged = False

        # Accumulate all data to log once at the end (avoids thread exhaustion)
        # Metrics: stream in chunks to bound memory (don't accumulate all 900k steps)
        # This avoids RAM explosion and allows real-time logging as data arrives
        streamed_metrics_count: int = 0
        buffered_metrics: dict[int | None, dict[str, float]] = {}  # step -> {metric_name: value}
        all_files_list: list[dict[str, object]] = []  # List of file batches to log chunked
        all_texts: dict[int | None, dict[str, object]] = {}  # step -> {name: Text}
        all_hists: dict[int | None, dict[str, object]] = {}  # step -> {name: Histogram}
        all_file_items_queued = 0

        last_heartbeat_ts = time.time()

        try:
            for part in run_data:
                for batch in part.to_batches(max_chunksize=self._batch_rows):
                    batch_table = pa.Table.from_batches([batch])
                    df = batch_table.to_pandas(split_blocks=True, self_destruct=True)

                    # -------- Params --------
                    if not self._skip_params:
                        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
                        params_df = df[df["attribute_type"].isin(param_types)]
                        if not params_df.empty:
                            params: dict[str, object] = {}
                            for _, row in params_df.iterrows():
                                name = self._sanitize_attribute_name(row["attribute_path"])
                                atype = row["attribute_type"]
                                if atype == "float" and pd.notna(row.get("float_value")):
                                    params[name] = float(row["float_value"])
                                elif atype == "int" and pd.notna(row.get("int_value")):
                                    params[name] = int(row["int_value"])
                                elif atype == "string" and pd.notna(row.get("string_value")):
                                    params[name] = str(row["string_value"])
                                elif atype == "bool" and pd.notna(row.get("bool_value")):
                                    params[name] = bool(row["bool_value"])
                                elif atype == "datetime" and pd.notna(row.get("datetime_value")):
                                    params[name] = str(row["datetime_value"])
                                elif atype == "string_set" and row.get("string_set_value") is not None:
                                    params[name] = list(row["string_set_value"])
                            if params and hasattr(op, "update_config"):
                                try:
                                    op.update_config(params)
                                    any_logged = True
                                except Exception:
                                    self._logger.debug("op.update_config failed", exc_info=True)

                    # -------- Metrics (float_series) --------
                    if not self._skip_metrics:
                        metrics_df = df[df["attribute_type"] == "float_series"]
                        if not metrics_df.empty:
                            for _, row in metrics_df.iterrows():
                                step = self._convert_step_to_int_optional(row.get("step"), step_multiplier)
                                if step is None or pd.isna(row.get("float_value")):
                                    continue

                                # Standard downsampling
                                if self._log_every > 1 and (step % self._log_every) != 0:
                                    continue

                                attr_name = self._sanitize_attribute_name(row.get("attribute_path"))
                                value = float(row.get("float_value"))

                                # Stream: buffer and flush when buffer gets large to avoid RAM explosion
                                # Note: if multiple values logged for same (step, metric), we keep only the last one
                                # (same behavior as dict overwrite; acceptable since Neptune export should have 1 per step)
                                if step not in buffered_metrics:
                                    buffered_metrics[step] = {}
                                buffered_metrics[step][attr_name] = value

                                # Flush buffer when it reaches threshold
                                if len(buffered_metrics) >= self._metrics_stream_buffer_steps:
                                    for buf_step in sorted(buffered_metrics.keys()):
                                        metrics_dict = buffered_metrics[buf_step]
                                        try:
                                            op.log(metrics_dict, step=buf_step)
                                            streamed_metrics_count += 1
                                            any_logged = True
                                        except Exception:
                                            self._logger.debug(
                                                "Failed to stream metrics for step %s", buf_step, exc_info=True
                                            )
                                    buffered_metrics.clear()

                    # -------- Collect Files / Text / Hists (log at end) --------
                    other_df = df[
                        df["attribute_type"].isin(
                            ["file", "artifact", "file_series", "string_series", "histogram_series"]
                        )
                    ]

                    for _, row in other_df.iterrows():
                        atype = row.get("attribute_type")
                        apath = self._sanitize_attribute_name(row.get("attribute_path"))
                        step = self._convert_step_to_int_optional(row.get("step"), step_multiplier)

                        if step is not None and self._log_every > 1:
                            if atype in ("file_series", "string_series", "histogram_series"):
                                if (step % self._log_every) != 0:
                                    continue

                        # --- Files (collect only) ---
                        if atype in ("file", "artifact", "file_series"):
                            if self._skip_files:
                                continue
                            if self._max_files_per_run and all_file_items_queued >= self._max_files_per_run:
                                continue

                            if isinstance(row.get("file_value"), dict):
                                fv = row.get("file_value")
                                file_path = files_directory / fv.get("path", "")
                                if file_path.exists() and hasattr(pluto, "Artifact"):
                                    try:
                                        art = pluto.Artifact(data=str(file_path), caption=apath)

                                        # Use unique key to prevent overwrites when merging chunks
                                        # (multiple files can have the same apath; counter ensures uniqueness)
                                        key = f"{apath}__{step if step is not None else 'nostep'}__{all_file_items_queued}"
                                        all_files_list.append({key: art})
                                        all_file_items_queued += 1

                                        if all_file_items_queued % 1000 == 0:
                                            now = time.time()
                                            if now - last_heartbeat_ts > 30:
                                                self._logger.info("Collected %d files so far...", all_file_items_queued)
                                                last_heartbeat_ts = now
                                    except Exception:
                                        self._logger.debug("Failed to create artifact", exc_info=True)

                        # --- Text (collect only) ---
                        elif atype == "string_series":
                            if self._skip_text:
                                continue
                            if pd.notna(row.get("string_value")) and hasattr(pluto, "Text"):
                                try:
                                    txt = pluto.Text(str(row.get("string_value")), caption=apath)
                                    if step not in all_texts:
                                        all_texts[step] = {}
                                    all_texts[step][apath] = txt
                                except Exception:
                                    self._logger.debug("Failed to create Text object", exc_info=True)

                        # --- Hist (collect only) ---
                        elif atype == "histogram_series":
                            if self._skip_hist:
                                continue
                            if isinstance(row.get("histogram_value"), dict) and hasattr(pluto, "Histogram"):
                                try:
                                    hist = pluto.Histogram(row.get("histogram_value"))
                                    if step not in all_hists:
                                        all_hists[step] = {}
                                    all_hists[step][apath] = hist
                                except Exception:
                                    self._logger.debug("Failed to create Histogram object", exc_info=True)

                    del df
                    del batch_table
                    gc.collect()

            # -------- Log all collected data at the end --------

            # Flush any remaining buffered metrics
            if buffered_metrics:
                for buf_step in sorted(buffered_metrics.keys()):
                    metrics_dict = buffered_metrics[buf_step]
                    try:
                        op.log(metrics_dict, step=buf_step)
                        streamed_metrics_count += 1
                        any_logged = True
                    except Exception:
                        self._logger.debug("Failed to log final metrics batch for step %s", buf_step, exc_info=True)
                buffered_metrics.clear()

            # Log files in chunks (50-200 at a time) to avoid overload + 502s
            # Each chunk: log, flush, short sleep
            file_chunk_size = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_FILE_CHUNK_SIZE", "100"))
            if file_chunk_size < 1:
                file_chunk_size = 100
            file_sleep_seconds = float(os.getenv("NEPTUNE_EXPORTER_PLUTO_FILE_CHUNK_SLEEP", "0.5"))

            for i in range(0, len(all_files_list), file_chunk_size):
                chunk = all_files_list[i : i + file_chunk_size]
                if chunk:
                    try:
                        # Merge all artifacts in chunk into one dict
                        files_dict: dict[str, object] = {}
                        for item_dict in chunk:
                            files_dict.update(item_dict)
                        
                        self._logger.info(
                            "Pluto: logging file chunk %d-%d (%d artifacts)...",
                            i, i + len(chunk), len(files_dict)
                        )
                        op.log(files_dict)
                        self._maybe_flush(op)
                        any_logged = True
                        
                        if file_sleep_seconds > 0:
                            time.sleep(file_sleep_seconds)
                    except Exception:
                        self._logger.exception("Failed to log file chunk %d-%d (non-fatal)", i, i + len(chunk))

            # Log texts by step
            for step, text_dict in sorted(all_texts.items()):
                if text_dict:
                    try:
                        op.log(text_dict, step=step)
                        any_logged = True
                    except Exception:
                        self._logger.debug("Failed to log text batch for step %s", step, exc_info=True)

            # Log hists by step
            for step, hist_dict in sorted(all_hists.items()):
                if hist_dict:
                    try:
                        op.log(hist_dict, step=step)
                        any_logged = True
                    except Exception:
                        self._logger.debug("Failed to log histogram batch for step %s", step, exc_info=True)

            self._maybe_flush(op)

            self._logger.info(
                "Pluto: streamed and logged %d metric steps (log_every=%s), "
                "%d file artifacts, %d text steps, %d histogram steps",
                streamed_metrics_count, self._log_every,
                all_file_items_queued, len(all_texts), len(all_hists)
            )

        except KeyboardInterrupt:
            self._logger.warning("Pluto loader interrupted by user (Ctrl+C)")
            self._finish_op(op, run_id_str, interrupted=True)
            # Don't raise - let it exit cleanly so status update completes
            return
        finally:
            # Only finish if not already finished by KeyboardInterrupt handler
            if run_id_str in self._ops:
                self._finish_op(op, run_id_str)

        if not any_logged:
            raise RuntimeError(f"Pluto loader finished run {run_id_str} but did not log any data.")

        key = self._run_id_to_key.get(run_id_str)
        if key:
            self._mark_run_loaded(key)

        self._logger.info("Uploaded data for Pluto Op run %s", run_id_str)

    def _finish_op(self, op: object, run_id_str: str, interrupted: bool = False) -> None:
        """Finish/close a Pluto Op to release threads and flush pending uploads."""
        try:
            if hasattr(op, "finish"):
                # If interrupted (Ctrl+C), mark run as canceled (SIGINT=2)
                import signal
                op.finish(code=signal.SIGINT if interrupted else None)
            elif hasattr(op, "close"):
                op.close()
        except Exception:
            self._logger.debug("Failed to finish Pluto Op %s", run_id_str, exc_info=True)
        finally:
            self._ops.pop(run_id_str, None)
            self._run_id_to_key.pop(run_id_str, None)
            gc.collect()
