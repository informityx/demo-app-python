"""
In-memory job store for async CV evaluation.
Jobs expire after a short TTL to avoid unbounded growth.
"""
import threading
import time
import uuid
from typing import Any, Dict, Optional

# job_id -> {status, result?, error?, created_at}
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
JOB_TTL_SECONDS = 3600  # 1 hour


def create_job() -> str:
    with _lock:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
        return job_id


def set_running(job_id: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"


def set_completed(job_id: str, result: Dict[str, Any]) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["error"] = None


def set_failed(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = error
            _jobs[job_id]["result"] = None


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        _expire_old()
        return _jobs.get(job_id)


def _expire_old() -> None:
    now = time.time()
    to_remove = [jid for jid, data in _jobs.items() if now - data["created_at"] > JOB_TTL_SECONDS]
    for jid in to_remove:
        del _jobs[jid]
