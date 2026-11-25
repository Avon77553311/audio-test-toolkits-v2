"""持久化、FIFO、按资源限流的统一后台任务调度器。"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

import requests

from billing_service import BillingError, BillingService, BillingUnavailable

ACTIVE_STATUSES = frozenset({"queued", "running"})
FINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
RESOURCE_TYPES = frozenset({"cpu", "gpu"})
_BILLING_EXEMPT_KEY = "_server_billing_exempt"
LOGGER = logging.getLogger(__name__)


GLOBAL_LIMITS = {"cpu": 3, "gpu": 5}

_current_context: contextvars.ContextVar["TaskContext | None"] = contextvars.ContextVar(
    "task_context", default=None
)


def _http_status(exc: Exception, text: str) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    lowered = text.lower()
    for candidate in (401, 403, 429, 502, 503, 504):
        if (
            f"http {candidate}" in lowered
            or f"{candidate} client error" in lowered
            or f"{candidate} server error" in lowered
        ):
            return candidate
    return None


class NoUsableChordResult(RuntimeError):
    """A safe, user-actionable chord-analysis terminal condition."""


class NoUsableSpeechResult(RuntimeError):
    """A safe, user-actionable speech-to-text terminal condition."""


class NoUsableSongInfoResult(RuntimeError):
    """A safe, user-actionable song-information terminal condition."""


def _public_task_error(exc: Exception) -> str:
    """Map every task module's internal failures to a small, non-sensitive whitelist."""
    text = str(exc)
    lowered = text.lower()
    status = _http_status(exc, text)
    if isinstance(exc, NoUsableChordResult):
        return "处理失败：未识别到有效和弦或低音，请检查音频内容或调整参数"
    if isinstance(exc, NoUsableSpeechResult):
        return "处理失败：未检测到可识别的语音内容，请检查音频是否清晰或为静音"
    if isinstance(exc, NoUsableSongInfoResult):
        return "处理失败：未检测到可分析的音频内容，请检查音频是否为静音"
    if isinstance(exc, ResultSerializationError):
        return "处理失败：任务结果格式无效，请联系管理员并提供任务 ID"
    if (
        status in {401, 403}
        or "api key" in lowered
        or "权限" in text
        and "runpod" in lowered
    ):
        return "处理失败：云端服务认证或权限配置异常，请联系管理员"
    if status == 429:
        return "处理失败：云端服务请求过于频繁，请稍后重试"
    if status in {502, 503, 504}:
        return "处理失败：云端服务暂时不可用，请稍后重试"
    if isinstance(exc, (TimeoutError, requests.Timeout)) or "超时" in text:
        if "回调" in text or "回传" in text:
            return "处理失败：RunPod 结果回传超时，请稍后重试"
        if "子进程" in text:
            return "处理失败：本地处理超时，请稍后重试"
        if "轮询" in text or "状态" in text or "runpod" in lowered:
            return "处理失败：RunPod 状态查询超时，请稍后重试"
        return "处理失败：任务执行超时，请稍后重试"
    if "任务已取消" in text:
        return "任务已取消"
    if "底层" in text and "状态丢失" in text:
        return "处理失败：底层任务状态丢失，请重新提交"
    if "底层任务执行失败" in text:
        return "处理失败：任务执行失败，请稍后重试"
    if "子进程" in text and any(
        marker in text for marker in ("异常退出", "失败", "无法启动")
    ):
        return "处理失败：本地处理进程异常，请稍后重试"
    if "回调" in text or "回传" in text:
        return "处理失败：RunPod 结果回传失败，请稍后重试"
    if "轨道" in text or "压缩包" in text:
        return "处理失败：RunPod 返回结果校验失败"
    if "提交失败" in text and any(
        marker in lowered for marker in ("runpod", "seed-vc", "stem")
    ):
        return "处理失败：RunPod 任务提交失败，请稍后重试"
    if any(marker in lowered for marker in ("runpod", "seed-vc")) and any(
        marker in text for marker in ("任务失败", "处理失败", "返回结果", "响应")
    ):
        return "处理失败：RunPod 任务执行失败，请稍后重试"
    if isinstance(exc, ValueError):
        return "处理失败：输入或任务配置无效"
    return "处理失败：系统异常，请联系管理员并提供任务 ID"


class TaskError(RuntimeError):
    pass


class ActiveTaskExists(TaskError):
    pass


class AccountNotActive(TaskError):
    pass


class TaskNotFound(TaskError):
    pass


class TaskNotCancelable(TaskError):
    pass


class LegacyTaskCancelled(TaskError):
    """底层执行器已进入 cancelled 终态。"""


class ReconciliationPending(TaskError):
    """远端状态已取得，但结果尚不足以安全完成本地任务。"""


class RemoteTaskFailed(TaskError):
    """远端明确报告终态失败。"""


class ResultSerializationError(TaskError):
    """执行器结果无法安全保存为 JSON。"""


def _json_safe(value: Any) -> Any:
    """Recursively normalize common result types without stringifying unknown objects."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON object key must be str, got {type(key).__name__}"
                )
            normalized[key] = _json_safe(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def _json_safe_result(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ResultSerializationError("handler result must be a mapping")
    try:
        result = _json_safe(value)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ResultSerializationError(
            "handler result is not JSON serializable"
        ) from exc
    return result


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_not_cancelled() -> None:
    """RunPod 客户端在 POST /run 前调用，关闭取消与计费之间的竞态窗口。"""
    context = _current_context.get()
    if context is not None:
        context.begin_runpod_submission()
    elif subprocess_cancel_requested():
        raise TaskNotCancelable("任务已取消，未提交 RunPod")


def notify_runpod_started(job_id: str) -> None:
    """供 RunPod 客户端在拿到远端 job id 的精确时刻报告不可取消边界。"""
    folder = os.environ.get('STUDIO_TASK_FOLDER')
    if folder:
        from pathlib import Path
        # Private reconciliation receipt; never include API credentials.
        receipt={'job_id': str(job_id), 'state': 'accepted', 'created': utcnow()}
        Path(folder, 'remote-receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        with Path(folder, 'remote-receipts.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(receipt)+'\n')
    context = _current_context.get()
    if context is not None:
        context.mark_runpod_started(job_id)
        return
    event_file = os.environ.get("AUDIO_TOOLKIT_TASK_EVENT_FILE")
    if event_file:
        # 子进程仅追加一行小型 JSON；父进程负责验证并写数据库。
        payload = (
            json.dumps(
                {"event": "runpod_started", "job_id": str(job_id)}, ensure_ascii=True
            )
            + "\n"
        )
        fd = os.open(event_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)


def subprocess_cancel_requested() -> bool:
    """子进程在 RunPod /run 前通过父进程维护的取消标记做最后检查。"""
    cancel_file = os.environ.get("AUDIO_TOOLKIT_TASK_CANCEL_FILE")
    return bool(cancel_file and os.path.exists(cancel_file))


class TaskRepository:
    """Supabase PostgREST 持久层；所有用户过滤均在请求中下推。"""

    columns = (
        "id,user_id,task_type,resource_type,status,payload,result,progress,stage,message,"
        "runpod_job_id,runpod_started_at,cancel_requested_at,worker_id,lease_token,"
        "lease_expires_at,heartbeat_at,execution_started_at,submission_token,submission_started_at,"
        "monitor_token,monitor_lease_expires_at,parent_id,created_at,started_at,completed_at,updated_at"
    )

    def __init__(
        self,
        base_url: str,
        service_key: str,
        timeout: float = 8.0,
        session: requests.Session | None = None,
    ):
        if not base_url or not service_key:
            raise TaskError("Supabase 任务存储尚未配置")
        self.endpoint = base_url.rstrip("/") + "/rest/v1/tasks"
        self.settings_endpoint = base_url.rstrip("/") + "/rest/v1/runtime_settings"
        self.rpc_base = base_url.rstrip("/") + "/rest/v1/rpc"
        self.timeout = timeout
        self.http = session or requests.Session()
        headers = {
            "apikey": service_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if service_key.count(".") == 2:
            headers["Authorization"] = f"Bearer {service_key}"
        self.http.headers.update(headers)

    def _request(
        self,
        method: str,
        *,
        params=None,
        payload=None,
        prefer=None,
        endpoint=None,
        retry_safe: bool = False,
    ):
        headers = {"Prefer": prefer} if prefer else None
        attempts = 3 if retry_safe or method.upper() == "GET" else 1
        response = None
        for attempt in range(attempts):
            try:
                response = self.http.request(
                    method,
                    endpoint or self.endpoint,
                    params=params,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt + 1 >= attempts:
                    raise TaskError("任务服务暂时不可用") from exc
                time.sleep(0.1 * (2**attempt))
            except requests.RequestException as exc:
                raise TaskError("任务服务暂时不可用") from exc
        if response is None:
            raise TaskError("任务服务暂时不可用")
        if response.status_code >= 400:
            text = response.text[:500]
            if "account_not_active" in text:
                raise AccountNotActive("账号不存在或已注销，无法创建任务")
            if "task_not_found" in text:
                raise TaskNotFound("任务不存在")
            if "task_not_cancelable" in text or "task_not_submittable" in text:
                raise TaskNotCancelable("任务已结束或已进入 RunPod，不能取消")
            if response.status_code == 409 or "tasks_one_active_per_user" in text:
                raise ActiveTaskExists("每个用户同时只能有一个排队中或运行中的任务")
            raise TaskError(f"任务存储请求失败（HTTP {response.status_code}）")
        if response.status_code == 204 or not response.content:
            return []
        try:
            data = response.json()
        except ValueError as exc:
            raise TaskError("任务服务返回了无效响应") from exc
        return data if isinstance(data, list) else [data]

    def create(
        self,
        user_id: str,
        task_type: str,
        resource_type: str,
        payload: Mapping[str, Any],
        task_id: str | None = None,
    ) -> dict[str, Any]:
        if resource_type not in RESOURCE_TYPES:
            raise ValueError("resource_type 仅支持 cpu 或 gpu")
        task_uuid = task_id or str(uuid.uuid4())
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/create_user_task",
            payload={
                "p_id": task_uuid,
                "p_user_id": str(user_id),
                "p_task_type": task_type,
                "p_resource_type": resource_type,
                "p_payload": dict(payload),
            },
            retry_safe=True,
        )
        if not rows:
            raise TaskError("任务创建失败")
        return rows[0]

    def get(self, task_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        params = {"id": f"eq.{task_id}", "select": self.columns, "limit": "1"}
        if user_id is not None:
            params["user_id"] = f"eq.{user_id}"
        rows = self._request("GET", params=params)
        return rows[0] if rows else None

    def owns_output(self, user_id: str, output_path: str) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/user_owns_task_output",
            payload={"p_user_id": str(user_id), "p_output_path": str(output_path)},
            retry_safe=True,
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def list_queued(self) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            params={
                "status": "eq.queued",
                "select": self.columns,
                "order": "created_at.asc,id.asc",
            },
        )

    def list_recoverable(self) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            params={
                "status": "in.(queued,running)",
                "select": self.columns,
                "order": "created_at.asc,id.asc",
            },
        )

    def list_active(self, resource_type: str | None = None) -> list[dict[str, Any]]:
        params = {
            "status": "in.(queued,running)",
            "select": self.columns,
            "order": "created_at.asc,id.asc",
        }
        if resource_type:
            params["resource_type"] = f"eq.{resource_type}"
        return self._request("GET", params=params)

    def find_active(
        self, user_id: str, task_type: str | None = None
    ) -> dict[str, Any] | None:
        params = {
            "user_id": f"eq.{user_id}",
            "status": "in.(queued,running)",
            "select": self.columns,
            "order": "created_at.desc,id.desc",
            "limit": "1",
        }
        if task_type:
            params["task_type"] = f"eq.{task_type}"
        rows = self._request("GET", params=params)
        return rows[0] if rows else None

    def update(
        self,
        task_id: str,
        values: Mapping[str, Any],
        *,
        expected_status: str | None = None,
        require_no_runpod: bool = False,
        expected_lease_token: str | None = None,
    ) -> dict[str, Any] | None:
        params = {"id": f"eq.{task_id}", "select": self.columns}
        if expected_status:
            params["status"] = f"eq.{expected_status}"
        if require_no_runpod:
            params["runpod_job_id"] = "is.null"
        if expected_lease_token:
            params["lease_token"] = f"eq.{expected_lease_token}"
        body = dict(values)
        body.setdefault("updated_at", utcnow())
        rows = self._request(
            "PATCH", params=params, payload=body, prefer="return=representation"
        )
        return rows[0] if rows else None

    def fetch_runtime_limits(self) -> dict[str, int]:
        """读取 service-only 运行时设置；数据库 claim 仍是最终并发闸门。"""
        rows = self._request(
            "GET",
            endpoint=self.settings_endpoint,
            params={
                "setting_key": "in.(cpu_concurrency,gpu_concurrency)",
                "select": "setting_key,value",
            },
        )
        limits = dict(GLOBAL_LIMITS)
        for row in rows:
            key = {"cpu_concurrency": "cpu", "gpu_concurrency": "gpu"}.get(
                row.get("setting_key")
            )
            value = row.get("value")
            if key and isinstance(value, int) and 1 <= value <= 20:
                limits[key] = value
        return limits

    def claim_next(
        self, resource_type: str, worker_id: str, cpu_limit: int = 3, gpu_limit: int = 5
    ) -> dict[str, Any] | None:
        """由数据库事务原子检查全局配额并领取该资源最早的 queued 任务。"""
        if resource_type not in RESOURCE_TYPES:
            raise ValueError("resource_type 仅支持 cpu 或 gpu")
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/claim_next_task",
            payload={
                "p_resource_type": resource_type,
                "p_worker_id": worker_id,
                "p_cpu_limit": int(cpu_limit),
                "p_gpu_limit": int(gpu_limit),
            },
        )
        return rows[0] if rows else None

    def recover_worker(self, worker_id: str) -> list[dict[str, Any]]:
        """仅恢复同一稳定 worker_id 留下的 running；绝不改写其他 worker。"""
        return self._request(
            "POST",
            endpoint=f"{self.rpc_base}/recover_worker_tasks",
            payload={"p_worker_id": worker_id},
        )

    def reconcile_tasks(
        self, worker_id: str, timeout_seconds: int, queue_alert_seconds: int
    ) -> list[dict[str, Any]]:
        """原子回收失联本地任务，并取得仅需恢复轮询的 RunPod 任务。"""
        self._request(
            "POST",
            endpoint=f"{self.rpc_base}/stage6_reconcile_tasks",
            payload={
                "p_worker_id": worker_id,
                "p_running_timeout_seconds": int(timeout_seconds),
                "p_queue_alert_seconds": int(queue_alert_seconds),
            },
        )
        return self._request(
            "POST",
            endpoint=f"{self.rpc_base}/claim_runpod_monitors",
            payload={"p_worker_id": worker_id, "p_limit": 20},
        )

    def observability_snapshot(self) -> dict[str, Any]:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/stage6_observability_snapshot",
            payload={},
        )
        return rows[0] if rows and isinstance(rows[0], dict) else {}

    def claim(self, task_id: str, worker_id: str) -> dict[str, Any] | None:
        """兼容测试/管理用途；生产调度只使用 claim_next 原子 RPC。"""
        return self.update(
            task_id,
            {
                "status": "running",
                "stage": "running",
                "message": "任务执行中",
                "worker_id": worker_id,
                "started_at": utcnow(),
            },
            expected_status="queued",
        )

    def requeue(self, task_id: str) -> dict[str, Any] | None:
        return self.update(
            task_id,
            {
                "status": "queued",
                "stage": "queued",
                "worker_id": None,
                "message": "服务重启，任务已恢复排队",
            },
            expected_status="running",
            require_no_runpod=True,
        )

    def begin_runpod_submission(
        self, task_id: str, worker_id: str, lease_token: str
    ) -> str:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/begin_runpod_submission",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
            },
        )
        if not rows or not rows[0]:
            raise TaskNotCancelable("任务已取消，未提交 RunPod")
        value = rows[0]
        return str(value if not isinstance(value, dict) else next(iter(value.values())))

    def complete_runpod_submission(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        submission_token: str,
        job_id: str,
    ) -> dict[str, Any] | None:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/complete_runpod_submission",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
                "p_submission_token": submission_token,
                "p_job_id": str(job_id),
            },
        )
        return rows[0] if rows else None

    def begin_remote_job_submission(
        self, task_id: str, segment_key: str, worker_id: str, lease_token: str
    ) -> str:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/begin_remote_job_submission",
            payload={
                "p_task_id": task_id,
                "p_segment_key": segment_key,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
            },
        )
        if not rows or not rows[0]:
            raise TaskNotCancelable("任务已取消，未提交 RunPod")
        value = rows[0]
        return str(value if not isinstance(value, dict) else next(iter(value.values())))

    def complete_remote_job_submission(
        self, task_id: str, segment_key: str, submission_token: str, job_id: str
    ) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/complete_remote_job_submission",
            payload={
                "p_task_id": task_id,
                "p_segment_key": segment_key,
                "p_submission_token": submission_token,
                "p_job_id": str(job_id),
            },
            retry_safe=True,
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def abort_remote_job_submission(
        self, task_id: str, segment_key: str, submission_token: str
    ) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/abort_remote_job_submission",
            payload={
                "p_task_id": task_id,
                "p_segment_key": segment_key,
                "p_submission_token": submission_token,
            },
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def finish_remote_job(self, task_id: str, segment_key: str, status: str) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/finish_remote_job",
            payload={
                "p_task_id": task_id,
                "p_segment_key": segment_key,
                "p_status": status,
            },
            retry_safe=True,
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def abort_runpod_submission(
        self, task_id: str, worker_id: str, lease_token: str, submission_token: str
    ) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/abort_runpod_submission",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
                "p_submission_token": submission_token,
            },
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def admit_execution(self, task_id: str, worker_id: str, lease_token: str) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/admit_task_execution",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
            },
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def renew_lease(self, task_id: str, worker_id: str, lease_token: str) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/renew_task_lease",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
            },
        )
        if not rows:
            return False
        value = rows[0]
        return bool(
            value if not isinstance(value, dict) else next(iter(value.values()), False)
        )

    def finalize(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        result: Mapping[str, Any],
        message: str,
    ) -> dict[str, Any] | None:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/finalize_task",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
                "p_result": dict(result),
                "p_message": message,
            },
        )
        return rows[0] if rows else None

    def register_stem_callback(
        self, task_id: str, token: str, output_path: str, expires_at: str
    ) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/register_stem_callback",
            payload={
                "p_task_id": task_id,
                "p_token": token,
                "p_output_path": output_path,
                "p_expires_at": expires_at,
            },
            retry_safe=True,
        )
        return bool(rows)

    def get_stem_callback(self, task_id: str) -> dict[str, Any] | None:
        params = {"task_id": f"eq.{task_id}", "select": "*", "limit": "1"}
        rows = self._request(
            "GET",
            endpoint=self.endpoint.replace("/tasks", "/stem_callbacks"),
            params=params,
        )
        return rows[0] if rows else None

    def receive_stem_callback(
        self, task_id: str, token: str, error: str | None = None
    ) -> bool:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/receive_stem_callback",
            payload={"p_task_id": task_id, "p_token": token, "p_error": error},
            retry_safe=True,
        )
        return bool(rows)

    def complete_gpu(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        unit_price: int,
        module_name: str,
        result: Mapping[str, Any],
        message: str,
    ) -> dict[str, Any]:
        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/complete_gpu_task",
            payload={
                "p_task_id": task_id,
                "p_worker_id": worker_id,
                "p_lease_token": lease_token,
                "p_unit_price": int(unit_price),
                "p_module_name": module_name,
                "p_result": dict(result),
                "p_message": message,
            },
        )
        return rows[0] if rows and isinstance(rows[0], dict) else {}

    def request_cancel(self, task_id: str, user_id: str) -> dict[str, Any]:
        # 分段任务进入远端边界后同样禁止假取消；最终裁决仍由 RPC 原子完成。
        try:
            remote_jobs = self._request(
                "GET",
                endpoint=self.endpoint.replace(
                    "/rest/v1/tasks", "/rest/v1/task_remote_jobs"
                ),
                params={
                    "task_id": f"eq.{task_id}",
                    "select": "id",
                    "limit": "1",
                    "status": "in.(submitting,running)",
                },
            )
        except TaskError:
            remote_jobs = []
        if remote_jobs:
            raise TaskNotCancelable("任务已进入 RunPod，不能取消")

        rows = self._request(
            "POST",
            endpoint=f"{self.rpc_base}/request_task_cancel",
            payload={"p_task_id": task_id, "p_user_id": user_id},
        )
        if not rows:
            raise TaskNotCancelable("任务已结束或已进入 RunPod，不能取消")
        return rows[0]


class TaskContext:
    def __init__(
        self,
        scheduler: "TaskScheduler",
        task_id: str,
        cancel_event: threading.Event,
        lease_token: str | None = None,
    ):
        self.scheduler = scheduler
        self.task_id = task_id
        self.cancel_event = cancel_event
        self.lease_token = lease_token
        self.submission_token: str | None = None
        self.segment_key: str | None = None
        self.segment_submission_token: str | None = None

    def set_segment(self, key: str | None) -> None:
        self.segment_key = key
        self.segment_submission_token = None

    def finish_segment(self, status: str) -> None:
        if not self.segment_key:
            return
        finish = getattr(self.scheduler.repository, "finish_remote_job", None)
        if finish is not None:
            finish(self.task_id, self.segment_key, status)
        self.segment_submission_token = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise LegacyTaskCancelled("任务已取消")

    def update(
        self,
        *,
        progress: int | None = None,
        stage: str | None = None,
        message: str | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if progress is not None:
            values["progress"] = max(0, min(100, int(progress)))
        if stage is not None:
            values["stage"] = stage
        if message is not None:
            values["message"] = message
        if values:
            if (
                isinstance(self.scheduler.repository, TaskRepository)
                and self.lease_token
            ):
                if not self.scheduler.repository.renew_lease(
                    self.task_id, self.scheduler.worker_id, self.lease_token
                ):
                    self.cancel_event.set()
                    current = self.scheduler.repository.get(self.task_id)
                    if current and current.get("cancel_requested_at"):
                        raise LegacyTaskCancelled("任务已取消")
                    raise TaskNotCancelable("任务租约已失效")
                updated = self.scheduler.repository.update(
                    self.task_id,
                    values,
                    expected_status="running",
                    expected_lease_token=self.lease_token,
                )
            else:
                updated = self.scheduler.repository.update(
                    self.task_id, values, expected_status="running"
                )
            if not updated:
                current = self.scheduler.repository.get(self.task_id)
                if current and (
                    current.get("status") == "cancelled"
                    or current.get("cancel_requested_at")
                ):
                    self.cancel_event.set()
                    raise LegacyTaskCancelled("任务已取消")

    def begin_runpod_submission(self) -> None:
        """在外部 POST 前原子关闭取消窗口；响应未知时保留该边界供人工对账。"""
        self.raise_if_cancelled()
        if self.segment_key:
            begin = getattr(
                self.scheduler.repository, "begin_remote_job_submission", None
            )
            if begin is not None and self.lease_token:
                self.segment_submission_token = begin(
                    self.task_id,
                    self.segment_key,
                    self.scheduler.worker_id,
                    self.lease_token,
                )
        else:
            begin = getattr(self.scheduler.repository, "begin_runpod_submission", None)
            if begin is not None and self.lease_token:
                self.submission_token = begin(
                    self.task_id, self.scheduler.worker_id, self.lease_token
                )

    def abort_runpod_submission(self) -> None:
        """仅在确认远端请求尚未发出时重新开放取消窗口。"""
        if self.segment_key:
            if self.segment_submission_token:
                abort = getattr(
                    self.scheduler.repository, "abort_remote_job_submission", None
                )
                if abort is not None:
                    abort(self.task_id, self.segment_key, self.segment_submission_token)
            self.segment_submission_token = None
            return

        if not self.submission_token or not self.lease_token:
            return
        abort = getattr(self.scheduler.repository, "abort_runpod_submission", None)
        if abort is not None and abort(
            self.task_id,
            self.scheduler.worker_id,
            self.lease_token,
            self.submission_token,
        ):
            self.submission_token = None

    def mark_runpod_started(self, job_id: str) -> None:
        with self.scheduler._condition:
            if self.cancel_event.is_set():
                raise TaskNotCancelable("任务已取消，未提交 RunPod")

            if self.segment_key and self.segment_submission_token:
                complete = getattr(
                    self.scheduler.repository, "complete_remote_job_submission", None
                )
                if complete and complete(
                    self.task_id,
                    self.segment_key,
                    self.segment_submission_token,
                    job_id,
                ):
                    return
                raise TaskNotCancelable("任务已取消或租约已失效")

            # 原生任务提交边界
            complete = getattr(
                self.scheduler.repository,
                "complete_runpod_submission",
                getattr(self.scheduler.repository, "mark_runpod_started", None),
            )
            try:
                if complete == getattr(
                    self.scheduler.repository, "complete_runpod_submission", None
                ):
                    updated = complete(
                        self.task_id,
                        self.scheduler.worker_id,
                        self.lease_token,
                        self.submission_token,
                        job_id,
                    )
                else:
                    updated = complete(self.task_id, job_id)
            except (TypeError, AttributeError):
                # 兼容内存/旧版仓库
                updated = self.scheduler.repository.update(
                    self.task_id,
                    {"runpod_job_id": job_id, "runpod_started_at": utcnow()},
                    expected_status="running",
                )

            if not updated:
                self.cancel_event.set()
                raise TaskNotCancelable("任务租约已失效，RunPod 结果必须进入对账")


Handler = Callable[[TaskContext, Mapping[str, Any]], Optional[Mapping[str, Any]]]


class TaskScheduler:
    """单进程协调器；数据库是真实状态源，CPU/GPU 分别全局 FIFO 限流。"""

    def __init__(
        self,
        repository,
        cpu_concurrency: int = 3,
        gpu_concurrency: int = 5,
        estimated_seconds: Mapping[str, int] | None = None,
        worker_id: str | None = None,
        autostart: bool = True,
        billing_service: BillingService | None = None,
        reconcile_interval_seconds: float | None = None,
        running_timeout_seconds: int | None = None,
        queue_alert_seconds: int | None = None,
    ):
        self.repository = repository
        self.billing_service = billing_service
        self._billing_enforced = billing_service is not None or isinstance(
            repository, TaskRepository
        )
        requested_limits = {
            "cpu": min(20, max(1, int(cpu_concurrency))),
            "gpu": min(20, max(1, int(gpu_concurrency))),
        }
        self.limits = requested_limits
        self._limits_refresh_seconds = 2.0
        self._last_limits_refresh = 0.0
        self.estimated_seconds = {"cpu": 60, "gpu": 180, **(estimated_seconds or {})}
        if worker_id:
            self.worker_id = worker_id
        else:
            worker_base = os.environ.get("TASK_WORKER_ID") or "worker"
            instance_id = (
                os.environ.get("RENDER_INSTANCE_ID")
                or os.environ.get("HOSTNAME")
                or "local"
            )
            # TASK_WORKER_ID 仅作逻辑前缀；实例与进程后缀防止多 worker 误认彼此租约。
            self.worker_id = f"{worker_base}:{instance_id}:{os.getpid()}"
        self.reconcile_interval_seconds = max(
            0.1,
            float(
                reconcile_interval_seconds
                if reconcile_interval_seconds is not None
                else os.environ.get("TASK_RECONCILE_INTERVAL_SECONDS", 30)
            ),
        )
        self.running_timeout_seconds = max(
            1,
            int(
                running_timeout_seconds
                if running_timeout_seconds is not None
                else os.environ.get("TASK_RUNNING_TIMEOUT_SECONDS", 1800)
            ),
        )
        self.queue_alert_seconds = max(
            1,
            int(
                queue_alert_seconds
                if queue_alert_seconds is not None
                else os.environ.get("TASK_QUEUE_ALERT_SECONDS", 600)
            ),
        )
        self.observability_snapshot: dict[str, Any] = {}
        self.handlers: dict[str, Handler] = {}
        self.monitors: dict[str, Handler] = {}
        self._runtime_handlers: dict[str, Handler] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancelled_before_launch: set[str] = set()
        self._monitoring_tasks: set[str] = set()
        self._condition = threading.Condition(threading.RLock())
        # 线程池按允许运行时上限预留；实际执行数由数据库 claim 原子限制。
        self._executor = ThreadPoolExecutor(
            max_workers=40, thread_name_prefix="persistent-task"
        )
        self._stopped = False
        self._started = False
        self._threads: list[threading.Thread] = []
        if autostart:
            self.start()

    def register(
        self, task_type: str, handler: Handler, monitor: Handler | None = None
    ) -> None:
        self.handlers[task_type] = handler
        if monitor:
            self.monitors[task_type] = monitor
        with self._condition:
            self._condition.notify_all()

    def apply_runtime_limits(
        self, cpu_concurrency: int | None, gpu_concurrency: int | None
    ) -> dict[str, int]:
        """应用已由数据库验证并持久化的设置，并唤醒当前实例。"""
        with self._condition:
            for resource, value in (("cpu", cpu_concurrency), ("gpu", gpu_concurrency)):
                if value is not None:
                    value = int(value)
                    if not 1 <= value <= 20:
                        raise ValueError("并发必须介于 1 和 20")
                    self.limits[resource] = value
            self._last_limits_refresh = time.monotonic()
            self._condition.notify_all()
            return dict(self.limits)

    def refresh_runtime_limits(self, force: bool = False) -> dict[str, int]:
        """周期读取共享设置；失败时保留旧值，数据库 claim 仍会 fail-safe 限流。"""
        now = time.monotonic()
        if not force and now - self._last_limits_refresh < self._limits_refresh_seconds:
            return dict(self.limits)
        fetch = getattr(self.repository, "fetch_runtime_limits", None)
        if fetch is None:
            self._last_limits_refresh = now
            return dict(self.limits)
        limits = fetch()
        return self.apply_runtime_limits(limits.get("cpu"), limits.get("gpu"))

    def notify_task_cancelled(self, task_id: str) -> None:
        """管理 RPC 已取消本实例运行的本地任务时，通知执行器尽快退出。"""
        with self._condition:
            event = self._cancel_events.get(str(task_id))
            if event:
                event.set()
            self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            try:
                self._recover_locked()
            except TaskError:
                # Supabase 短暂不可用不应阻止 Web 进程启动；调度循环会重试。
                pass
            dispatcher = threading.Thread(
                target=self._dispatch_loop, name="task-dispatcher", daemon=True
            )
            maintenance = threading.Thread(
                target=self._maintenance_loop, name="task-maintenance", daemon=True
            )
            self._threads = [dispatcher, maintenance]
            dispatcher.start()
            maintenance.start()
            self._condition.notify_all()

    def _recover_locked(self) -> None:
        """启动时仅恢复本 worker 的遗留任务；此方法不会在调度循环中调用。"""
        for task in self.repository.recover_worker(self.worker_id):
            if task["task_type"] in self.monitors:
                self._launch_locked(task, monitor=True)

    def submit(
        self,
        user_id: str,
        task_type: str,
        resource_type: str,
        payload: Mapping[str, Any],
        handler: Handler | None = None,
        *,
        billing_exempt: bool = False,
        submission_key: str | None = None,
    ) -> dict[str, Any]:
        submission_id = (str(uuid.uuid5(uuid.NAMESPACE_URL,
                         f"audio-toolkit:{user_id}:{task_type}:{submission_key}"))
                         if submission_key else None)
        exempt = bool(billing_exempt and resource_type == "gpu")
        stored_payload = dict(payload)
        stored_payload.pop(_BILLING_EXEMPT_KEY, None)
        if exempt:
            stored_payload[_BILLING_EXEMPT_KEY] = True
        if resource_type == "gpu" and self._billing_enforced and not exempt:
            if self.billing_service is None:
                raise BillingUnavailable("billing unavailable")
            self.billing_service.check_gpu_eligibility(
                user_id, task_type, resource_type
            )
        with self._condition:
            existing = self.repository.get(submission_id, user_id) if submission_id else None
            if existing:
                return self.describe(existing)
            try:
                task = self.repository.create(
                    user_id, task_type, resource_type, stored_payload,
                    **({"task_id": submission_id} if submission_id else {}),
                )
            except ActiveTaskExists:
                existing = self.repository.get(submission_id, user_id) if submission_id else None
                if existing:
                    return self.describe(existing)
                raise
            if handler and (not submission_id or task.get("payload") == stored_payload):
                self._runtime_handlers[task["id"]] = handler
            self._condition.notify_all()
        return self.describe(task)

    def get(self, task_id: str, user_id: str) -> dict[str, Any] | None:
        task = self.repository.get(task_id, user_id)
        return self.describe(task) if task else None

    def get_active(
        self, user_id: str, task_type: str | None = None
    ) -> dict[str, Any] | None:
        task = self.repository.find_active(user_id, task_type)
        return self.describe(task) if task else None

    def cancel(self, task_id: str, user_id: str) -> dict[str, Any]:
        task = self.repository.request_cancel(task_id, user_id)
        with self._condition:
            event = self._cancel_events.get(task_id)
            if event:
                event.set()
            elif task.get("lease_token") and task.get("worker_id") == self.worker_id:
                # claim 已落库但 dispatcher 尚未创建 event 时保存取消信号，
                # 由 _launch_locked 继承，避免已取消任务仍开始执行。
                self._cancelled_before_launch.add(task_id)
            self._condition.notify_all()
        return self.describe(task)

    def describe(self, task: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(task)
        status = result.get("status")
        result["task_id"] = result.get("id")
        result["success"] = status not in {"failed", "cancelled"}

        # 区分租约过期
        is_running = status == "running"
        lease_expires_at = result.get("lease_expires_at")
        if is_running and lease_expires_at:
            try:
                expires = datetime.fromisoformat(
                    str(lease_expires_at).replace("Z", "+00:00")
                )
                if expires < datetime.now(timezone.utc):
                    result["lease_expired"] = True
                    # 如果租约已过期且没有 RunPod job，前端可以认为它是待重试/挂起的
                    if not result.get("runpod_job_id"):
                        result["message"] = "任务租约已过期，等待自动恢复"
            except (ValueError, TypeError):
                pass

        result["cancelable"] = status == "queued" or (
            status == "running"
            and not result.get("cancel_requested_at")
            and not result.get("runpod_job_id")
            and not result.get("submission_token")
        )
        result["queue_position"] = 0
        result["estimated_wait_seconds"] = 0
        if status == "queued":
            active = self.repository.list_active(result["resource_type"])
            queued = [row for row in active if row["status"] == "queued"]
            queued.sort(key=lambda row: (row.get("created_at") or "", row["id"]))
            position = next(
                (i for i, row in enumerate(queued, 1) if row["id"] == result["id"]), 1
            )
            running = sum(row["status"] == "running" for row in active)
            result["queue_position"] = position
            waves = max(0, running + position - self.limits[result["resource_type"]])
            result["estimated_wait_seconds"] = (
                (waves + self.limits[result["resource_type"]] - 1)
                // self.limits[result["resource_type"]]
                * self.estimated_seconds[result["resource_type"]]
            )
        return result

    def reconcile_once(self) -> list[dict[str, Any]]:
        reconcile = getattr(self.repository, "reconcile_tasks", None)
        snapshot = getattr(self.repository, "observability_snapshot", None)
        tasks = (
            reconcile(
                self.worker_id, self.running_timeout_seconds, self.queue_alert_seconds
            )
            if reconcile
            else []
        )
        if snapshot:
            value = snapshot()
            if isinstance(value, dict):
                self.observability_snapshot = value
        with self._condition:
            if self._stopped:
                return tasks
            for task in tasks:
                if task.get("task_type") in self.monitors:
                    self._launch_locked(task, monitor=True)
            self._condition.notify_all()
        return tasks

    def _maintenance_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
            try:
                self.reconcile_once()
            except (TaskError, AttributeError, ValueError):
                # 数据库/迁移暂不可用时等待下一轮；绝不在客户端猜测任务结果。
                pass
            with self._condition:
                if self._stopped:
                    return
                self._condition.wait(timeout=self.reconcile_interval_seconds)

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
            launched = False
            try:
                # 数据库与运行时配置请求不得持有进程内条件锁。Supabase 网络抖动时，
                # Web 请求仍需能够提交/查询/取消任务，而不是阻塞到 Gunicorn 超时。
                self.refresh_runtime_limits()
                for resource in ("cpu", "gpu"):
                    while True:
                        claimed = self.repository.claim_next(
                            resource,
                            self.worker_id,
                            cpu_limit=self.limits["cpu"],
                            gpu_limit=self.limits["gpu"],
                        )
                        if not claimed:
                            break
                        should_stop = False
                        with self._condition:
                            if self._stopped:
                                should_stop = True
                            else:
                                self._launch_locked(claimed, monitor=False)
                                launched = True
                        if should_stop:
                            # claim 已成功但执行器尚未启动；用本次租约 fencing 安全释放。
                            if isinstance(
                                self.repository, TaskRepository
                            ) and claimed.get("lease_token"):
                                self.repository.update(
                                    str(claimed["id"]),
                                    {
                                        "status": "queued",
                                        "stage": "queued",
                                        "worker_id": None,
                                        "lease_token": None,
                                        "lease_expires_at": None,
                                        "heartbeat_at": None,
                                        "started_at": None,
                                        "message": "服务停止，任务已恢复排队",
                                    },
                                    expected_status="running",
                                    require_no_runpod=True,
                                    expected_lease_token=str(claimed["lease_token"]),
                                )
                            else:
                                self.repository.requeue(str(claimed["id"]))
                            return
            except (TaskError, AttributeError):
                # RPC 缺失或数据库不可用时 fail closed：不做非原子 PATCH claim。
                with self._condition:
                    if self._stopped:
                        return
                    self._condition.wait(timeout=1.0)
                continue
            if not launched:
                with self._condition:
                    if self._stopped:
                        return
                    self._condition.wait(timeout=0.25)

    def _launch_locked(self, task: Mapping[str, Any], monitor: bool) -> None:
        task_id = str(task["id"])
        if monitor and task_id in self._monitoring_tasks:
            return
        if monitor:
            self._monitoring_tasks.add(task_id)
        event = self._cancel_events.setdefault(task_id, threading.Event())
        if task_id in self._cancelled_before_launch:
            event.set()
            self._cancelled_before_launch.discard(task_id)
        self._executor.submit(self._execute, dict(task), event, monitor)

    def _execute(
        self, task: dict[str, Any], cancel_event: threading.Event, monitor: bool
    ) -> None:
        task_id = task["id"]
        context = TaskContext(self, task_id, cancel_event, task.get("lease_token"))
        token = _current_context.set(context)

        def update_running(values: Mapping[str, Any]):
            if isinstance(self.repository, TaskRepository) and context.lease_token:
                return self.repository.update(
                    task_id,
                    values,
                    expected_status="running",
                    expected_lease_token=context.lease_token,
                )
            return self.repository.update(task_id, values, expected_status="running")

        try:
            if isinstance(self.repository, TaskRepository) and context.lease_token:
                if not self.repository.admit_execution(
                    task_id, self.worker_id, context.lease_token
                ):
                    try:
                        current = self.repository.get(task_id)
                    except Exception:
                        current = None
                    if current and (
                        current.get("status") == "cancelled"
                        or current.get("cancel_requested_at")
                    ):
                        raise LegacyTaskCancelled("任务已取消")
                    raise TaskNotCancelable("任务执行租约已失效")
            context.raise_if_cancelled()
            handler = (
                self.monitors.get(task["task_type"])
                if monitor
                else self._runtime_handlers.get(task_id)
                or self.handlers.get(task["task_type"])
            )
            if handler is None:
                raise TaskError("没有可用于恢复此任务的执行器")
            execution_payload = dict(task.get("payload") or {})
            billing_exempt = (
                execution_payload.pop(_BILLING_EXEMPT_KEY, False) is True
            )
            result = _json_safe_result(handler(context, execution_payload))
            current = self.repository.get(task_id)
            if current and current["status"] == "running":
                message = result.get("message", "处理完成")
                lease_token = context.lease_token
                if billing_exempt:
                    result["billing"] = {
                        "billable": False,
                        "charged_minor": 0,
                        "used_free_quota": False,
                        "admin_exempt": True,
                    }
                if isinstance(self.repository, TaskRepository) and lease_token:
                    if (
                        current.get("resource_type") == "gpu"
                        and self._billing_enforced
                        and not billing_exempt
                    ):
                        if self.billing_service is None:
                            raise BillingUnavailable("billing unavailable")
                        unit_price = self.billing_service.price_for(
                            current["task_type"], current["resource_type"]
                        )
                        billing = self.repository.complete_gpu(
                            task_id,
                            self.worker_id,
                            lease_token,
                            unit_price,
                            current["task_type"],
                            result,
                            message,
                        )
                        result["billing"] = billing
                    else:
                        if not self.repository.finalize(
                            task_id, self.worker_id, lease_token, result, message
                        ):
                            raise TaskNotCancelable("任务租约已失效")
                else:
                    if (
                        current.get("resource_type") == "gpu"
                        and self._billing_enforced
                        and not billing_exempt
                    ):
                        if self.billing_service is None:
                            raise BillingUnavailable("billing unavailable")
                        billing = self.billing_service.charge_completed_task(
                            task_id,
                            current["task_type"],
                            current["resource_type"],
                            module_name=current["task_type"],
                        )
                        result["billing"] = billing
                    self.repository.update(
                        task_id,
                        {
                            "status": "completed",
                            "stage": "completed",
                            "progress": 100,
                            "message": message,
                            "result": result,
                            "completed_at": utcnow(),
                        },
                        expected_status="running",
                    )
        except TaskNotCancelable:
            pass
        except LegacyTaskCancelled:
            current = self.repository.get(task_id)
            if current and current["status"] == "running":
                update_running(
                    {
                        "status": "cancelled",
                        "stage": "cancelled",
                        "progress": 0,
                        "message": "任务已取消",
                        "cancel_requested_at": utcnow(),
                        "completed_at": utcnow(),
                    }
                )
        except ReconciliationPending:
            current = self.repository.get(task_id)
            if current and current["status"] == "running":
                update_running(
                    {
                        "stage": "pending_reconciliation",
                        "message": "RunPod 已结束，本地产物待安全恢复",
                    }
                )
        except RemoteTaskFailed:
            current = self.repository.get(task_id)
            if current and current["status"] == "running":
                update_running(
                    {
                        "status": "failed",
                        "stage": "failed",
                        "message": "RunPod 任务失败",
                        "completed_at": utcnow(),
                    }
                )
        except BillingError as exc:
            current = self.repository.get(task_id)
            if current and current["status"] == "running":
                update_running(
                    {
                        "status": "failed",
                        "stage": "billing_failed",
                        "message": "计费失败",
                        "completed_at": utcnow(),
                    }
                )
        except Exception as exc:
            LOGGER.exception(
                "Task execution failed task_id=%s task_type=%s",
                task_id,
                task.get("task_type"),
            )
            current = self.repository.get(task_id)
            if current and current["status"] == "running":
                if monitor:
                    # 轮询失败不等于远端失败；保留 running 供后续维护轮次再次对账。
                    update_running(
                        {
                            "stage": "pending_reconciliation",
                            "message": "RunPod 状态暂不可确认，等待下次对账",
                        }
                    )
                else:
                    message = _public_task_error(exc)
                    update_running(
                        {
                            "status": "failed",
                            "stage": "failed",
                            "message": message,
                            "completed_at": utcnow(),
                        }
                    )
        finally:
            _current_context.reset(token)
            with self._condition:
                self._cancel_events.pop(task_id, None)
                self._runtime_handlers.pop(task_id, None)
                if monitor:
                    self._monitoring_tasks.discard(task_id)
                self._condition.notify_all()

    def stop(self, wait: bool = True) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if wait:
            current = threading.current_thread()
            for thread in self._threads:
                if thread is not current:
                    thread.join(
                        timeout=max(1.0, self.repository.timeout + 1.0)
                        if isinstance(self.repository, TaskRepository)
                        else 2.0
                    )
        self._executor.shutdown(wait=wait, cancel_futures=False)


class InMemoryTaskRepository:
    """与 TaskRepository 同契约的线程安全实现，仅用于测试或显式本地开发。"""

    def __init__(self):
        self.rows: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def create(self, user_id, task_type, resource_type, payload, task_id=None):
        if resource_type not in RESOURCE_TYPES:
            raise ValueError("resource_type 仅支持 cpu 或 gpu")
        with self.lock:
            if any(
                row["user_id"] == str(user_id) and row["status"] in ACTIVE_STATUSES
                for row in self.rows.values()
            ):
                raise ActiveTaskExists("每个用户同时只能有一个排队中或运行中的任务")
            now = utcnow()
            row = {
                "id": task_id or str(uuid.uuid4()),
                "user_id": str(user_id),
                "task_type": task_type,
                "resource_type": resource_type,
                "status": "queued",
                "payload": dict(payload),
                "result": None,
                "progress": 0,
                "stage": "queued",
                "message": "任务已提交",
                "runpod_job_id": None,
                "runpod_started_at": None,
                "cancel_requested_at": None,
                "worker_id": None,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "updated_at": now,
            }
            self.rows[row["id"]] = row
            return dict(row)

    def get(self, task_id, user_id=None):
        with self.lock:
            row = self.rows.get(task_id)
            if not row or (user_id is not None and row["user_id"] != str(user_id)):
                return None
            return dict(row)

    def _list(self, statuses):
        with self.lock:
            rows = [
                dict(row) for row in self.rows.values() if row["status"] in statuses
            ]
        return sorted(rows, key=lambda row: (row["created_at"], row["id"]))

    def list_queued(self):
        return self._list({"queued"})

    def list_recoverable(self):
        return self._list(ACTIVE_STATUSES)

    def list_active(self, resource_type=None):
        rows = self._list(ACTIVE_STATUSES)
        return [
            row
            for row in rows
            if resource_type is None or row["resource_type"] == resource_type
        ]

    def find_active(self, user_id, task_type=None):
        with self.lock:
            rows = [
                dict(row)
                for row in self.rows.values()
                if row["user_id"] == str(user_id)
                and row["status"] in ACTIVE_STATUSES
                and (task_type is None or row["task_type"] == task_type)
            ]
        rows.sort(
            key=lambda row: (row.get("created_at") or "", row["id"]),
            reverse=True,
        )
        return rows[0] if rows else None

    def update(self, task_id, values, *, expected_status=None, require_no_runpod=False):
        with self.lock:
            row = self.rows.get(task_id)
            if not row or (expected_status and row["status"] != expected_status):
                return None
            if require_no_runpod and row.get("runpod_job_id"):
                return None
            row.update(dict(values))
            if "updated_at" not in values:
                row["updated_at"] = utcnow()
            return dict(row)

    def claim_next(self, resource_type, worker_id, cpu_limit=3, gpu_limit=5):
        """在同一锁内完成全局计数、FIFO 选择和状态更新，模拟数据库 RPC。"""
        if resource_type not in RESOURCE_TYPES:
            raise ValueError("resource_type 仅支持 cpu 或 gpu")
        limit = cpu_limit if resource_type == "cpu" else gpu_limit
        with self.lock:
            running = sum(
                row["status"] == "running" and row["resource_type"] == resource_type
                for row in self.rows.values()
            )
            if running >= limit:
                return None
            queued = sorted(
                (
                    row
                    for row in self.rows.values()
                    if row["status"] == "queued"
                    and row["resource_type"] == resource_type
                ),
                key=lambda row: (row["created_at"], row["id"]),
            )
            if not queued:
                return None
            row = queued[0]
            row.update(
                status="running",
                stage="running",
                message="任务执行中",
                worker_id=worker_id,
                started_at=row.get("started_at") or utcnow(),
                updated_at=utcnow(),
            )
            return dict(row)

    def recover_worker(self, worker_id):
        with self.lock:
            monitors = []
            for row in self.rows.values():
                if row["status"] != "running" or row.get("worker_id") != worker_id:
                    continue
                if row.get("runpod_job_id"):
                    monitors.append(dict(row))
                else:
                    row.update(
                        status="queued",
                        stage="queued",
                        worker_id=None,
                        started_at=None,
                        message="原 worker 重启，任务已恢复排队",
                        updated_at=utcnow(),
                    )
            return sorted(monitors, key=lambda row: (row["created_at"], row["id"]))

    def reconcile_tasks(self, worker_id, timeout_seconds, queue_alert_seconds):
        cutoff = time.time() - int(timeout_seconds)
        monitors = []
        with self.lock:
            for row in self.rows.values():
                if row["status"] != "running":
                    continue
                stamp = (
                    row.get("updated_at")
                    or row.get("started_at")
                    or row.get("created_at")
                )
                try:
                    stale = (
                        datetime.fromisoformat(
                            str(stamp).replace("Z", "+00:00")
                        ).timestamp()
                        <= cutoff
                    )
                except (TypeError, ValueError):
                    stale = False
                if not stale:
                    continue
                if row.get("runpod_job_id"):
                    row.update(
                        stage="pending_reconciliation",
                        worker_id=worker_id,
                        message="RunPod 状态待对账",
                        updated_at=utcnow(),
                    )
                    monitors.append(dict(row))
                else:
                    row.update(
                        status="queued",
                        stage="queued",
                        worker_id=None,
                        started_at=None,
                        message="运行令牌超时，任务已恢复排队",
                        updated_at=utcnow(),
                    )
        return monitors

    def observability_snapshot(self):
        now = time.time()
        with self.lock:
            rows = [dict(row) for row in self.rows.values()]
        queued = [row for row in rows if row["status"] == "queued"]
        waits = []
        for row in queued:
            try:
                waits.append(
                    max(
                        0.0,
                        now
                        - datetime.fromisoformat(
                            str(row["created_at"]).replace("Z", "+00:00")
                        ).timestamp(),
                    )
                )
            except (TypeError, ValueError):
                pass
        completed = sum(row["status"] == "completed" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        cancelled = sum(row["status"] == "cancelled" for row in rows)
        final = completed + failed + cancelled
        return {
            "queue_length_cpu": sum(
                row["status"] == "queued" and row["resource_type"] == "cpu"
                for row in rows
            ),
            "queue_length_gpu": sum(
                row["status"] == "queued" and row["resource_type"] == "gpu"
                for row in rows
            ),
            "running_cpu": sum(
                row["status"] == "running" and row["resource_type"] == "cpu"
                for row in rows
            ),
            "running_gpu": sum(
                row["status"] == "running" and row["resource_type"] == "gpu"
                for row in rows
            ),
            "oldest_queue_wait_seconds": max(waits, default=0),
            "success_count": completed,
            "failure_count": failed,
            "cancel_count": cancelled,
            "success_rate": completed / final if final else 0,
            "failure_rate": failed / final if final else 0,
            "cancel_rate": cancelled / final if final else 0,
            "billing_failure_count": sum(
                row.get("stage") == "billing_failed" for row in rows
            ),
            "open_alert_count": 0,
        }

    def claim(self, task_id, worker_id):
        return self.update(
            task_id,
            {
                "status": "running",
                "stage": "running",
                "message": "任务执行中",
                "worker_id": worker_id,
                "started_at": utcnow(),
            },
            expected_status="queued",
        )

    def requeue(self, task_id):
        return self.update(
            task_id,
            {
                "status": "queued",
                "stage": "queued",
                "worker_id": None,
                "message": "服务重启，任务已恢复排队",
            },
            expected_status="running",
            require_no_runpod=True,
        )

    def mark_runpod_started(self, task_id, job_id):
        return self.update(
            task_id,
            {
                "runpod_job_id": str(job_id),
                "runpod_started_at": utcnow(),
                "message": "RunPod 任务已提交",
            },
            expected_status="running",
            require_no_runpod=True,
        )

    def request_cancel(self, task_id, user_id):
        with self.lock:
            row = self.rows.get(task_id)
            if not row or row["user_id"] != str(user_id):
                raise TaskNotFound("任务不存在")
            if row["status"] not in ACTIVE_STATUSES or (
                row["status"] == "running" and row.get("runpod_job_id")
            ):
                raise TaskNotCancelable("任务已结束或已进入 RunPod，不能取消")
            now = utcnow()
            row.update(
                status="cancelled",
                stage="cancelled",
                message="任务已取消",
                cancel_requested_at=now,
                completed_at=now,
                updated_at=now,
            )
            return dict(row)
