"""服务端钱包计费封装；价格、身份与 Supabase 凭据均不接受客户端输入。"""
from __future__ import annotations

import os
import time
from typing import Any, Mapping

import requests


class BillingError(RuntimeError):
    """可安全向调用方展示的计费错误。"""


class BillingUnavailable(BillingError):
    pass


class InsufficientBalance(BillingError):
    pass


class ActiveTasksExist(BillingError):
    """账户存在排队中或运行中的任务，暂不可注销。"""


DEFAULT_GPU_PRICES = {
    "stems": 100,
    "vocal-transform": 150,
    "song-info": 50,
    "chord-analysis": 100,
    "audio-to-midi": 100,
}
PRICE_ENV_VARS = {
    "stems": "GPU_PRICE_STEMS_MINOR",
    "vocal-transform": "GPU_PRICE_VOCAL_TRANSFORM_MINOR",
    "song-info": "GPU_PRICE_SONG_INFO_MINOR",
    "chord-analysis": "GPU_PRICE_CHORD_ANALYSIS_MINOR",
    "audio-to-midi": "GPU_PRICE_AUDIO_TO_MIDI_MINOR",
}


def _positive_price(value: Any, default: int) -> int:
    try:
        price = int(value)
    except (TypeError, ValueError):
        price = default
    return price if price > 0 else default


class BillingService:
    """调用 SECURITY DEFINER RPC；不提供任意价格透传接口。"""

    def __init__(self, base_url: str, service_key: str, *, timeout: float = 8.0,
                 prices: Mapping[str, int] | None = None,
                 session: requests.Session | None = None):
        self.configured = bool(base_url and service_key)
        self.rpc_base = base_url.rstrip("/") + "/rest/v1/rpc" if base_url else ""
        self.timeout = timeout
        self.http = session or requests.Session()
        if service_key:
            headers = {"apikey": service_key, "Accept": "application/json",
                       "Content-Type": "application/json"}
            # legacy service_role JWT 可作 Bearer；sb_secret_* 只能放 apikey。
            if service_key.count(".") == 2:
                headers["Authorization"] = f"Bearer {service_key}"
            self.http.headers.update(headers)
        configured_prices = dict(prices or {})
        self.prices = {
            task_type: _positive_price(
                configured_prices.get(task_type, os.environ.get(PRICE_ENV_VARS[task_type])),
                default,
            )
            for task_type, default in DEFAULT_GPU_PRICES.items()
        }

    @classmethod
    def from_app(cls, app) -> "BillingService":
        prices = {
            task_type: app.config.get(env_name, os.environ.get(env_name))
            for task_type, env_name in PRICE_ENV_VARS.items()
        }
        return cls(app.config.get("SUPABASE_URL", ""),
                   app.config.get("SUPABASE_SERVICE_KEY", ""), prices=prices)

    def price_for(self, task_type: str, resource_type: str) -> int:
        if resource_type != "gpu":
            return 0
        try:
            return self.prices[task_type]
        except KeyError as exc:
            raise BillingError(f"GPU 任务类型未配置服务端价格: {task_type}") from exc

    def _rpc(self, name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.configured:
            raise BillingUnavailable("billing unavailable")
        response = None
        for attempt in range(3):
            try:
                response = self.http.request(
                    "POST", f"{self.rpc_base}/{name}", json=dict(payload), timeout=self.timeout)
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == 2:
                    raise BillingUnavailable("billing unavailable") from exc
                time.sleep(0.1 * (2 ** attempt))
            except requests.RequestException as exc:
                raise BillingUnavailable("billing unavailable") from exc
        if response is None:
            raise BillingUnavailable("billing unavailable")
        if response.status_code >= 400:
            body = response.text[:500].lower()
            if "insufficient_balance" in body or "余额不足" in body:
                raise InsufficientBalance("余额不足，无法执行 GPU 任务")
            raise BillingUnavailable("billing unavailable")
        try:
            data = response.json()
        except ValueError as exc:
            raise BillingUnavailable("billing unavailable") from exc
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            raise BillingUnavailable("billing unavailable")
        if data.get("eligible") is False:
            raise InsufficientBalance(data.get("message") or "余额不足，无法执行 GPU 任务")
        return data

    def get_gpu_usage_history(self, user_id: str, *, limit: int = 20,
                              offset: int = 0) -> dict[str, Any]:
        """按时间倒序读取当前用户的 GPU 使用记录。"""
        return self._rpc("get_gpu_usage_history", {
            "user_id": str(user_id), "limit": int(limit), "offset": int(offset),
        })

    def cancel_user_account(self, user_id: str) -> dict[str, Any]:
        result = self._rpc("cancel_user_account", {"user_id": str(user_id)})
        if result.get("error") == "active_tasks_exist":
            raise ActiveTasksExist("存在进行中任务，请等待任务结束后再注销")
        if result.get("success") is not True:
            raise BillingUnavailable("account service unavailable")
        return result

    def check_gpu_eligibility(self, user_id: str, task_type: str,
                              resource_type: str = "gpu") -> dict[str, Any]:
        unit_price = self.price_for(task_type, resource_type)
        if resource_type != "gpu":
            return {"eligible": True, "unit_price_minor": 0, "billable": False}
        return self._rpc("check_gpu_billing_eligibility", {
            "p_user_id": str(user_id), "p_task_type": task_type,
            "p_unit_price": unit_price,
        })

    def charge_completed_task(self, task_id: str, task_type: str,
                              resource_type: str, module_name: str | None = None) -> dict[str, Any]:
        unit_price = self.price_for(task_type, resource_type)
        if resource_type != "gpu":
            return {"billable": False, "charged_minor": 0, "used_free_quota": False}
        result = self._rpc("charge_completed_gpu_task", {
            "p_task_id": str(task_id), "p_unit_price": unit_price,
            "p_module_name": module_name or task_type,
        })
        result.setdefault("billable", True)
        result.setdefault("unit_price_minor", unit_price)
        return result

    def get_wallet_summary(self, user_id: str) -> dict[str, Any]:
        return self._rpc("get_wallet_summary", {"p_user_id": str(user_id)})


def init_billing_service(app) -> BillingService:
    service = app.extensions.get("billing_service")
    if service is None:
        service = BillingService.from_app(app)
        app.extensions["billing_service"] = service
    return service
