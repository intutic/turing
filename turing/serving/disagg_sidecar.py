"""
P/D Disaggregation Sidecar for llm-d Router Integration.
Handles request header inspection and cross-pod prefill/decode delegation when
participating in llm-d disaggregated inference topologies.
"""

import json
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import urllib.request
import urllib.error


class DisaggRoutingCoordinator:
    """
    Coordinates prefill-decode disaggregation for Turing Engine pods.
    Extracts x-prefiller-host-port from llm-d EPP, delegates prompt prefill to the
    designated prefill worker, and drives local decoding on the target decode worker.
    """

    def __init__(self, local_engine_port: int = 8000, timeout_s: float = 30.0):
        self.local_engine_port = local_engine_port
        self.timeout_s = timeout_s

    def should_disaggregate(self, headers: Dict[str, str]) -> bool:
        return "x-prefiller-host-port" in headers or "X-Prefiller-Host-Port" in headers

    def extract_prefiller_endpoint(self, headers: Dict[str, str]) -> Optional[str]:
        return headers.get("x-prefiller-host-port") or headers.get("X-Prefiller-Host-Port")

    def execute_remote_prefill(
        self,
        prefiller_host_port: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Sends truncated prefill request (max_tokens=1) to the designated prefill pod.
        """
        prefill_payload = dict(payload)
        prefill_payload["max_tokens"] = 1
        prefill_payload["stream"] = False

        req_data = json.dumps(prefill_payload).encode("utf-8")
        url = f"http://{prefiller_host_port}/v1/completions"

        req_headers = {"Content-Type": "application/json"}
        if headers:
            for k, v in headers.items():
                if k.lower().startswith("x-turing-"):
                    req_headers[k] = v

        req = urllib.request.Request(url, data=req_data, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                resp_bytes = response.read()
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to delegate prefill to {prefiller_host_port}: {e}")


def create_disagg_proxy_app(coordinator: DisaggRoutingCoordinator) -> FastAPI:
    """Creates FastAPI proxy app for disaggregated sidecar deployment."""
    app = FastAPI(title="Turing Disaggregation Proxy Sidecar", version="0.3.5")








    @app.get("/health")
    async def health():
        return {"status": "healthy", "mode": "disagg_sidecar"}

    @app.post("/v1/completions")
    async def proxy_completions(request: Request):
        headers_dict = dict(request.headers)
        body = await request.json()

        if coordinator.should_disaggregate(headers_dict):
            prefiller = coordinator.extract_prefiller_endpoint(headers_dict)
            if prefiller:
                coordinator.execute_remote_prefill(prefiller, body, headers_dict)

        # Forward to local decode engine
        local_url = f"http://127.0.0.1:{coordinator.local_engine_port}/v1/completions"
        req_data = json.dumps(body).encode("utf-8")
        fwd_headers = {"Content-Type": "application/json"}
        for k, v in headers_dict.items():
            if k.lower().startswith("x-turing-"):
                fwd_headers[k] = v

        req = urllib.request.Request(local_url, data=req_data, headers=fwd_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=coordinator.timeout_s) as response:
                resp_data = response.read()
                return JSONResponse(content=json.loads(resp_data.decode("utf-8")))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Local engine error: {e}")

    return app
