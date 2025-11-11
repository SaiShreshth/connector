from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
# from loguru import 

import logging
logger = logging.getLogger(__name__)

from app.utils.redaction import redact_headers, truncate_text
from app.db.postgres import insert_shopify_call


class ShopifyClient:
    def __init__(
        self,
        shop_domain: str,
        access_token: str,
        storefront_access_token: Optional[str] = None,
        api_version: str = "2024-10",
        timeout_seconds: float = 15.0,
        log_response_bodies: bool = True,
    ) -> None:
        self.base_url = f"https://{shop_domain}"
        self.api_version = api_version
        self.access_token = access_token
        self.storefront_access_token = storefront_access_token
        self.timeout_seconds = timeout_seconds
        self.log_response_bodies = log_response_bodies
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "X-Shopify-Access-Token": self.access_token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            # Follow redirects automatically (max 5 redirects)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout_seconds,
                follow_redirects=True,
                max_redirects=5
            )
        return self._client

    def _get_headers_for_path(self, path: str) -> Dict[str, str]:
        """Return appropriate headers based on the API path."""
        if path.startswith("/api/"):
            # Storefront API (GraphQL)
            if self.storefront_access_token:
                return {
                    # "X-Shopify-Storefront-Access-Token": self.storefront_access_token,
                    "X-Shopify-Storefront-Access-Token": self.storefront_access_token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            else:
                logger.warning("Storefront access token not configured, falling back to admin token for /api/ path")
                return {
                    "X-Shopify-Access-Token": self.access_token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
        else:
            # Admin API
            return {
                "X-Shopify-Access-Token": self.access_token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        client = await self._ensure_client()
        corr_id = str(uuid.uuid4())
        url_path = path
        url_full = f"{self.base_url}{url_path}"
        logger.info("Shopify request start | corr_id=%s method=%s path=%s", corr_id, method.upper(), url_path)

        # Prepare headers (merged) - use path-specific headers
        path_headers = self._get_headers_for_path(path)
        req_headers = dict(path_headers)
        if headers:
            req_headers.update(headers)

        # For storefront API, ensure we don't send admin token
        if path.startswith("/api/") and self.storefront_access_token:
            # Remove admin token if present
            req_headers.pop("X-Shopify-Access-Token", None)

        request_ts = datetime.now(timezone.utc)
        error_text = None
        status_code = None
        response_headers: Dict[str, Any] = {}
        response_body_text: Optional[str] = None

        response = None
        try:
            response = await client.request(method=method, url=url_path, params=params, json=json_body)
            status_code = response.status_code
            logger.debug(f"Received response | corr_id={corr_id} status={status_code} url={response.url}")
            # Convert httpx headers to regular dict (handle case-insensitive headers)
            response_headers = {str(k): str(v) for k, v in response.headers.items()}
            if self.log_response_bodies:
                try:
                    # Try to get text, fallback to bytes if needed
                    if hasattr(response, 'text') and response.text:
                        response_body_text = truncate_text(response.text)
                    elif hasattr(response, 'content') and response.content:
                        # Decode bytes to string
                        try:
                            body_str = response.content.decode('utf-8')
                            response_body_text = truncate_text(body_str)
                        except UnicodeDecodeError:
                            response_body_text = truncate_text(f"[Binary content, {len(response.content)} bytes]")
                except Exception as e:
                    logger.warning(f"Failed to extract response body ({corr_id}): {e}")
                    response_body_text = None
            # Don't raise on status - we want to log all responses
            # Only raise if it's a critical error (not HTTP status errors)
            return response
        except httpx.HTTPStatusError as exc:
            # HTTP error (4xx, 5xx) - we still have a response
            response = exc.response
            status_code = response.status_code if response else None
            if response:
                response_headers = {str(k): str(v) for k, v in response.headers.items()}
                if self.log_response_bodies:
                    try:
                        if hasattr(response, 'text') and response.text:
                            response_body_text = truncate_text(response.text)
                        elif hasattr(response, 'content') and response.content:
                            try:
                                body_str = response.content.decode('utf-8')
                                response_body_text = truncate_text(body_str)
                            except UnicodeDecodeError:
                                response_body_text = truncate_text(f"[Binary content, {len(response.content)} bytes]")
                    except Exception as e:
                        logger.warning(f"Failed to extract response body from error ({corr_id}): {e}")
                        response_body_text = None
            error_text = str(exc)
            logger.warning(f"Shopify request failed ({corr_id}): {error_text}")
            raise
        except Exception as exc:
            # Other errors (network, timeout, etc.) - no response
            error_text = str(exc)
            logger.warning(f"Shopify request failed ({corr_id}): {error_text}")
            raise
        finally:
            response_ts = datetime.now(timezone.utc)
            latency_ms = int((response_ts - request_ts).total_seconds() * 1000)
            masked_req_headers = redact_headers(req_headers)
            masked_resp_headers = redact_headers(response_headers)

            # Safe serialization of request body
            req_body_text = None
            if json_body is not None:
                try:
                    req_body_text = truncate_text(json.dumps(json_body, ensure_ascii=False, default=str))
                except Exception as e:
                    logger.warning(f"Failed to serialize request body ({corr_id}): {e}")
                    req_body_text = truncate_text(str(json_body))

            # Log ALL responses to database (2xx, 3xx, 4xx, 5xx)
            # Only skip if we have no status code at all (network error before response)
            logger.info(
                "Preparing to log | corr_id=%s status_code=%s response_headers_count=%s has_response_body=%s",
                corr_id,
                status_code,
                len(response_headers) if response_headers else 0,
                response_body_text is not None,
            )
            if status_code is not None:
                try:
                    await insert_shopify_call(
                        {
                            "request_ts": request_ts,
                            "response_ts": response_ts,
                            "latency_ms": latency_ms,
                            "method": method.upper(),
                            "url": url_full,
                            "path": url_path,
                            "status_code": status_code,
                            "request_headers": masked_req_headers,
                            "request_body": req_body_text,
                            "response_headers": masked_resp_headers,
                            "response_body": response_body_text,
                            "error": error_text,
                            "correlation_id": corr_id,
                        }
                    )
                    logger.info(
                        "Shopify request end   | corr_id=%s method=%s path=%s status=%s latency_ms=%s [LOGGED TO DB]",
                        corr_id,
                        method.upper(),
                        url_path,
                        status_code,
                        latency_ms,
                    )
                except Exception as insert_exc:
                    logger.error(
                        "CRITICAL: Failed to insert Shopify call to DB | corr_id=%s error=%s",
                        corr_id,
                        str(insert_exc),
                        exc_info=True,
                    )
            else:
                logger.info(
                    "Shopify request end   | corr_id=%s method=%s path=%s status=None latency_ms=%s [NOT LOGGED - no response received]",
                    corr_id,
                    method.upper(),
                    url_path,
                    latency_ms,
                )

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None):
        return await self.request("GET", path, params=params, headers=headers)

    async def post(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ):
        return await self.request("POST", path, json_body=json_body, headers=headers)

    async def put(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ):
        return await self.request("PUT", path, json_body=json_body, headers=headers)

    async def delete(self, path: str, headers: Optional[Dict[str, Any]] = None):
        return await self.request("DELETE", path, headers=headers)


