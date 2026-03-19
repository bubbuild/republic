"""Tool execution helpers for Republic."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, NoReturn

from pydantic import ValidationError

from republic.core.errors import ErrorKind
from republic.core.results import RepublicError, ToolExecution
from republic.tools.context import ToolContext
from republic.tools.schema import Tool, ToolInput, normalize_tools


class ToolExecutor:
    """Execute tool calls with predictable validation and serialization."""

    def execute(
        self,
        response: list[dict[str, Any]] | dict[str, Any] | str,
        tools: ToolInput = None,
        *,
        context: ToolContext | None = None,
    ) -> ToolExecution:
        tool_calls, tool_map = self._prepare_execution(response, tools)
        if not tool_map:
            if tool_calls:
                raise RepublicError(ErrorKind.TOOL, "No runnable tools are available.")
            return ToolExecution(tool_calls=[], tool_results=[])

        results: list[Any] = []
        error: RepublicError | None = None
        for tool_response in tool_calls:
            try:
                result = self._handle_tool_response(tool_response, tool_map, context)
            except RepublicError as exc:
                error = exc
                result = exc.as_dict()
            results.append(result)

        return ToolExecution(tool_calls=tool_calls, tool_results=results, error=error)

    async def execute_async(
        self,
        response: list[dict[str, Any]] | dict[str, Any] | str,
        tools: ToolInput = None,
        *,
        context: ToolContext | None = None,
    ) -> ToolExecution:
        tool_calls, tool_map = self._prepare_execution(response, tools)
        if not tool_map:
            if tool_calls:
                raise RepublicError(ErrorKind.TOOL, "No runnable tools are available.")
            return ToolExecution(tool_calls=[], tool_results=[])

        results: list[Any] = []
        error: RepublicError | None = None
        gathered = await asyncio.gather(
            *(self._handle_tool_response_async(tool_response, tool_map, context) for tool_response in tool_calls),
            return_exceptions=True,
        )
        for resp in gathered:
            if isinstance(resp, RepublicError):
                error = resp
                results.append(resp.as_dict())
            elif isinstance(resp, BaseException):  # This should not happen, but we catch it just in case.
                raise resp
            else:
                results.append(resp)

        return ToolExecution(tool_calls=tool_calls, tool_results=results, error=error)

    def _prepare_execution(
        self,
        response: list[dict[str, Any]] | dict[str, Any] | str,
        tools: ToolInput,
    ) -> tuple[list[dict[str, Any]], dict[str, Tool]]:
        tool_calls = self._normalize_response(response)
        tool_map = self._build_tool_map(tools)
        return tool_calls, tool_map

    def _resolve_tool_call(
        self,
        tool_response: Any,
        tool_map: dict[str, Tool],
    ) -> tuple[str, Tool, dict[str, Any]]:
        if not isinstance(tool_response, dict):
            raise RepublicError(ErrorKind.INVALID_INPUT, "Each tool call must be an object.")
        tool_name = tool_response.get("function", {}).get("name")
        if not tool_name:
            raise RepublicError(ErrorKind.INVALID_INPUT, "Tool call is missing name.")
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            raise RepublicError(ErrorKind.TOOL, f"Unknown tool name: {tool_name}.")
        tool_args = tool_response.get("function", {}).get("arguments", {})
        tool_args = self._normalize_tool_args(tool_name, tool_args)
        return tool_name, tool_obj, tool_args

    def _invoke_tool(
        self,
        *,
        tool_name: str,
        tool_obj: Tool,
        tool_args: dict[str, Any],
        context: ToolContext | None,
    ) -> Any:
        if tool_obj.context:
            if context is None:
                raise RepublicError(
                    ErrorKind.INVALID_INPUT, f"Tool '{tool_name}' requires context but none was provided."
                )
            return tool_obj.run(context=context, **tool_args)
        return tool_obj.run(**tool_args)

    def _handle_tool_response(
        self,
        tool_response: Any,
        tool_map: dict[str, Tool],
        context: ToolContext | None,
    ) -> Any:
        tool_name, tool_obj, tool_args = self._resolve_tool_call(tool_response, tool_map)
        try:
            result = self._invoke_tool(
                tool_name=tool_name,
                tool_obj=tool_obj,
                tool_args=tool_args,
                context=context,
            )
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                self._raise_async_execute_error(tool_name)
        except RepublicError:
            raise
        except ValidationError as exc:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                f"Tool '{tool_name}' argument validation failed.",
                details={"errors": exc.errors()},
            ) from exc
        except Exception as exc:
            raise RepublicError(
                ErrorKind.TOOL,
                f"Tool '{tool_name}' execution failed.",
                details={"error": repr(exc)},
            ) from exc
        else:
            return result

    async def _handle_tool_response_async(
        self,
        tool_response: Any,
        tool_map: dict[str, Tool],
        context: ToolContext | None,
    ) -> Any:
        tool_name, tool_obj, tool_args = self._resolve_tool_call(tool_response, tool_map)
        try:
            result = self._invoke_tool(
                tool_name=tool_name,
                tool_obj=tool_obj,
                tool_args=tool_args,
                context=context,
            )
            if inspect.isawaitable(result):
                return await result
        except RepublicError:
            raise
        except ValidationError as exc:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                f"Tool '{tool_name}' argument validation failed.",
                details={"errors": json.loads(exc.json())},
            ) from exc
        except Exception as exc:
            raise RepublicError(
                ErrorKind.TOOL,
                f"Tool '{tool_name}' execution failed.",
                details={"error": repr(exc)},
            ) from exc
        else:
            return result

    def _raise_async_execute_error(self, tool_name: str) -> NoReturn:
        raise RepublicError(
            ErrorKind.INVALID_INPUT,
            f"Tool '{tool_name}' is async; use execute_async() instead of execute().",
        )

    def _normalize_response(
        self,
        response: list[dict[str, Any]] | dict[str, Any] | str,
    ) -> list[dict[str, Any]]:
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise RepublicError(
                    ErrorKind.INVALID_INPUT,
                    "Tool response is not a valid JSON string.",
                    details={"error": str(exc)},
                ) from exc
        if isinstance(response, dict):
            response = [response]
        if not isinstance(response, list):
            raise RepublicError(ErrorKind.INVALID_INPUT, "Tool response must be a list of objects.")
        return response

    def _build_tool_map(self, tools: ToolInput) -> dict[str, Tool]:
        if tools is None:
            raise RepublicError(ErrorKind.INVALID_INPUT, "No tools provided.")
        try:
            toolset = normalize_tools(tools)
        except (ValueError, TypeError) as exc:
            raise RepublicError(ErrorKind.INVALID_INPUT, str(exc)) from exc

        return {tool_obj.name: tool_obj for tool_obj in toolset.runnable if tool_obj.name}

    def _normalize_tool_args(self, tool_name: str, tool_args: Any) -> dict[str, Any]:
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError as exc:
                raise RepublicError(
                    ErrorKind.INVALID_INPUT,
                    f"Tool '{tool_name}' arguments are not valid JSON.",
                ) from exc
        if isinstance(tool_args, dict):
            return dict(tool_args)
        raise RepublicError(
            ErrorKind.INVALID_INPUT,
            f"Tool '{tool_name}' arguments must be an object.",
        )
