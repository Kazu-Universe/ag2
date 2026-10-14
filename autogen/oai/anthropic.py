# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Create an OpenAI-compatible client for the Anthropic API.

Example usage:
Install the `anthropic` package by running `pip install --upgrade anthropic`.
- https://docs.anthropic.com/en/docs/quickstart-guide

```python
import autogen

config_list = [
    {
        "model": "claude-3-sonnet-20240229",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
```

Example usage for Anthropic Bedrock:

Install the `anthropic` package by running `pip install --upgrade anthropic`.
- https://docs.anthropic.com/en/docs/quickstart-guide

```python
import autogen

config_list = [
    {
        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "aws_access_key":<accessKey>,
        "aws_secret_key":<secretKey>,
        "aws_session_token":<sessionTok>,
        "aws_region":"us-east-1",
        "api_type": "anthropic",
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
```

Example usage for Anthropic VertexAI:

Install the `anthropic` package by running `pip install anthropic[vertex]`.
- https://docs.anthropic.com/en/docs/quickstart-guide

```python

import autogen
config_list = [
    {
        "model": "claude-3-5-sonnet-20240620-v1:0",
        "gcp_project_id": "dummy_project_id",
        "gcp_region": "us-west-2",
        "gcp_auth_token": "dummy_auth_token",
        "api_type": "anthropic",
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
```python
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
import warnings
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..import_utils import optional_import_block, require_optional_import
from ..llm_config import LLMConfigEntry, register_llm_config
from .client_utils import FormatterProtocol, validate_parameter
from .oai_models import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, Choice, CompletionUsage

with optional_import_block():
    from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex
    from anthropic import __version__ as anthropic_version
    from anthropic.types import Message, TextBlock, ToolUseBlock

    TOOL_ENABLED = anthropic_version >= "0.23.1"
    if TOOL_ENABLED:
        pass


ANTHROPIC_PRICING_1k = {
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-3-7-sonnet-20250219": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-5-sonnet-20240620": (0.003, 0.015),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-2.1": (0.008, 0.024),
    "claude-2.0": (0.008, 0.024),
    "claude-instant-1.2": (0.008, 0.024),
}


@register_llm_config
class AnthropicLLMConfigEntry(LLMConfigEntry):
    api_type: Literal["anthropic"] = "anthropic"
    timeout: Optional[int] = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stop_sequences: Optional[list[str]] = None
    stream: bool = False
    max_tokens: int = Field(default=4096, ge=1)
    price: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)
    tool_choice: Optional[dict] = None
    thinking: Optional[dict] = None

    gcp_project_id: Optional[str] = None
    gcp_region: Optional[str] = None
    gcp_auth_token: Optional[str] = None

    def create_client(self):
        raise NotImplementedError("AnthropicLLMConfigEntry.create_client is not implemented.")


@require_optional_import("anthropic", "anthropic")
class AnthropicClient:
    def __init__(self, **kwargs: Any):
        """Initialize the Anthropic API client.

        Args:
            **kwargs: The configuration parameters for the client.
        """
        self._api_key = kwargs.get("api_key")
        self._aws_access_key = kwargs.get("aws_access_key")
        self._aws_secret_key = kwargs.get("aws_secret_key")
        self._aws_session_token = kwargs.get("aws_session_token")
        self._aws_region = kwargs.get("aws_region")
        self._gcp_project_id = kwargs.get("gcp_project_id")
        self._gcp_region = kwargs.get("gcp_region")
        self._gcp_auth_token = kwargs.get("gcp_auth_token")
        self._base_url = kwargs.get("base_url")

        if not self._api_key:
            self._api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self._aws_access_key:
            self._aws_access_key = os.getenv("AWS_ACCESS_KEY")

        if not self._aws_secret_key:
            self._aws_secret_key = os.getenv("AWS_SECRET_KEY")

        if not self._aws_region:
            self._aws_region = os.getenv("AWS_REGION")

        if not self._gcp_region:
            self._gcp_region = os.getenv("GCP_REGION")

        if self._api_key is None:
            if self._aws_region:
                if self._aws_access_key is None or self._aws_secret_key is None:
                    raise ValueError("API key or AWS credentials are required to use the Anthropic API.")
            elif self._gcp_region:
                if self._gcp_project_id is None or self._gcp_region is None:
                    raise ValueError("API key or GCP credentials are required to use the Anthropic API.")
            else:
                raise ValueError("API key or AWS credentials or GCP credentials are required to use the Anthropic API.")

        if self._api_key is not None:
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = Anthropic(**client_kwargs)
        elif self._gcp_region is not None:
            kw = {}
            for i, p in enumerate(inspect.signature(AnthropicVertex).parameters):
                if hasattr(self, f"_gcp_{p}"):
                    kw[p] = getattr(self, f"_gcp_{p}")
            if self._base_url:
                kw["base_url"] = self._base_url
            self._client = AnthropicVertex(**kw)
        else:
            client_kwargs = {
                "aws_access_key": self._aws_access_key,
                "aws_secret_key": self._aws_secret_key,
                "aws_session_token": self._aws_session_token,
                "aws_region": self._aws_region,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = AnthropicBedrock(**client_kwargs)

        self._last_tooluse_status = {}

        # Store the response format, if provided (for structured outputs)
        self._response_format: Optional[type[BaseModel]] = None

    def load_config(self, params: dict[str, Any]):
        """Load the configuration for the Anthropic API client."""
        anthropic_params = {}

        anthropic_params["model"] = params.get("model")
        assert anthropic_params["model"], "Please provide a `model` in the config_list to use the Anthropic API."

        anthropic_params["temperature"] = validate_parameter(
            params, "temperature", (float, int), False, 1.0, (0.0, 1.0), None
        )
        anthropic_params["max_tokens"] = validate_parameter(params, "max_tokens", int, False, 4096, (1, None), None)
        anthropic_params["timeout"] = validate_parameter(params, "timeout", int, True, None, (1, None), None)
        anthropic_params["top_k"] = validate_parameter(params, "top_k", int, True, None, (1, None), None)
        anthropic_params["top_p"] = validate_parameter(params, "top_p", (float, int), True, None, (0.0, 1.0), None)
        anthropic_params["stop_sequences"] = validate_parameter(params, "stop_sequences", list, True, None, None, None)
        anthropic_params["stream"] = validate_parameter(params, "stream", bool, False, False, None, None)
        if "thinking" in params:
            anthropic_params["thinking"] = params["thinking"]

        if anthropic_params["stream"]:
            warnings.warn(
                "Streaming is not currently supported, streaming will be disabled.",
                UserWarning,
            )
            anthropic_params["stream"] = False

        # Note the Anthropic API supports "tool" for tool_choice but you must specify the tool name so we will ignore that here
        # Dictionary, see options here: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#controlling-claudes-output
        # type = auto, any, tool, none | name = the name of the tool if type=tool
        anthropic_params["tool_choice"] = validate_parameter(params, "tool_choice", dict, True, None, None, None)

        return anthropic_params

    def cost(self, response) -> float:
        """Calculate the cost of the completion using the Anthropic pricing."""
        return response.cost

    @property
    def api_key(self):
        return self._api_key

    @property
    def aws_access_key(self):
        return self._aws_access_key

    @property
    def aws_secret_key(self):
        return self._aws_secret_key

    @property
    def aws_session_token(self):
        return self._aws_session_token

    @property
    def aws_region(self):
        return self._aws_region

    @property
    def gcp_project_id(self):
        return self._gcp_project_id

    @property
    def gcp_region(self):
        return self._gcp_region

    @property
    def gcp_auth_token(self):
        return self._gcp_auth_token

    def create(self, params: dict[str, Any]) -> ChatCompletion:
        """Creates a completion using the Anthropic API."""
        if "tools" in params:
            converted_functions = self.convert_tools_to_functions(params["tools"])
            params["functions"] = params.get("functions", []) + converted_functions

        # Convert AG2 messages to Anthropic messages
        anthropic_messages = oai_messages_to_anthropic_messages(params)
        anthropic_params = self.load_config(params)

        # If response_format exists, we want structured outputs
        # Anthropic doesn't support response_format, so using Anthropic's "JSON Mode":
        # https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb
        if params.get("response_format"):
            self._response_format = params["response_format"]
            self._add_response_format_to_system(params)

        # TODO: support stream
        params = params.copy()
        if "functions" in params:
            tools_configs = params.pop("functions")
            tools_configs = [self.openai_func_to_anthropic(tool) for tool in tools_configs]
            params["tools"] = tools_configs

        # Anthropic doesn't accept None values, so we need to use keyword argument unpacking instead of setting parameters.
        # Copy params we need into anthropic_params
        # Remove any that don't have values
        anthropic_params["messages"] = anthropic_messages
        if "system" in params:
            anthropic_params["system"] = params["system"]
        if "tools" in params:
            anthropic_params["tools"] = params["tools"]
        # hep-theory fork fix: params["tool_choice"] was being silently
        # dropped here - never copied into anthropic_params at all, so a
        # forced tool_choice (used by terminator, plan_recorder,
        # review_recorder, etc.) never reached the Anthropic API call.
        # ag2 builds tool_choice in OpenAI's schema unconditionally:
        # {"type": "function", "function": {"name": X}}. Anthropic's
        # forced-single-tool schema ({"type": "tool", "name": X}) was
        # never implemented here. Without this, any agent using a forced
        # tool_choice gets no forcing at all when backed by a
        # Claude/Fable model - not even a 400, just silently ignored.
        if "tool_choice" in params:
            tc = params["tool_choice"]
            if isinstance(tc, dict) and tc.get("type") == "function":
                fn = tc.get("function", {})
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    anthropic_params["tool_choice"] = {"type": "tool", "name": name}
            elif isinstance(tc, dict) and tc.get("type") == "tool":
                # Already in Anthropic's own schema - pass through as-is.
                anthropic_params["tool_choice"] = tc
        if anthropic_params["top_k"] is None:
            del anthropic_params["top_k"]
        if anthropic_params["top_p"] is None:
            del anthropic_params["top_p"]
        if anthropic_params["stop_sequences"] is None:
            del anthropic_params["stop_sequences"]
        if anthropic_params["tool_choice"] is None:
            del anthropic_params["tool_choice"]

        response = self._client.messages.create(**anthropic_params)

        # hep-theory fork: per-call usage log (Phase 0 of docs/prompt_caching_plan.md;
        # inert unless CMBAGENT_CALL_LOG is set). Read-only observer: it only builds
        # a record from the request/response; client.py writes it out with the agent name.
        if os.environ.get("CMBAGENT_CALL_LOG"):
            self._cmbagent_last_call = _cmbagent_call_record(anthropic_params, response)

        tool_calls = []
        message_text = ""

        if self._response_format:
            try:
                parsed_response = self._extract_json_response(response)
                message_text = _format_json_response(parsed_response)
            except ValueError as e:
                message_text = str(e)

            anthropic_finish = "stop"
        else:
            if response is not None:
                # If we have tool use as the response, populate completed tool calls for our return OAI response
                if response.stop_reason == "tool_use":
                    anthropic_finish = "tool_calls"
                    for content in response.content:
                        # hep-theory fork: use the API's own "type" discriminator
                        # field instead of a strict `type(content) == ToolUseBlock`
                        # check. The strict check is fragile across anthropic SDK/
                        # model versions - e.g. a content block that is genuinely a
                        # tool use (content.type == "tool_use") can come back
                        # wrapped in a class that isn't exactly the ToolUseBlock
                        # symbol imported here (a beta/typed variant, a subclass,
                        # or a rebuilt pydantic model under a different module
                        # path), causing the strict check to silently miss it.
                        # When that happened, tool_calls stayed [] (not None)
                        # despite stop_reason == "tool_use", and client.py's
                        # `tool_calls is not None` check then passed a message
                        # with an empty-but-present "tool_calls" list all the way
                        # through to AG2's group tool executor, which crashes
                        # with UnboundLocalError (it assumes a non-empty list
                        # whenever the key is present at all).
                        if getattr(content, "type", None) == "tool_use":
                            tool_calls.append(
                                ChatCompletionMessageToolCall(
                                    id=content.id,
                                    function={"name": content.name, "arguments": json.dumps(content.input)},
                                    type="function",
                                )
                            )

                    # hep-theory fork: defensive fallback. If the API says
                    # stop_reason == "tool_use" but nothing above was recognized
                    # as a tool-use block (e.g. an as-yet-unseen content block
                    # shape), don't silently forward an empty-but-present
                    # tool_calls list downstream - it crashes the group tool
                    # executor with an unhelpful UnboundLocalError. Surface it
                    # loudly instead and fall back to a text-only message so the
                    # run can at least continue and be retried, rather than
                    # crashing the whole process.
                    if not tool_calls:
                        import warnings as _warnings
                        _warnings.warn(
                            f"Anthropic response for model {anthropic_params.get('model')} "
                            "had stop_reason='tool_use' but no content block was "
                            "recognized as a tool call. Treating this turn as if no "
                            "tool call was made - this usually means a new content "
                            "block shape isn't handled yet and needs investigating.",
                            UserWarning,
                        )
                        tool_calls = None
                else:
                    anthropic_finish = "stop"
                    tool_calls = None

                # Retrieve any text content from the response
                for content in response.content:
                    # hep-theory fork: same robustness fix as above - use the
                    # "type" discriminator instead of a strict class-identity
                    # check, for the same reasons.
                    if getattr(content, "type", None) == "text":
                        message_text = content.text
                        break

                # hep-theory fork: surfaced silent max_tokens truncation.
                # Previously, a response cut off at the token limit before
                # any TextBlock or completed ToolUseBlock existed produced
                # a totally empty {'content': '', 'tool_calls': None}
                # message with no error - deep_research() would finish
                # cleanly as if the turn had succeeded. Make this loud
                # instead: if nothing usable was extracted and the stop
                # reason was max_tokens, say so explicitly in the message
                # content and emit a warning, so it is visible in the
                # transcript and in stdout rather than silently swallowed.
                if not message_text and not tool_calls and response.stop_reason == "max_tokens":
                    import warnings as _warnings
                    _warnings.warn(
                        f"Anthropic response for model {anthropic_params.get('model')} "
                        f"was truncated at max_tokens ({anthropic_params.get('max_tokens')}) "
                        "before any usable text or tool call was produced. "
                        "This turn's output is lost - consider raising max_tokens "
                        "or splitting the task into smaller sub-steps.",
                        UserWarning,
                    )
                    message_text = (
                        "[TRUNCATED: this response hit the max_tokens limit "
                        f"({anthropic_params.get('max_tokens')}) before producing any "
                        "usable text or a complete tool call. No content was recovered. "
                        "This turn must be retried, ideally with a smaller sub-task or "
                        "a higher max_tokens setting.]"
                    )

        # Calculate and save the cost onto the response
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        # Convert output back to AG2 response format
        message = ChatCompletionMessage(
            role="assistant",
            content=message_text,
            function_call=None,
            tool_calls=tool_calls,
        )
        choices = [Choice(finish_reason=anthropic_finish, index=0, message=message)]

        response_oai = ChatCompletion(
            id=response.id,
            model=anthropic_params["model"],
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost=_calculate_cost(prompt_tokens, completion_tokens, anthropic_params["model"]),
        )

        return response_oai

    def message_retrieval(self, response) -> list:
        """Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        return [choice.message for choice in response.choices]

    @staticmethod
    def openai_func_to_anthropic(openai_func: dict) -> dict:
        res = openai_func.copy()
        res["input_schema"] = res.pop("parameters")
        return res

    @staticmethod
    def get_usage(response: ChatCompletion) -> dict:
        """Get the usage of tokens and their cost information."""
        return {
            "prompt_tokens": response.usage.prompt_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage is not None else 0,
            "total_tokens": response.usage.total_tokens if response.usage is not None else 0,
            "cost": response.cost if hasattr(response, "cost") else 0.0,
            "model": response.model,
        }

    @staticmethod
    def convert_tools_to_functions(tools: list) -> list:
        """
        Convert tool definitions into Anthropic-compatible functions,
        updating nested $ref paths in property schemas.

        Args:
            tools (list): List of tool definitions.

        Returns:
            list: List of functions with updated $ref paths.
        """

        def update_refs(obj, defs_keys, prop_name):
            """Recursively update $ref values that start with "#/$defs/"."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "$ref" and isinstance(value, str) and value.startswith("#/$defs/"):
                        ref_key = value[len("#/$defs/") :]
                        if ref_key in defs_keys:
                            obj[key] = f"#/properties/{prop_name}/$defs/{ref_key}"
                    else:
                        update_refs(value, defs_keys, prop_name)
            elif isinstance(obj, list):
                for item in obj:
                    update_refs(item, defs_keys, prop_name)

        functions = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                function = tool["function"]
                parameters = function.get("parameters", {})
                properties = parameters.get("properties", {})
                for prop_name, prop_schema in properties.items():
                    if "$defs" in prop_schema:
                        defs_keys = set(prop_schema["$defs"].keys())
                        update_refs(prop_schema, defs_keys, prop_name)
                functions.append(function)
        return functions

    def _add_response_format_to_system(self, params: dict[str, Any]):
        """Add prompt that will generate properly formatted JSON for structured outputs to system parameter.

        Based on Anthropic's JSON Mode cookbook, we ask the LLM to put the JSON within <json_response> tags.

        Args:
            params (dict): The client parameters
        """
        if not params.get("system"):
            return

        # Get the schema of the Pydantic model
        if isinstance(self._response_format, dict):
            schema = self._response_format
        else:
            schema = self._response_format.model_json_schema()

        # Add instructions for JSON formatting
        format_content = f"""Please provide your response as a JSON object that matches the following schema:
{json.dumps(schema, indent=2)}

Format your response as valid JSON within <json_response> tags.
Do not include any text before or after the tags.
Ensure the JSON is properly formatted and matches the schema exactly."""

        # Add formatting to last user message
        params["system"] += "\n\n" + format_content

    def _extract_json_response(self, response: Message) -> Any:
        """Extract and validate JSON response from the output for structured outputs.

        Args:
            response (Message): The response from the API.

        Returns:
            Any: The parsed JSON response.
        """
        if not self._response_format:
            return response

        # Extract content from response
        content = response.content[0].text if response.content else ""

        # Try to extract JSON from tags first
        json_match = re.search(r"<json_response>(.*?)</json_response>", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback to finding first JSON object
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start == -1 or json_end == -1:
                raise ValueError("No valid JSON found in response for Structured Output.")
            json_str = content[json_start : json_end + 1]

        try:
            # Parse JSON and validate against the Pydantic model if Pydantic model was provided
            json_data = json.loads(json_str)
            if isinstance(self._response_format, dict):
                return json_str
            else:
                return self._response_format.model_validate(json_data)

        except Exception as e:
            raise ValueError(f"Failed to parse response as valid JSON matching the schema for Structured Output: {e!s}")


def _format_json_response(response: Any) -> str:
    """Formats the JSON response for structured outputs using the format method if it exists."""
    if isinstance(response, str):
        return response
    elif isinstance(response, FormatterProtocol):
        return response.format()
    else:
        return response.model_dump_json()


def process_image_content(content_item: dict[str, Any]) -> dict[str, Any]:
    """Process an OpenAI image content item into Claude format."""
    if content_item["type"] != "image_url":
        return content_item

    url = content_item["image_url"]["url"]
    try:
        # Handle data URLs
        if url.startswith("data:"):
            data_url_pattern = r"data:image/([a-zA-Z]+);base64,(.+)"
            match = re.match(data_url_pattern, url)
            if match:
                media_type, base64_data = match.groups()
                return {
                    "type": "image",
                    "source": {"type": "base64", "media_type": f"image/{media_type}", "data": base64_data},
                }

        else:
            print("Error processing image.")
            # Return original content if image processing fails
            return content_item

    except Exception as e:
        print(f"Error processing image image: {e}")
        # Return original content if image processing fails
        return content_item


def process_message_content(message: dict[str, Any]) -> Union[str, list[dict[str, Any]]]:
    """Process message content, handling both string and list formats with images."""
    content = message.get("content", "")

    # Handle empty content
    if content == "":
        return content

    # If content is already a string, return as is
    if isinstance(content, str):
        return content

    # Handle list content (mixed text and images)
    if isinstance(content, list):
        processed_content = []
        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                processed_content.append(process_image_content(item))
        return processed_content

    return content


@require_optional_import("anthropic", "anthropic")
def oai_messages_to_anthropic_messages(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert messages from OAI format to Anthropic format.
    We correct for any specific role orders and types, etc.
    """
    # Track whether we have tools passed in. If not,  tool use / result messages should be converted to text messages.
    # Anthropic requires a tools parameter with the tools listed, if there are other messages with tool use or tool results.
    # This can occur when we don't need tool calling, such as for group chat speaker selection.
    has_tools = "tools" in params

    # Convert messages to Anthropic compliant format
    processed_messages = []

    # Used to interweave user messages to ensure user/assistant alternating
    user_continue_message = {"content": "Please continue.", "role": "user"}
    assistant_continue_message = {"content": "Please continue.", "role": "assistant"}

    tool_use_messages = 0
    tool_result_messages = 0
    last_tool_use_index = -1
    last_tool_result_index = -1
    for message in params["messages"]:
        if message["role"] == "system":
            content = process_message_content(message)
            if isinstance(content, list):
                # For system messages with images, concatenate only the text portions
                text_content = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
                params["system"] = params.get("system", "") + (" " if "system" in params else "") + text_content
            else:
                params["system"] = params.get("system", "") + ("\n" if "system" in params else "") + content
        else:
            # New messages will be added here, manage role alternations
            expected_role = "user" if len(processed_messages) % 2 == 0 else "assistant"

            if "tool_calls" in message:
                # Map the tool call options to Anthropic's ToolUseBlock
                tool_uses = []
                tool_names = []
                for tool_call in message["tool_calls"]:
                    tool_uses.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=tool_call["id"],
                            name=tool_call["function"]["name"],
                            input=json.loads(tool_call["function"]["arguments"]),
                        )
                    )
                    if has_tools:
                        tool_use_messages += 1
                    tool_names.append(tool_call["function"]["name"])

                if expected_role == "user":
                    # Insert an extra user message as we will append an assistant message
                    processed_messages.append(user_continue_message)

                if has_tools:
                    processed_messages.append({"role": "assistant", "content": tool_uses})
                    last_tool_use_index = len(processed_messages) - 1
                else:
                    # Not using tools, so put in a plain text message
                    processed_messages.append({
                        "role": "assistant",
                        "content": f"Some internal function(s) that could be used: [{', '.join(tool_names)}]",
                    })
            elif "tool_call_id" in message:
                if has_tools:
                    # Map the tool usage call to tool_result for Anthropic
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    }

                    # If the previous message also had a tool_result, add it to that
                    # Otherwise append a new message
                    if last_tool_result_index == len(processed_messages) - 1:
                        processed_messages[-1]["content"].append(tool_result)
                    else:
                        if expected_role == "assistant":
                            # Insert an extra assistant message as we will append a user message
                            processed_messages.append(assistant_continue_message)

                        processed_messages.append({"role": "user", "content": [tool_result]})
                        last_tool_result_index = len(processed_messages) - 1

                    tool_result_messages += 1
                else:
                    # Not using tools, so put in a plain text message
                    processed_messages.append({
                        "role": "user",
                        "content": f"Running the function returned: {message['content']}",
                    })
            elif message["content"] == "":
                # Ignoring empty messages
                pass
            else:
                if expected_role != message["role"]:
                    # Inserting the alternating continue message
                    processed_messages.append(
                        user_continue_message if expected_role == "user" else assistant_continue_message
                    )
                # Process messages for images
                processed_content = process_message_content(message)
                processed_message = message.copy()
                processed_message["content"] = processed_content
                processed_messages.append(processed_message)

    # We'll replace the last tool_use if there's no tool_result (occurs if we finish the conversation before running the function)
    if has_tools and tool_use_messages != tool_result_messages:
        processed_messages[last_tool_use_index] = assistant_continue_message

    # name is not a valid field on messages
    for message in processed_messages:
        if "name" in message:
            message.pop("name", None)

    # Note: When using reflection_with_llm we may end up with an "assistant" message as the last message and that may cause a blank response
    # So, if the last role is not user, add a 'user' continue message at the end
    # hep-theory fork: empty processed_messages fix - an empty list trivially
    # doesn't end in "user", so it should get the continue message appended,
    # not crash with IndexError (happens during forced-termination sequences
    # at max_rounds).
    if not processed_messages or processed_messages[-1]["role"] != "user":
        processed_messages.append(user_continue_message)

    return processed_messages


def _cmbagent_call_record(params: dict[str, Any], response: Any) -> dict[str, Any]:
    """hep-theory fork: per-call usage log record (Phase 0 of the caching plan).

    Pure observer: reads ``params`` and ``response`` and never mutates them.
    Token counts are recorded only if they are real ints (a mocked or absent
    usage field gives ``None``). Hashes cover the full system block, the tools
    list, the first message and the ``thinking`` / ``tool_choice`` options (all
    of which take part in cache validity), so prefix changes between calls are
    visible. Never raises: on any failure it returns ``{"error": repr(e)}``.
    """
    try:
        import hashlib

        def _h(obj: Any) -> Optional[str]:
            if obj is None:
                return None
            s = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, default=str)
            return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]

        usage = getattr(response, "usage", None)

        def _tokens(name: str) -> Optional[int]:
            v = getattr(usage, name, None)
            return v if isinstance(v, int) and not isinstance(v, bool) else None

        system = params.get("system", "")
        messages = params.get("messages") or []
        return {
            "model": params.get("model"),
            "input_tokens": _tokens("input_tokens"),
            "cache_creation_input_tokens": _tokens("cache_creation_input_tokens"),
            "cache_read_input_tokens": _tokens("cache_read_input_tokens"),
            "output_tokens": _tokens("output_tokens"),
            "system_sha256": _h(system),
            "system_chars": len(system) if isinstance(system, str) else len(json.dumps(system, default=str)),
            "tools_sha256": _h(params.get("tools")),
            "cache_opts_sha256": _h({"thinking": params.get("thinking"), "tool_choice": params.get("tool_choice")}),
            "n_messages": len(messages),
            "first_msg_sha256": _h(messages[0]) if messages else None,
            "stop_reason": getattr(response, "stop_reason", None),
        }
    except Exception as e:  # never let logging affect a call
        return {"error": repr(e)}


def _calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of the completion using the Anthropic pricing."""
    total = 0.0

    if model in ANTHROPIC_PRICING_1k:
        input_cost_per_1k, output_cost_per_1k = ANTHROPIC_PRICING_1k[model]
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total = input_cost + output_cost
    else:
        warnings.warn(f"Cost calculation not available for model {model}", UserWarning)

    return total
