import aiohttp
import asyncio
import json
from enum import Enum
from typing import Any, Callable, Optional
from aiohttp import web
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
import time
import random

class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2

class ToolResult:
    text: str
    destination: ToolResultDirection

    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if type(self.text) == str else json.dumps(self.text)

class Tool:
    target: Callable[..., ToolResult]
    schema: Any

    def __init__(self, target: Any, schema: Any):
        self.target = target
        self.schema = schema

class RTToolCall:
    tool_call_id: str
    previous_id: str

    def __init__(self, tool_call_id: str, previous_id: str):
        self.tool_call_id = tool_call_id
        self.previous_id = previous_id

class RTMiddleTier:
    endpoint: str
    deployment: str
    key: Optional[str] = None
    
    tools: dict[str, Tool] = {}

    model: Optional[str] = None
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    disable_audio: Optional[bool] = None

    _tools_pending = {}
    _token_provider = None

    def __init__(self, endpoint: str, deployment: str, credentials: AzureKeyCredential | DefaultAzureCredential):
        self.endpoint = endpoint
        self.deployment = deployment
        if isinstance(credentials, AzureKeyCredential):
            self.key = credentials.key
        else:
            self._token_provider = get_bearer_token_provider(credentials, "https://cognitiveservices.azure.com/.default")
            self._token_provider()

    async def _process_message_to_client(self, msg: str, client_ws: web.WebSocketResponse, server_ws: web.WebSocketResponse) -> Optional[str]:
        message = json.loads(msg.data)
        updated_message = msg.data
        if message is not None:
            match message["type"]:
                case "session.created":
                    session = message["session"]
                    # Hide the instructions, tools and max tokens from clients, if we ever allow client-side 
                    # tools, this will need updating
                    session["instructions"] = ""
                    session["tools"] = []
                    session["tool_choice"] = "none"
                    session["max_response_output_tokens"] = None
                    updated_message = json.dumps(message)

                case "response.output_item.added":
                    if "item" in message and message["item"]["type"] == "function_call":
                        updated_message = None

                case "conversation.item.created":
                    if "item" in message and message["item"]["type"] == "function_call":
                        item = message["item"]
                        if item["call_id"] not in self._tools_pending:
                            self._tools_pending[item["call_id"]] = RTToolCall(item["call_id"], message["previous_item_id"])
                        updated_message = None
                    elif "item" in message and message["item"]["type"] == "function_call_output":
                        updated_message = None

                case "response.function_call_arguments.delta":
                    updated_message = None
                
                case "response.function_call_arguments.done":
                    updated_message = None

                case "response.output_item.done":
                    if "item" in message and message["item"]["type"] == "function_call":
                        item = message["item"]
                        tool_call = self._tools_pending[message["item"]["call_id"]]
                        tool = self.tools[item["name"]]
                        args = item["arguments"]
                        print("query", args)
                        try:
                            result = tool.target(json.loads(args))  # Decoding the JSON
                        except json.JSONDecodeError as e:
                            result = ToolResult(
                                destination=ToolResultDirection.TO_SERVER,
                            )
                        await server_ws.send_json({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": item["call_id"],
                                "output": result.to_text() if result.destination == ToolResultDirection.TO_SERVER else ""
                            }
                        })
                        if result.destination == ToolResultDirection.TO_CLIENT:
                            # TODO: this will break clients that don't know about this extra message, rewrite 
                            # this to be a regular text message with a special marker of some sort
                            await client_ws.send_json({
                                "type": "extension.middle_tier_tool_response",
                                "previous_item_id": tool_call.previous_id,
                                "tool_name": item["name"],
                                "tool_result": result.to_text()
                            })
                        updated_message = None

                case "response.done":
                    if len(self._tools_pending) > 0:
                        self._tools_pending.clear() # Any chance tool calls could be interleaved across different outstanding responses?
                        await server_ws.send_json({
                            "type": "response.create"
                        })
                    if "response" in message:
                        replace = False
                        for i, output in enumerate(reversed(message["response"]["output"])):
                            if output["type"] == "function_call":
                                message["response"]["output"].pop(i)
                                replace = True
                        if replace:
                            updated_message = json.dumps(message)                        

        return updated_message

    async def _process_message_to_server(self, msg: str, ws: web.WebSocketResponse) -> Optional[str]:
        message = json.loads(msg.data)
        updated_message = msg.data
        if message is not None:
            match message["type"]:
                case "session.update":
                    session = message["session"]
                    if self.system_message is not None:
                        session["instructions"] = self.system_message
                    if self.temperature is not None:
                        session["temperature"] = self.temperature
                    if self.max_tokens is not None:
                        session["max_response_output_tokens"] = self.max_tokens
                    if self.disable_audio is not None:
                        session["disable_audio"] = self.disable_audio
                    session["tool_choice"] = "auto" if len(self.tools) > 0 else "none"
                    session["tools"] = [tool.schema for tool in self.tools.values()]
                    updated_message = json.dumps(message)

        return updated_message

    async def _forward_messages(self, ws: web.WebSocketResponse):
        async with aiohttp.ClientSession(base_url=self.endpoint) as session:
            params = {"api-version": "2024-10-01-preview", "deployment": self.deployment}
            headers = {}
            
            if "x-ms-client-request-id" in ws.headers:
                headers["x-ms-client-request-id"] = ws.headers["x-ms-client-request-id"]
            
            if self.key is not None:
                headers = {"api-key": self.key}
            else:
                headers = {"Authorization": f"Bearer {self._token_provider()}"}

            # Retry logic
            max_retries = 20  # You can adjust this value based on your needs
            retry_count = 0
            backoff_factor = 1  # Exponential backoff factor

            while retry_count < max_retries:
                try:
                    async with session.ws_connect("/openai/realtime", headers=headers, params=params) as target_ws:
                        async def from_client_to_server():
                            try:
                                async for msg in ws:
                                    try:
                                        if msg.type == aiohttp.WSMsgType.TEXT:
                                            new_msg = await self._process_message_to_server(msg, ws)
                                            if new_msg is not None:
                                                await target_ws.send_str(new_msg)
                                        else:
                                            print("Error: unexpected message type:", msg.type)
                                    except Exception as e:
                                        print(f"Error in client-to-server communication: {e}")
                                        continue  # Skip to the next message and continue processing
                            except RuntimeError as e:
                                if "WebSocket connection is closed" in str(e):
                                    print("Client WebSocket connection closed.")

                        async def from_server_to_client():
                            try:
                                async for msg in target_ws:
                                    try:
                                        if msg.type == aiohttp.WSMsgType.TEXT:
                                            new_msg = await self._process_message_to_client(msg, ws, target_ws)
                                            if new_msg is not None:
                                                await ws.send_str(new_msg)
                                        else:
                                            print("Error: unexpected message type:", msg.type)
                                    except Exception as e:
                                        print(f"Error in server-to-client communication: {e}")
                                        continue  # Skip to the next message and continue processing
                            except RuntimeError as e:
                                if "WebSocket connection is closed" in str(e):
                                    print("Server WebSocket connection closed.")

                        try:
                            await asyncio.gather(from_client_to_server(), from_server_to_client())
                            break  # Exit the loop if the connection is successful
                        except ConnectionResetError:
                            pass  # Ignore the errors resulting from the client disconnecting the socket

                except aiohttp.client_exceptions.WSServerHandshakeError as e:
                    if e.status == 429:  # Too many requests error
                        retry_count += 1
                        # Exponential backoff with jitter
                        sleep_time = backoff_factor * (retry_count) + random.uniform(0, 1)
                        print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise  # If it's another error, raise it

            if retry_count == max_retries:
                print("Max retries reached. Unable to connect to the server.")           

    async def _websocket_handler(self, request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self._forward_messages(ws)
        return ws
    
    def attach_to_app(self, app, path):
        app.router.add_get(path, self._websocket_handler)
