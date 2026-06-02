"""Agent runtime for multi-step reasoning over the knowledge base.

The agent runtime executes a task by repeatedly deciding whether to call tools
(e.g., KB search) or return a final answer. It coordinates LLM interactions
with tool execution and logs activity for transparency.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator

from twin.llm.base import LLMProvider, LLMResponse, ToolCall
from twin.agent.tools import ToolDispatcher
from twin.agent.log import AgentLog
from twin.rag.prompts import SystemPrompts


@dataclass
class AgentOutput:
    """Output from the agent runtime."""

    final_answer: str
    """The agent's final answer to the task."""

    tool_calls: int
    """Number of tool calls made during execution."""

    activity_log: list[dict]
    """Detailed log of each agent step and tool call."""


class AgentRuntime:
    """
    Orchestrates multi-step reasoning using an LLM and tool access.

    The runtime maintains a conversation with an LLM, routes tool calls
    through the dispatcher, and terminates when the LLM returns a final
    answer or the iteration limit is reached.
    """

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(
        self,
        llm: LLMProvider,
        tool_dispatcher: ToolDispatcher,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """
        Initialize the agent runtime.

        Args:
            llm: LLMProvider instance for generating responses.
            tool_dispatcher: ToolDispatcher for routing and executing tool calls.
            max_iterations: Maximum number of tool call iterations (default: 5).
        """
        self._llm = llm
        self._tool_dispatcher = tool_dispatcher
        self._max_iterations = max_iterations
        self._log = AgentLog()

    async def execute(self, task: str) -> AgentOutput:
        """
        Execute a task using the agent loop.

        The loop:
        1. Sends the task and conversation history to the LLM.
        2. If the response contains tool calls, executes them and continues.
        3. If the response is a final answer, returns it.
        4. Stops after max_iterations tool calls even without a final answer.

        Args:
            task: The user's task description.

        Returns:
            AgentOutput with the final answer, tool call count, and activity log.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        tool_calls_made = 0
        response = LLMResponse(content="")  # sentinel; overwritten on first iteration

        for iteration in range(self._max_iterations):
            tool_defs = self._tool_dispatcher.get_tool_definitions()
            response = await self._llm.complete(
                messages=messages,
                tools=tool_defs,
                system=SystemPrompts.AGENT_SYSTEM,
            )
            self._log.log_llm_response(iteration, response)

            if self._has_tool_call(response):
                tool_call = self._extract_tool_call(response)
                tool_calls_made += 1

                self._log.log_tool_call(iteration, tool_call.name, tool_call.input)
                tool_result = self._tool_dispatcher.dispatch(tool_call.name, tool_call.input)
                self._log.log_tool_result(iteration, tool_result)

                # Append tool exchange in Anthropic content-block format.
                # Non-Anthropic providers convert this in their complete() methods.
                assistant_content: list[dict] = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_result,
                    }],
                })
            else:
                final_answer = self._llm.extract_answer(response)
                self._log.log_final_answer(iteration, final_answer)
                return AgentOutput(
                    final_answer=final_answer,
                    tool_calls=tool_calls_made,
                    activity_log=self._log.get_log(),
                )

        # Max iterations reached without a final answer.
        final_answer = self._llm.extract_answer(response)
        self._log.log_final_answer(
            self._max_iterations - 1, final_answer, reason="max_iterations"
        )
        return AgentOutput(
            final_answer=final_answer,
            tool_calls=tool_calls_made,
            activity_log=self._log.get_log(),
        )

    async def execute_stream(self, task: str) -> AsyncIterator[dict[str, Any]]:
        """
        Execute a task, yielding tool-activity events then streaming the final answer.

        Tool-call iterations use complete() (needed to detect tool calls). Once
        no further tool calls are requested, the final answer is streamed via
        provider.stream().

        Yields:
            Dicts keyed on 'type':
            - {"type": "tool_call", "name": str, "input": dict, "result": str,
               "iteration": int}
            - {"type": "token", "text": str}
            - {"type": "done", "tool_calls": int, "activity_log": list[dict]}

        Args:
            task: The user's task description.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        tool_calls_made = 0

        for iteration in range(self._max_iterations):
            tool_defs = self._tool_dispatcher.get_tool_definitions()
            response = await self._llm.complete(
                messages=messages,
                tools=tool_defs,
                system=SystemPrompts.AGENT_SYSTEM,
            )
            self._log.log_llm_response(iteration, response)

            if self._has_tool_call(response):
                tool_call = self._extract_tool_call(response)
                tool_calls_made += 1

                self._log.log_tool_call(iteration, tool_call.name, tool_call.input)
                tool_result = self._tool_dispatcher.dispatch(tool_call.name, tool_call.input)
                self._log.log_tool_result(iteration, tool_result)

                yield {
                    "type": "tool_call",
                    "name": tool_call.name,
                    "input": tool_call.input,
                    "result": tool_result,
                    "iteration": iteration,
                }

                assistant_content: list[dict] = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_result,
                    }],
                })
            else:
                # No tool call — stream the final answer
                self._log.log_final_answer(iteration, "")
                async for token in self._llm.stream(
                    messages=messages,
                    tools=None,
                    system=SystemPrompts.AGENT_SYSTEM,
                ):
                    yield {"type": "token", "text": token}
                yield {
                    "type": "done",
                    "tool_calls": tool_calls_made,
                    "activity_log": self._log.get_log(),
                }
                return

        # Max iterations reached — stream whatever the model says next
        self._log.log_final_answer(
            self._max_iterations - 1, "", reason="max_iterations"
        )
        async for token in self._llm.stream(
            messages=messages,
            tools=None,
            system=SystemPrompts.AGENT_SYSTEM,
        ):
            yield {"type": "token", "text": token}
        yield {
            "type": "done",
            "tool_calls": tool_calls_made,
            "activity_log": self._log.get_log(),
        }

    def _has_tool_call(self, response: LLMResponse) -> bool:
        """
        Check whether the LLM response contains tool calls.

        Args:
            response: Normalized LLMResponse from complete().

        Returns:
            True if the response requests at least one tool call.
        """
        return bool(response.tool_calls)

    def _extract_tool_call(self, response: LLMResponse) -> ToolCall:
        """
        Return the first tool call from a response.

        Args:
            response: Normalized LLMResponse containing at least one tool call.

        Returns:
            The first ToolCall in the response.

        Raises:
            ValueError: If the response contains no tool calls.
        """
        if not response.tool_calls:
            raise ValueError("No tool call found in response")
        return response.tool_calls[0]
