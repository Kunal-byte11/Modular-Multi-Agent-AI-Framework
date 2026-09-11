"""
agents/react_agent.py
----------------------
Autonomous ReAct (Reasoning + Acting) Agent implementation.
Parses LLM thoughts and executes tools dynamically in a safety-bounded loop.
"""

from typing import List, Optional
import re
from core.base_agent import BaseAgent
from core.base_tool import BaseTool
from core.base_memory import BaseMemory
from core.base_llm import BaseLLM


class ReActAgent(BaseAgent):
    """
    An agent that implements the ReAct loop:
    1. THOUGHT: Analyzes the current goal
    2. ACTION: Calls an appropriate tool
    3. OBSERVATION: Records tool result into memory
    4. Repeats until 'FINAL ANSWER:' is produced or max_iterations reached.
    """

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in self._tools.values()])
        return (
            f"You are {self.name}, a {self.role}.\n"
            f"{self.system_prompt}\n\n"
            f"Available Tools:\n{tools_desc or 'None'}\n\n"
            "You MUST follow this exact format:\n"
            "THOUGHT: <your reasoning step>\n"
            "ACTION: tool_name(argument)\n"
            "OBSERVATION: <tool output will be provided here>\n"
            "... (repeat THOUGHT/ACTION/OBSERVATION if needed)\n"
            "FINAL ANSWER: <your complete final response to the user>"
        )

    def run(self, user_query: str) -> str:
        # 1. Initialize memory with system instructions and user request
        if len(self.memory) == 0:
            self.memory.add_message("system", self._build_system_prompt())
        self.memory.add_user_message(user_query)

        print(f"\n🤖 [{self.name}] Started processing: '{user_query}'")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"  🔄 Loop Step {iteration}/{self.max_iterations}")

            # 2. Ask LLM for next thought / action
            context = self.memory.get_context_window()
            response = self.llm.generate(context)

            # 3. Check if final answer is reached
            if "FINAL ANSWER:" in response:
                final_answer = response.split("FINAL ANSWER:")[-1].strip()
                self.memory.add_assistant_message(response)
                print(f"  ✨ [{self.name}] Reached Final Answer!")
                return final_answer

            # 4. Parse ACTION: tool_name(arg1, arg2...)
            action_match = re.search(r"ACTION:\s*([a-zA-Z0-9_]+)\((.*)\)", response)
            if action_match:
                tool_name = action_match.group(1).strip()
                raw_args = action_match.group(2).strip()

                print(f"  🛠️ [{self.name}] Executing Tool: {tool_name}({raw_args})")
                tool_obj = self.get_tool(tool_name)

                if tool_obj:
                    # Clean arguments
                    args = [a.strip().strip("'\"") for a in raw_args.split(",") if a.strip()]
                    try:
                        observation = tool_obj.execute(*args)
                    except Exception as err:
                        observation = f"Tool execution failed: {str(err)}"
                else:
                    observation = f"Error: Tool '{tool_name}' does not exist."

                print(f"  👁️ [{self.name}] Observation: {observation}")
                
                # Add reasoning and observation to memory
                self.memory.add_assistant_message(response)
                self.memory.add_tool_message(tool_name, f"OBSERVATION: {observation}")
            else:
                # Direct response without tools
                self.memory.add_assistant_message(response)
                return response

        return f"⚠️ [{self.name}] Warning: Maximum iterations ({self.max_iterations}) exceeded before finding final answer."
