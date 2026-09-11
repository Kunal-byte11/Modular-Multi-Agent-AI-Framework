"""
agents/react_agent.py
----------------------
Autonomous ReAct (Reasoning + Acting) Agent implementation.
Strict tool enforcement, step-by-step trace collection, and loop circuit breakers.
"""

from typing import List, Optional, Tuple, Dict, Any
import re
from core.base_agent import BaseAgent
from core.base_tool import BaseTool
from core.base_memory import BaseMemory
from core.base_llm import BaseLLM


class ReActAgent(BaseAgent):
    """
    An agent that strictly implements the ReAct loop:
    1. THOUGHT: Analyzes the goal
    2. ACTION: Calls an appropriate tool
    3. OBSERVATION: Captures tool response
    4. FINAL ANSWER: Returns synthesis based on real tool output
    """

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in self._tools.values()])
        return (
            f"You are {self.name}, a {self.role}.\n"
            f"{self.system_prompt}\n\n"
            f"Available Tools:\n{tools_desc or 'None'}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You do NOT have real-time live data or calculation authority on your own. You MUST NOT answer using your own assumptions or pre-training memory.\n"
            "2. If tools are available, your VERY FIRST response MUST ALWAYS call a tool using the ACTION syntax:\n"
            "   THOUGHT: <explain which tool you need and why>\n"
            "   ACTION: tool_name(arguments)\n"
            "3. Wait for the OBSERVATION from the tool.\n"
            "4. Only after receiving the OBSERVATION may you write:\n"
            "   FINAL ANSWER: <concise summary based directly on the tool observation>\n\n"
            "EXAMPLE INTERACTION:\n"
            "User: Screen the banking sector.\n"
            "Assistant:\n"
            "THOUGHT: I need to screen the banking sector using indian_market_screener.\n"
            "ACTION: indian_market_screener(banking)\n"
            "Observation: Top Pick: SBI (CMP: ₹780, Net Profit Up: +15%)\n"
            "Assistant:\n"
            "THOUGHT: I have received the verified data from the tool.\n"
            "FINAL ANSWER: Based on our market screener, SBI is the top banking pick at CMP ₹780."
        )

    def run(self, user_query: str) -> str:
        # Reset memory for fresh task
        self.memory.clear()
        self.memory.add_message("system", self._build_system_prompt())
        self.memory.add_user_message(user_query)

        print(f"\n🤖 [{self.name}] Started processing: '{user_query}'")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"  🔄 Loop Step {iteration}/{self.max_iterations}")

            context = self.memory.get_context_window()
            response = self.llm.generate(context)

            # Strip any markdown code fences if model wraps output
            clean_resp = response.replace("```text", "").replace("```json", "").replace("```", "").strip()

            # 1. Check if tool action is present
            action_match = re.search(r"ACTION:\s*([a-zA-Z0-9_]+)\((.*?)\)", clean_resp, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                raw_args = action_match.group(2).strip()

                print(f"  🛠️ [{self.name}] Executing Tool: {tool_name}({raw_args})")
                tool_obj = self.get_tool(tool_name)

                if tool_obj:
                    # Clean up argument list
                    args = [a.strip().strip("'\"") for a in raw_args.split(",") if a.strip()]
                    try:
                        observation = tool_obj.execute(*args)
                    except Exception as err:
                        observation = f"Tool execution failed: {str(err)}"
                else:
                    observation = f"Error: Tool '{tool_name}' does not exist in registry."

                print(f"  👁️ [{self.name}] Observation: {observation}")

                # Save thought and observation to memory
                self.memory.add_assistant_message(clean_resp)
                self.memory.add_user_message(
                    f"OBSERVATION: {observation}\n"
                    "Now synthesize this observation and produce your 'FINAL ANSWER:'."
                )
                continue

            # 2. Check if Final Answer is reached
            if "FINAL ANSWER:" in clean_resp:
                final_answer = clean_resp.split("FINAL ANSWER:")[-1].strip()
                self.memory.add_assistant_message(clean_resp)
                print(f"  ✨ [{self.name}] Reached Final Answer!")
                return final_answer

            # 3. If model tries to bypass tool on step 1, enforce ReAct
            if iteration == 1 and self._tools:
                tool_names = ", ".join(self._tools.keys())
                self.memory.add_assistant_message(clean_resp)
                self.memory.add_user_message(
                    f"SYSTEM ERROR: You did not call a tool! You MUST call one of [{tool_names}] using 'ACTION: tool_name(args)'."
                )
                continue

            # Return direct answer if no tools needed
            self.memory.add_assistant_message(clean_resp)
            return clean_resp

        return f"⚠️ [{self.name}] Maximum iterations ({self.max_iterations}) reached."
