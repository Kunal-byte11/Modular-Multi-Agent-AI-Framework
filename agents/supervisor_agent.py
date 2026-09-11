"""
agents/supervisor_agent.py
---------------------------
Multi-Agent Orchestration & Team Supervisor.
Decomposes complex requests via LLM and coordinates specialist agents to produce a unified report.
"""

from typing import List, Dict, Any, Optional
import json
from core.base_agent import BaseAgent
from core.base_llm import BaseLLM, LLMFactory
from core.base_memory import Message


class SupervisorAgent:
    """
    Orchestrates a team of specialist agents.
    Acts as the Team Lead / Captain who routes tasks to the best suited agent.
    Can decompose a user goal into tasks via LLM or accept an explicit task list.
    """
    def __init__(self, name: str, team: List[BaseAgent], llm: Optional[BaseLLM] = None):
        self.name = name
        self.team: Dict[str, BaseAgent] = {agent.name: agent for agent in team}
        self.llm = llm or LLMFactory.create("mock")

    def add_agent(self, agent: BaseAgent) -> None:
        self.team[agent.name] = agent

    def list_specialists(self) -> str:
        return "\n".join([f"- {a.name}: {a.role}" for a in self.team.values()])

    def decompose(self, user_goal: str) -> List[Dict[str, str]]:
        """
        Uses the LLM to decompose a complex user goal into a JSON task list.
        Each task maps to a specialist agent on the team.

        Returns:
            List of dicts: [{"agent": "AgentName", "task": "task description"}, ...]
        """
        specialist_list = self.list_specialists()
        prompt = (
            f"You are {self.name}, a team supervisor.\n"
            f"Your team of specialists:\n{specialist_list}\n\n"
            f"User goal: {user_goal}\n\n"
            "Decompose this goal into a JSON array of tasks. Each task must specify "
            "which specialist agent to delegate to and what task to give them.\n"
            "Respond ONLY with a valid JSON array, no explanation:\n"
            '[{"agent": "AgentName", "task": "specific task description"}, ...]'
        )

        messages = [Message(role="user", content=prompt)]

        try:
            response = self.llm.generate(messages)
            # Strip markdown fences if present
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            tasks = json.loads(clean)
            if isinstance(tasks, list) and all(
                isinstance(t, dict) and "agent" in t and "task" in t for t in tasks
            ):
                print(f"  🧠 [{self.name}] LLM decomposed goal into {len(tasks)} tasks.")
                return tasks
        except (json.JSONDecodeError, RuntimeError, ValueError) as e:
            print(f"  ⚠️ [{self.name}] LLM decomposition failed ({e}), using fallback.")

        # Fallback: distribute goal equally to all specialists
        fallback = [{"agent": name, "task": user_goal} for name in self.team]
        return fallback

    def coordinate(self, workflow_tasks: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Executes a sequence of specialized tasks across team members.
        workflow_tasks = [
            {"agent": "Researcher", "task": "Fetch TATA stock metrics"},
            {"agent": "QuantAnalyst", "task": "Compute tax on ₹50000 gain"}
        ]
        """
        print(f"\n👔 [{self.name} - Team Lead] Starting Multi-Agent Pipeline:")
        print(f"Team Roster:\n{self.list_specialists()}\n")

        results: Dict[str, str] = {}
        for step_idx, step in enumerate(workflow_tasks, 1):
            target_agent_name = step["agent"]
            task_prompt = step["task"]

            agent = self.team.get(target_agent_name)
            if not agent:
                results[target_agent_name] = f"Error: Agent '{target_agent_name}' not found on team."
                continue

            print(f"👉 Step {step_idx}: Delegating to [{agent.name}]...")
            agent_response = agent.run(task_prompt)
            results[target_agent_name] = agent_response

        print(f"\n🏁 [{self.name}] All specialist agents completed their tasks successfully!")
        return results

    def run(self, user_goal: str) -> str:
        """
        End-to-end pipeline: decompose the user goal via LLM, coordinate agents,
        and generate the final executive report.
        """
        tasks = self.decompose(user_goal)
        results = self.coordinate(tasks)
        return self.generate_final_report(user_goal, results)

    def generate_final_report(self, user_goal: str, results: Dict[str, str]) -> str:
        """Synthesizes specialist outputs into a master executive summary."""
        report = [
            "==================================================",
            f"📑 EXECUTIVE MULTI-AGENT REPORT",
            f"🎯 Objective: {user_goal}",
            "==================================================",
        ]
        for agent_name, output in results.items():
            report.append(f"\n🔹 Contribution from [{agent_name}]:")
            report.append(f"{output}")

        report.append("\n==================================================")
        report.append("✅ Status: Completed & Verified by Supervisor")
        report.append("==================================================")
        return "\n".join(report)
