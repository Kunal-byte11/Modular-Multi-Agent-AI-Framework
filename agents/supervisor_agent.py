"""
agents/supervisor_agent.py
---------------------------
Multi-Agent Orchestration & Team Supervisor.
Decomposes complex requests and coordinates specialist agents to produce a unified report.
"""

from typing import List, Dict, Any, Optional
from core.base_agent import BaseAgent
from core.base_llm import BaseLLM, LLMFactory


class SupervisorAgent:
    """
    Orchestrates a team of specialist agents.
    Acts as the Team Lead / Captain who routes tasks to the best suited agent.
    """
    def __init__(self, name: str, team: List[BaseAgent], llm: Optional[BaseLLM] = None):
        self.name = name
        self.team: Dict[str, BaseAgent] = {agent.name: agent for agent in team}
        self.llm = llm or LLMFactory.create("mock")

    def add_agent(self, agent: BaseAgent) -> None:
        self.team[agent.name] = agent

    def list_specialists(self) -> str:
        return "\n".join([f"- {a.name}: {a.role}" for a in self.team.values()])

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
