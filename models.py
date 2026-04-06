from dataclasses import dataclass, field
from typing import Optional, List
from openenv.core.env_server import Action, Observation, State


@dataclass
class TriageAction(Action):
    """Action the agent takes: classify, prioritize, or reply to emails."""
    action_type: str = ""          # "classify" | "prioritize" | "reply"
    email_id: str = ""             # which email to act on
    category: str = ""             # for classify: "billing"|"support"|"spam"|"inquiry"
    priority_order: List[str] = field(default_factory=list)  # for prioritize: ordered list of email_ids
    reply_text: str = ""           # for reply: the draft response text


@dataclass
class TriageObservation(Observation):
    """What the agent sees after each action."""
    task_id: int = 0
    task_description: str = ""
    emails: List[dict] = field(default_factory=list)
    last_action_result: str = ""
    score: float = 0.0
    done: bool = False
    feedback: str = ""


@dataclass
class TriageState(State):
    """Full internal state (for debugging/grading)."""
    current_task: int = 0
    steps_taken: int = 0
    total_reward: float = 0.0
    task_scores: List[float] = field(default_factory=list)
