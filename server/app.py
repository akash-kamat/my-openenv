
import json
import copy
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.email_triage_environment import (
    EMAILS,
    grade_classify,
    grade_prioritize,
    grade_reply,
    compute_step_reward,
)


def _clamp_task_score(score: float) -> float:
    """Ensure task scores are strictly between 0 and 1 (not 0.0 or 1.0)."""
    return round(0.05 + (max(0.0, min(1.0, score)) * 0.90), 3)

app = FastAPI(title="Email Triage Environment", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_state = {
    "current_task": 1,       # 1=classify, 2=prioritize, 3=reply
    "steps_taken": 0,
    "total_reward": 0.0,
    "task_scores": [],        # score per task
    "done": False,
    "classified": {},         # track classify results per email
    "classify_correct": 0,
}

TASK_DESCRIPTIONS = {
    1: (
        "TASK 1 (Easy) - Email Classification:\n"
        "Classify each of the 5 emails into one of these categories: "
        "'billing', 'support', 'spam', 'inquiry'.\n"
        "Send one action per email with action_type='classify', email_id=<id>, category=<your_category>.\n"
        "You need to classify all 5 emails (e1 through e5)."
    ),
    2: (
        "TASK 2 (Medium) - Priority Ordering:\n"
        "Order all 5 emails from most to least urgent.\n"
        "Send one action with action_type='prioritize', priority_order=[...list of email_ids...]"
    ),
    3: (
        "TASK 3 (Hard) - Draft a Reply:\n"
        "Write a professional customer support reply to email e2 (billing dispute: "
        "customer was charged $299 but plan is $99/month).\n"
        "Send action_type='reply', email_id='e2', reply_text='<your reply>'.\n"
        "A good reply apologizes, acknowledges the overcharge, promises a refund, gives a timeline, and closes professionally."
    ),
}


def _get_observation(last_result: str = "", score: float = 0.0, done: bool = False, feedback: str = ""):
    task = _state["current_task"]
    return {
        "task_id": task,
        "task_description": TASK_DESCRIPTIONS.get(task, "All tasks complete."),
        "emails": EMAILS,
        "last_action_result": last_result,
        "score": score,
        "done": done,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class ActionRequest(BaseModel):
    action_type: str = ""
    email_id: str = ""
    category: str = ""
    priority_order: list = []
    reply_text: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset():
    global _state
    _state = {
        "current_task": 1,
        "steps_taken": 0,
        "total_reward": 0.0,
        "task_scores": [],
        "done": False,
        "classified": {},
        "classify_correct": 0,
    }
    obs = _get_observation(last_result="Environment reset. Start with Task 1.")
    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {"message": "Reset successful"},
    }


@app.post("/step")
def step(action: ActionRequest):
    if _state["done"]:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset to start again.")

    _state["steps_taken"] += 1
    task = _state["current_task"]
    reward = 0.0
    feedback = ""
    advance_task = False

    # ------------------------------------------------------------------
    # Task 1: Classify emails (one at a time)
    # ------------------------------------------------------------------
    if task == 1:
        if action.action_type != "classify":
            return _error_response("For Task 1 use action_type='classify'")

        score, feedback = grade_classify(action.email_id, action.category)
        reward = compute_step_reward(1, score, _state["steps_taken"])
        _state["total_reward"] += reward

        _state["classified"][action.email_id] = score
        if score == 1.0:
            _state["classify_correct"] += 1

        # Advance when all 5 classified
        classified_ids = set(_state["classified"].keys())
        all_ids = {"e1", "e2", "e3", "e4", "e5"}

        if classified_ids >= all_ids:
            task1_score = _clamp_task_score(_state["classify_correct"] / 5.0)
            _state["task_scores"].append(task1_score)
            _state["current_task"] = 2
            advance_task = True
            feedback += f"\n✅ Task 1 complete! Score: {task1_score:.3f}. Moving to Task 2."

    # ------------------------------------------------------------------
    # Task 2: Prioritize emails
    # ------------------------------------------------------------------
    elif task == 2:
        if action.action_type != "prioritize":
            return _error_response("For Task 2 use action_type='prioritize'")

        score, feedback = grade_prioritize(action.priority_order)
        clamped_score = _clamp_task_score(score)
        reward = compute_step_reward(2, score, _state["steps_taken"])
        _state["total_reward"] += reward
        _state["task_scores"].append(clamped_score)
        _state["current_task"] = 3
        advance_task = True
        feedback += f"\n✅ Task 2 complete! Score: {clamped_score:.3f}. Moving to Task 3."

    # ------------------------------------------------------------------
    # Task 3: Draft a reply
    # ------------------------------------------------------------------
    elif task == 3:
        if action.action_type != "reply":
            return _error_response("For Task 3 use action_type='reply'")
        if action.email_id != "e2":
            return _error_response("For Task 3 reply to email_id='e2'")

        score, feedback = grade_reply(action.reply_text)
        clamped_score = _clamp_task_score(score)
        reward = compute_step_reward(3, score, _state["steps_taken"])
        _state["total_reward"] += reward
        _state["task_scores"].append(clamped_score)
        _state["done"] = True
        feedback += f"\n🏁 All tasks complete! Final scores: {_state['task_scores']}. Total reward: {_state['total_reward']:.3f}"

    done = _state["done"]
    obs = _get_observation(
        last_result=f"[Task {task}] {feedback}",
        score=reward,
        done=done,
        feedback=feedback,
    )

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": {
            "task": task,
            "steps_taken": _state["steps_taken"],
            "total_reward": _state["total_reward"],
            "task_scores": _state["task_scores"],
        },
    }


@app.get("/state")
def state():
    return {
        "current_task": _state["current_task"],
        "steps_taken": _state["steps_taken"],
        "total_reward": _state["total_reward"],
        "task_scores": _state["task_scores"],
        "done": _state["done"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(message: str):
    obs = _get_observation(last_result=f"❌ Error: {message}", score=0.0)
    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {"error": message},
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()