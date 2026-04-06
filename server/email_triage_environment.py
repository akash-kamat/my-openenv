"""
Email Triage Environment - Core Logic
Real-world task: An AI agent triages a customer support inbox.
Tasks range from easy (classify) → medium (prioritize) → hard (draft reply).
"""

import re
from typing import List, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Static email dataset (deterministic for reproducibility)
# ---------------------------------------------------------------------------

EMAILS = [
    {
        "id": "e1",
        "from": "alice@company.com",
        "subject": "Urgent: Server down, production blocked",
        "body": "Our entire production environment is down. Customers can't access the app. We need immediate help. This is costing us thousands per minute.",
        "received_at": "2024-01-15 09:02",
    },
    {
        "id": "e2",
        "from": "billing@customer.com",
        "subject": "Invoice #4521 dispute",
        "body": "I was charged $299 but my plan is $99/month. Please refund the difference. I've been a customer for 3 years.",
        "received_at": "2024-01-15 08:45",
    },
    {
        "id": "e3",
        "from": "random@spam.io",
        "subject": "You've won a prize!!!",
        "body": "Congratulations! You've been selected to win a free iPhone. Click here now to claim: http://totally-legit-prize.xyz",
        "received_at": "2024-01-15 09:10",
    },
    {
        "id": "e4",
        "from": "newuser@gmail.com",
        "subject": "How do I export my data?",
        "body": "Hi, I've been using your product for a week and I'm curious how I can export all my data to CSV. Is there a way to do this?",
        "received_at": "2024-01-15 08:30",
    },
    {
        "id": "e5",
        "from": "enterprise@bigcorp.com",
        "subject": "Pricing for 500-seat enterprise license",
        "body": "We are evaluating your product for our organization of 500 employees. Could you send pricing and availability for enterprise licensing? We need to make a decision by end of quarter.",
        "received_at": "2024-01-15 09:00",
    },
]

# ---------------------------------------------------------------------------
# Ground truth answers
# ---------------------------------------------------------------------------

# Task 1: Correct categories for each email
CORRECT_CATEGORIES = {
    "e1": "support",
    "e2": "billing",
    "e3": "spam",
    "e4": "support",
    "e5": "inquiry",
}

# Task 2: Correct urgency ordering (most urgent first)
# e1 (production down) > e5 (enterprise deal) > e2 (billing dispute) > e4 (how-to question) > e3 (spam)
CORRECT_PRIORITY_ORDER = ["e1", "e5", "e2", "e4", "e3"]

# Task 3: Reply to e2 (billing dispute) - rubric criteria
REPLY_RUBRIC = {
    "apologize": ["sorry", "apologize", "apolog", "sincerely"],
    "acknowledge_amount": ["299", "99", "difference", "overcharged", "incorrect charge", "extra"],
    "promise_refund": ["refund", "credit", "reimburse", "return", "correct"],
    "timeline": ["business day", "24 hour", "48 hour", "week", "shortly", "soon", "within"],
    "professional_close": ["regards", "sincerely", "thank you", "best", "team"],
}


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_classify(email_id: str, predicted_category: str) -> Tuple[float, str]:
    """Task 1 grader: exact match on category."""
    correct = CORRECT_CATEGORIES.get(email_id)
    if not correct:
        return 0.0, f"Unknown email_id: {email_id}"
    
    predicted = predicted_category.lower().strip()
    if predicted == correct:
        return 1.0, f"Correct! '{email_id}' is '{correct}'."
    else:
        return 0.0, f"Incorrect. '{email_id}' should be '{correct}', got '{predicted}'."


def grade_prioritize(predicted_order: List[str]) -> Tuple[float, str]:
    """
    Task 2 grader: Kendall tau distance (normalized).
    Measures how close the predicted ranking is to the correct one.
    Score 1.0 = perfect, 0.0 = completely reversed.
    """
    correct = CORRECT_PRIORITY_ORDER
    n = len(correct)

    if len(predicted_order) != n:
        return 0.0, f"Expected {n} email IDs, got {len(predicted_order)}."
    
    # Check all IDs are valid
    if set(predicted_order) != set(correct):
        return 0.0, f"Email IDs don't match. Got: {predicted_order}"

    # Count concordant pairs
    pos_correct = {eid: i for i, eid in enumerate(correct)}
    pos_pred = {eid: i for i, eid in enumerate(predicted_order)}

    concordant = 0
    total_pairs = n * (n - 1) // 2

    ids = list(correct)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            if (pos_correct[a] < pos_correct[b]) == (pos_pred[a] < pos_pred[b]):
                concordant += 1

    score = concordant / total_pairs
    feedback = f"Priority score: {score:.2f} ({concordant}/{total_pairs} pairs correct). Correct order: {correct}"
    return round(score, 3), feedback


def grade_reply(reply_text: str) -> Tuple[float, str]:
    """
    Task 3 grader: rubric-based scoring.
    Each criterion worth 0.2 points (5 criteria = 1.0 max).
    Partial credit is the key differentiator here.
    """
    text_lower = reply_text.lower()
    scores = {}
    feedback_parts = []

    for criterion, keywords in REPLY_RUBRIC.items():
        hit = any(kw in text_lower for kw in keywords)
        scores[criterion] = 1.0 if hit else 0.0
        status = "✓" if hit else "✗"
        feedback_parts.append(f"{status} {criterion}")

    total = sum(scores.values()) / len(scores)
    feedback = f"Reply score: {total:.2f}. Criteria: {', '.join(feedback_parts)}"
    return round(total, 3), feedback


# ---------------------------------------------------------------------------
# Reward shaping helpers
# ---------------------------------------------------------------------------

def compute_step_reward(task_id: int, action_result_score: float, steps_taken: int) -> float:
    """
    Compute reward for a single step with partial signals.
    - Positive reward for correct actions.
    - Small penalty for taking too many steps (encourages efficiency).
    - No reward for completely wrong actions.
    """
    efficiency_penalty = max(0, (steps_taken - 3) * 0.02)
    base_reward = action_result_score - efficiency_penalty
    return round(max(0.0, min(1.0, base_reward)), 3)
