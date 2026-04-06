# 📬 Email Triage Environment

A real-world OpenEnv environment where an AI agent triages a customer support inbox.
The agent classifies, prioritizes, and drafts replies for emails — tasks that real support teams do every day.

---

## 🧠 Environment Description

The agent receives a simulated inbox of 5 customer emails and must complete 3 progressive tasks:

| Task | Difficulty | Description |
|------|-----------|-------------|
| 1. Classification | Easy | Label each email: `billing`, `support`, `spam`, or `inquiry` |
| 2. Priority Ordering | Medium | Rank all 5 emails from most to least urgent |
| 3. Draft Reply | Hard | Write a professional reply to a billing dispute |

Each task has a programmatic grader that scores 0.0–1.0 with partial credit signals.

---

## 📐 Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | str | `"classify"` \| `"prioritize"` \| `"reply"` |
| `email_id` | str | Target email ID (`e1`–`e5`) |
| `category` | str | For classify: `billing`, `support`, `spam`, `inquiry` |
| `priority_order` | list[str] | For prioritize: ordered list of email IDs |
| `reply_text` | str | For reply: the full draft reply text |

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | Current task (1, 2, or 3) |
| `task_description` | str | Instructions for the current task |
| `emails` | list[dict] | Full inbox (id, from, subject, body, received_at) |
| `last_action_result` | str | Feedback from last action |
| `score` | float | Reward earned this step |
| `done` | bool | Whether episode is complete |
| `feedback` | str | Human-readable grader feedback |

---

## 🏆 Grading Logic

- **Task 1 (Classify):** Exact match per email. Score = correct / 5. Each correct classification also earns step reward.
- **Task 2 (Prioritize):** Kendall-tau distance. Measures how many pairs are in the correct relative order. Score = concordant_pairs / total_pairs.
- **Task 3 (Reply):** 5-criterion rubric (apologize, acknowledge amount, promise refund, give timeline, professional close). Each criterion = 0.2 points.
- **Efficiency bonus:** A small penalty (0.02/step) for excessive steps incentivizes directness.

**Correct answers (for reference):**
- Classifications: e1→support, e2→billing, e3→spam, e4→support, e5→inquiry
- Priority order: e1 > e5 > e2 > e4 > e3
- Reply rubric: must contain apologize + acknowledge $200 overcharge + promise refund + timeline + professional close

---

## 🚀 Setup & Usage

### Local development

```bash
# Clone / navigate to this folder
cd email-triage-env

# Install server dependencies
pip install -r server/requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

### Validate

```bash
openenv validate --url http://localhost:8000
```

### Run inference baseline

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:8000"

pip install -r requirements.txt
python inference.py
```

---

## 📊 Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| Task 1 - Classify | openai/gpt-4o-mini | 1.000 |
| Task 2 - Prioritize | openai/gpt-4o-mini | 0.900 |
| Task 3 - Reply | openai/gpt-4o-mini | 1.000 |
| **Average** | | **0.967** |
---

## 📁 Project Structure

```
email_triage_env/
├── openenv.yaml          # OpenEnv manifest
├── Dockerfile            # Container definition
├── inference.py          # Baseline inference script (root, required)
├── requirements.txt      # Inference script dependencies
├── models.py             # Typed Action/Observation/State models
└── server/
    ├── __init__.py
    ├── app.py                      # FastAPI server (step/reset/state/health)
    ├── email_triage_environment.py # Core graders and environment logic
    └── requirements.txt            # Server dependencies
```
