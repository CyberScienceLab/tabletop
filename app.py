import os
import csv
import json
import datetime
from typing import Optional, Dict, List

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# ==========================
# Setup
# ==========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTIONS_FILE = "questions.csv"
RESULTS_FILE = "results.csv"
SCORE_THRESHOLD = 3.5

# ==========================
# Load questions
# ==========================
df = pd.read_csv(QUESTIONS_FILE)
df["qid"] = df["qid"].astype(int)


def get_question(qid: int):
    row = df[df["qid"] == qid].iloc[0]
    return {
        "qid": int(row.qid),
        "category_id": str(row.category_id),
        "question": str(row.question),
        "gold_answer": str(row.gold_answer),
    }


def get_first_qid():
    return int(df.sort_values("qid").iloc[0].qid)


def get_next_in_category(qid: int):
    row = df[df["qid"] == qid].iloc[0]
    subset = df[df["category_id"] == row.category_id].sort_values("qid")
    ids = list(subset.qid)
    idx = ids.index(qid)
    return int(ids[idx + 1]) if idx + 1 < len(ids) else None


def get_next_category_first_qid(qid: int):
    row = df[df["qid"] == qid].iloc[0]
    cats = list(df["category_id"].unique())
    idx = cats.index(row.category_id)
    if idx + 1 < len(cats):
        next_cat = cats[idx + 1]
        sub = df[df["category_id"] == next_cat].sort_values("qid")
        return int(sub.iloc[0].qid)
    return None


def is_first_in_category(qid: int):
    row = df[df["qid"] == qid].iloc[0]
    first_qid = int(df[df["category_id"] == row.category_id].sort_values("qid").iloc[0].qid)
    return qid == first_qid


# ==========================
# Logging
# ==========================
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "participant_id", "category_id", "qid",
            "question", "gold_answer", "user_answer", "score", "analysis"
        ])


def log_result(pid, cat, qid, question, gold, user_ans, score, analysis):
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.utcnow().isoformat(),
            pid, cat, qid, question, gold, user_ans, f"{score:.2f}", analysis
        ])


# ==========================
# LLM Scoring
# ==========================
def grade_answer(question, gold_answer, user_answer):
    prompt = f"""
Grade this answer.

QUESTION:
{question}

IDEAL ANSWER:
{gold_answer}

USER ANSWER:
{user_answer}

Return only JSON:
{{
 "score": number,
 "analysis": "text"
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        return float(data["score"]), data["analysis"]
    except:
        return 0.0, raw


# ==========================
# State
# ==========================
def init_state():
    return {
        "participant_id": "",
        "qid": None,
        "done": False
    }


# ==========================
# Chat Logic
# ==========================
def chat_fn(message, history, state):

    state = state or init_state()
    history = history or []

    # Add user msg
    history.append({"role": "user", "content": message})

    # ID command
    if message.startswith("/id "):
        pid = message.replace("/id ", "").strip()
        state["participant_id"] = pid
        bot_msg = f"ID set to **{pid}** 🎉\n\nType **start** to begin."
        history.append({"role": "assistant", "content": bot_msg})
        return history, state, ""

    # Require ID
    if not state["participant_id"]:
        bot_msg = "👋 Please enter your participant ID using `/id 11`."
        history.append({"role": "assistant", "content": bot_msg})
        return history, state, ""

    # Start
    if message.lower() == "start":
        state["qid"] = get_first_qid()
        q = get_question(state["qid"])
        bot_msg = (
            # f"🚀 **Test Started**\n\n"
            f"Category: {q['category_id']}\n" #— Question {q['qid']}
            f"{q['question']}"
        )
        history.append({"role": "assistant", "content": bot_msg})
        return history, state, ""

    # No active question
    if state["qid"] is None:
        history.append({"role": "assistant", "content": "No active question. Type **start**."})
        return history, state, ""

    # Evaluate answer
    qid = state["qid"]
    q = get_question(qid)

    score, analysis = grade_answer(q["question"], q["gold_answer"], message)

    log_result(
        state["participant_id"], q["category_id"], qid,
        q["question"], q["gold_answer"], message, score, analysis
    )

    # feedback = f"**Score:** {score:.1f}/5\n\n{analysis}"

    # Determine next
    extra = ""
    next_qid = None

    if is_first_in_category(qid) and score < SCORE_THRESHOLD:
        extra = "\n\n⚠️ Low score — skipping category."
        next_qid = get_next_category_first_qid(qid)
    else:
        next_qid = get_next_in_category(qid)
        if next_qid is None:
            extra = "\n\n Category complete."
            next_qid = get_next_category_first_qid(qid)

    if next_qid is None:
        bot_msg = extra + "\n\n🏁 **Test complete. Thank you!**"  #feedback +  
        state["qid"] = None
    else:
        state["qid"] = next_qid
        q_next = get_question(next_qid)
        bot_msg = (
            # feedback + 
            extra +
            f"\n\nCategory: {q_next['category_id']} \n"  #/ Q{q_next['qid']}
            f"{q_next['question']}"
        )

    history.append({"role": "assistant", "content": bot_msg})
    return history, state, ""   # Clear textbox


# ==========================
# UI (HTML-based styling)
# ==========================
with gr.Blocks() as demo:

    gr.HTML("""
    <div style='text-align:center; margin-top:20px;'>
        <h1 style='font-size:32px; margin-bottom:5px;'>Ransomware Knowledge & Awareness Checker</h1>
        <p style='font-size:18px; color:#555;'>Please enter your ID using <b>/id </b> to begin.</p>
        <hr style='margin-top:20px;'>
    </div>
    """)

    
    chatbot = gr.Chatbot(label="Conversation")  # dict messages
    state = gr.State(init_state())

    msg = gr.Textbox(
        label="Your message",
        placeholder="Type your answer here...",
        autofocus=True
    )

    msg.submit(chat_fn, [msg, chatbot, state], [chatbot, state, msg])

demo.launch()
