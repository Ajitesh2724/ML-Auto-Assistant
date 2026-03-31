# =========================================================
# 🧠 LLM PROMPT TEMPLATES
# =========================================================


def get_decision_prompt(context):

    return f"""
You are an ML pipeline planner.

Pipeline state:

missing={context["missing"]}
categorical={context["categorical"]}
scaled={context["scaled"]}
trained={context["trained"]}

Steps done:
{context["steps_done"]}

Choose ONE action from:

analyze
handle_missing
encode
scale
feature_select
train_model
evaluate
stop

Rules:

- Do not repeat steps unnecessarily
- Follow correct ML order
- Return only JSON

Format:
{{"action":"<step>"}}
"""


# ---------------------------------------------------------
# ✅ EXPLANATION
# ---------------------------------------------------------
def get_explanation_prompt(results):

    return f"""
You are an expert Machine Learning assistant.

Analyze:

Task: {results['task_type']}
Model: {results['model']}
Metrics: {results['metrics']}
Top Features: {results['features'][:5]}

Answer:

1. Why this model was selected
2. Is performance good or bad
3. Overfitting/underfitting signs
4. 3 improvements
5. Business interpretation

Keep it simple and clear.
"""


# ---------------------------------------------------------
# ✅ CHAT
# ---------------------------------------------------------
def get_chat_prompt(results, user_question):

    return f"""
You are an ML assistant.

Context:
Task: {results['task_type']}
Model: {results['model']}
Metrics: {results['metrics']}
Features: {results['features']}

User Question:
{user_question}

Answer clearly and practically.
"""