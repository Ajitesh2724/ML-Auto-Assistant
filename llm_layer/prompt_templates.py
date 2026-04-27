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


def get_explanation_prompt(results):

    return f"""
You are an expert Machine Learning assistant.

Analyze the following:

Task: {results['task_type']}
Model: {results['model']}
Metrics: {results['metrics']}
Top Features: {results['features'][:5]}

IMPORTANT INSTRUCTIONS:
- Answer ALL sections fully
- Do NOT stop mid-sentence
- Do NOT change topic
- Keep answers brief.
- Consider your number of tokens to complete the answer with full-stop.

Format your response EXACTLY like this:

1. Why this model was selected:
<answer>

2. Is performance good or bad:
<answer>

3. Overfitting/underfitting signs:
<answer>

4. Business interpretation:
<answer>
"""


def get_chat_prompt(results, user_question):

    return f"""
You are an ML assistant answering a user's question.

IMPORTANT RULES:
- ONLY answer the question
- DO NOT repeat full analysis
- DO NOT regenerate overview
- Keep answer concise and relevant
- Stay on topic

Context (use only if needed):
Model: {results['model']}
Metrics: {results['metrics']}

User Question:
{user_question}

Answer:
"""