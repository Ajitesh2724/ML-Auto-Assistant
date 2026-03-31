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

STRICT RULES:

- If "analyze" NOT in steps_done → choose analyze
- If missing=True → choose handle_missing
- If categorical=True → choose encode
- If scaled=False → choose scale
- If not trained → choose train_model
- If trained=True and evaluate not done → choose evaluate
- If everything done → choose stop

- DO NOT repeat unnecessary steps
- DO NOT choose analyze more than once

Return JSON only:
{{"action":"<one_action>"}}
"""