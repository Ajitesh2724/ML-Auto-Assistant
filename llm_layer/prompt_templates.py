def get_decision_prompt(context):

    return f"""

Pipeline state:

missing={context["missing"]}
categorical={context["categorical"]}
scaled={context["scaled"]}
trained={context["trained"]}

Steps done:
{context["steps_done"]}

Choose ONE action:

analyze
handle_missing
encode
scale
feature_select
train_model
evaluate
stop

Return JSON only.

Example:
{{"action":"encode"}}

"""