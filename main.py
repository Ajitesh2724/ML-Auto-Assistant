import pandas as pd
import os

from llm_layer.llm_agent import LLMAgent
from llm_layer.decision_engine import DecisionEngine



path = r"C:\Users\abhra\OneDrive\Documents\GitHub\ML-Auto-Assistant\data\sleep_health_dataset.csv"

print("File exists:", os.path.exists(path))

if not os.path.exists(path):
    raise FileNotFoundError("Dataset not found. Check path!")

df = pd.read_csv(path)
print("Dataset Loaded Successfully\n")



print("Available columns:", df.columns.tolist())
target_column = input("Enter target column: ").strip()

if target_column not in df.columns:
    raise ValueError("Invalid target column selected!")



def detect_task_type(y):
    if y.dtype == "object":
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    return "regression"


task_type = detect_task_type(df[target_column])
print("\nDetected task:", task_type)



context = {
    "df": df,
    "target": target_column,
    "task_type": task_type,
    "steps_done": [],
    "missing": df.isnull().sum().sum() > 0,
    "categorical": len(df.select_dtypes(include="object").columns) > 0,
    "scaled": False,
    "trained": False,
    "evaluated": False
}


agent = LLMAgent()
engine = DecisionEngine()


def get_next_rule_action(context):

    if "analyze" not in context["steps_done"]:
        return "analyze"

    if context["missing"]:
        return "handle_missing"

    if context["categorical"]:
        return "encode"

    if not context["scaled"]:
        return "scale"

    if "feature_select" not in context["steps_done"]:
        return "feature_select"

    if not context["trained"]:
        return "train_model"

    if not context["evaluated"]:
        return "evaluate"

    return "stop"


MAX_STEPS = 15

for step in range(MAX_STEPS):

    print("\n====================")
    print("STEP", step)

    # 🔍 DEBUG CONTEXT
    print("\nCurrent Context:")
    for k, v in context.items():
        if k != "df":
            print(f"{k}: {v}")

   
    decision = agent.decide(context)
    llm_action = decision.get("action")

    print("LLM action:", llm_action)

    
    rule_action = get_next_rule_action(context)

    if llm_action != rule_action:
        print(f"⚠️ LLM WRONG → forcing {rule_action}")
        action = rule_action
    else:
        action = llm_action

   
    result = engine.run({"action": action}, context)

 
    if action == "stop":
        print("Pipeline stopped.")
        break

    if context["evaluated"]:
        print("Pipeline completed successfully ✅")
        break


print("\nFinal Steps:", context["steps_done"])