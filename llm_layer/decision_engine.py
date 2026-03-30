from llm_layer.tool_executor import execute_tool


class DecisionEngine:

    def __init__(self):
        self.history = []

    def run(self, decision, context):

        action = decision["action"]

        # 🚨 SAFETY GUARDS (ADD HERE)
        if action == "encode" and not context["categorical"]:
            print("⚠️ Skipping encode (no categorical data)")
            return context

        if action == "handle_missing" and not context["missing"]:
            print("⚠️ Skipping missing handling (no missing values)")
            return context

        if action == "scale" and context["scaled"]:
            print("⚠️ Skipping scaling (already scaled)")
            return context

        if action == "train_model" and context["trained"]:
            print("⚠️ Skipping training (already trained)")
            return context

        if action == "evaluate" and context["evaluated"]:
            print("⚠️ Skipping evaluation (already done)")
            return context

        # -------------------------------
        # EXECUTE TOOL
        # -------------------------------
        result = execute_tool(action, context)

        # -------------------------------
        # TRACK STEP
        # -------------------------------
        context.setdefault("steps_done", []).append(action)

        # -------------------------------
        # UPDATE STATE
        # -------------------------------
        if action == "handle_missing":
            context["missing"] = False

        elif action == "encode":
            context["categorical"] = False

        elif action == "scale":
            context["scaled"] = True

        elif action == "train_model":
            context["trained"] = True

        elif action == "evaluate":
            context["evaluated"] = True

        return result