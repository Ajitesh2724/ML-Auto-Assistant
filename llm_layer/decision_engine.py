from llm_layer.tool_executor import execute_tool


class DecisionEngine:

    def __init__(self):

        self.history = []


    def run(self, decision, context):

        action = decision["action"]

        result = execute_tool(action, context)

        context.setdefault("steps_done", []).append(action)

        return result