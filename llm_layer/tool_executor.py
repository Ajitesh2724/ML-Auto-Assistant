from llm_layer.tool_registry import TOOLS


def execute_tool(action, context, **kwargs):

    if action == "stop":

        print("Pipeline finished")
        return {"stop": True}

    print("Running:", action)

    result = TOOLS[action](context, **kwargs)

    return result