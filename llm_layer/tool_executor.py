from llm_layer.tool_registry import TOOLS


def execute_tool(action, context, **kwargs):

    if action == "stop":
        print("Pipeline finished")
        return {"stop": True}

    if action not in TOOLS:
        raise ValueError(f"Unknown action: {action}")

    print("Running:", action)

    updated_context = TOOLS[action](context, **kwargs)

    return updated_context