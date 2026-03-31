import requests
import json


class LLMAgent:

    def __init__(self, model="phi3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def call_llm(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 120
                }
            }
        )

        return response.json()["response"]

    def extract_json(self, text):
        import re

    # remove code block markers if present
        text = text.replace("```json", "").replace("```", "")

        matches = re.findall(r"\{.*?\}", text, re.DOTALL)

        for m in matches:
            try:
                return m
            except:
                continue

        return None


    def decide(self, context):

        from llm_layer.prompt_templates import get_decision_prompt

        prompt = get_decision_prompt(context)
        output = self.call_llm(prompt)

        print("\nRAW LLM OUTPUT:")
        print(output)

        json_text = self.extract_json(output)

        if not json_text:
            return {"action": "analyze", "reason": "fallback"}

        try:
            return json.loads(json_text)
        except:
            return {"action": "analyze", "reason": "json_error"}