import requests
import json
import re

from llm_layer.prompt_templates import get_decision_prompt


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

                    "temperature": 0,

                    "num_predict": 60

                }

            }

        )

        return response.json()["response"]


    def extract_json(self, text):

        """
        Extract FIRST json object only
        """

        match = re.search(r"\{[^{}]*\}", text)

        if match:

            return match.group()

        return None


    def decide(self, context):

        prompt = get_decision_prompt(context)

        output = self.call_llm(prompt)

        print("\nRAW LLM OUTPUT:")
        print(output)


        json_text = self.extract_json(output)

        if not json_text:

            raise ValueError("LLM did not return JSON")


        return json.loads(json_text)