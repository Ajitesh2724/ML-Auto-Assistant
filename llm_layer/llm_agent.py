import requests
import json
import re


class LLMAgent:

    def __init__(self, model="phi3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def call_llm(self, prompt, retry=True):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 250
            }
        }

        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()

            data = response.json()

            if "response" not in data:
                raise ValueError("Invalid response format from LLM")

            output = data["response"].strip()

            # CONTINUATION FIX WITH CONTEXT
            if retry and not output.endswith(('.', '!', '?')):
                continuation_prompt = (
                    "You were answering the following:\n\n"
                    f"{prompt}\n\n"
                    "Your previous response:\n"
                    f"{output}\n\n"
                    "Continue exactly from where you stopped. "
                    "Do not change topic. Do not introduce unrelated information. "
                    "Complete the answer."
                )

                continuation = self.call_llm(continuation_prompt, retry=False)
                output = output + " " + continuation

            return output

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to LLM server. Run: ollama run phi3"
            )

        except requests.exceptions.Timeout:
            if retry:
                short_prompt = prompt[:1000]
                return self.call_llm(short_prompt, retry=False)
            raise RuntimeError("LLM request timed out")

        except Exception as e:
            raise RuntimeError(f"LLM Error: {str(e)}")

    def extract_json(self, text):

        text = text.replace("```json", "").replace("```", "")

        matches = re.findall(r"\{.*?\}", text, re.DOTALL)

        for m in matches:
            try:
                json.loads(m)
                return m
            except:
                continue

        return None

    def decide(self, context):

        from llm_layer.prompt_templates import get_decision_prompt

        prompt = get_decision_prompt(context)

        try:
            output = self.call_llm(prompt)

            print("\nRAW LLM OUTPUT:")
            print(output)

            json_text = self.extract_json(output)

            if not json_text:
                return {
                    "action": "analyze",
                    "reason": "no_json_found",
                    "raw_output": output
                }

            return json.loads(json_text)

        except Exception as e:
            return {
                "action": "analyze",
                "reason": "llm_failure",
                "error": str(e)
            }