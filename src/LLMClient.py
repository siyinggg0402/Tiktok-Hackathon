import os
from openai import OpenAI
import logging

class LLMClient: 
    def __init__(self, model: str = "gpt-4o"):
        """
        initialises the LLM client using openai API.
        Please set "OPENAI_API_KEY" in your env variable, DO NOT insert it here.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set properly")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call_LLM(self, prompt) -> dict:
        try:
            print("Calling LLM...") 
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=prompt
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None