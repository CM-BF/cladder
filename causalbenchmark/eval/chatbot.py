import os

import openai
from openai import OpenAI, completions
import backoff
import json


class Chatbot(object):
    def __init__(self, model_version="gpt-3.5-turbo", system_prompt="You are a helpful assistant.",
                 max_tokens=100):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], organization=os.environ['OPENAI_ORG_ID'])
        self._history = [{"role": "user" if "o1" in model_version else "system", "content": system_prompt}]  # To track the history of the conversation
        self.model_version = model_version
        self._max_tokens = max_tokens

    def ask(self, prompt, max_tokens=None, **kwargs):
        # Adding the user prompt to the conversation history
        self._history.append({"role": "user", "content": prompt})

        # Creating a prompt with conversation history
        messages = self._history.copy()

        @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=6)
        def send_with_backoff():
            return self.client.chat.completions.create(
                model=self.model_version,  # Specify the model version
                messages=messages,
                max_completion_tokens=max_tokens or self._max_tokens,
            )

        completion = send_with_backoff()

        # Extracting the response content and adding to history
        reply = completion.choices[0].message.content.strip()
        self._history.append({"role": "assistant", "content": reply})

        return reply

    @property
    def history(self):
        return self._history

    def clean_history(self):
        self._history = self._history[:1]  # Resetting the history to the initial system prompt

    # def construct_batch_file(self, queries, output_file):
    #     assert output_file.endswith('.jsonl'), "Output file must be a JSONL file"
    #     with open(output_file, 'w') as f:
    #         for q_i, query in enumerate(queries):
    #             item = {
    #                 "custom_id": f'request-{q_i}',
    #                 "method": "POST",
    #                 "url": "/v1/chat/completions",
    #                 "body": {
    #                     "model": self.model_version,
    #                     "messages": data["messages"],
    #                     "max_tokens": data["max_tokens"]
    #                 }
    #             }
