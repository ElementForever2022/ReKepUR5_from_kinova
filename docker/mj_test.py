import time

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

def rate_limited_api_call(func, *args, **kwargs):
	max_retries = 5
	for i in range(max_retries):
		try:
			return func(*args, **kwargs)
		except OpenAI.RateLimitError:
			if i < max_retries - 1:
				time.sleep(2 ** i)  # Exponential backoff
			else:
				raise

# Use it like this:
response = rate_limited_api_call(chat_completion, model="gpt-3.5-turbo", messages=[...])