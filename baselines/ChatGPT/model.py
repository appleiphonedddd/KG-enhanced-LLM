class ChatGPTModel:
    def __init__(self, model='gpt-3.5-turbo', api_key=None):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt, max_tokens=128):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def generate_batch(self, prompts, max_tokens=128):
        return [self.generate(p, max_tokens) for p in prompts]
