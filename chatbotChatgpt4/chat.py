import openai
# Indica el API Key
openai.api_key = "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"
# Uso de ChapGPT en Python
model_engine = "text-davinci-003"
prompt = "la suma de 5 mas 10"
completion = openai.Completion.create(engine=model_engine,
                                    prompt=prompt,
                                    max_tokens=1024,
                                    n=1,
                                    stop=None,
                                    temperature=0.7)
respuesta=""
for choice in completion.choices:
    respuesta=respuesta+choice.text
    print(f"Response: %s" % choice.text)