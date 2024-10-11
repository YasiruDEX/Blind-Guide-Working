from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://192.168.137.1:1234/v1", api_key="lm-studio")

system_prompt = """
You are an election-related task classifier. For each prompt entered, you must categorize it into one of the following categories and return only the category name:

1. Semantic Analysis: If the input is related to understanding the meaning, sentiment, or tone of text (e.g., speeches, articles, or public statements).
2. Win Predictor: If the input is related to predicting election outcomes based on polling data, trends, or performance metrics.
3. Manifesto Comparator: If the input involves comparing political manifestos, policies, or agendas of different candidates or parties.
4. Chat Bot: If the input involves general conversation, Q&A, or interactive dialogue without a specific analysis or comparison task.

You must return only the category name (e.g., "Win Predictor") and nothing else.
"""


# Input prompt from user
user_input = "Compare the political manifestos of the two candidates and identify the key differences."

# Make the API request
completion = client.chat.completions.create(
    model="model-identifier",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ],
    temperature=0.7,
)

# Print the classification result
print(completion.choices[0].message.content)
