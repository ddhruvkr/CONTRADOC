import openai # set api keys if needed
def get_responses_from_gpt(prompts, model="gpt-4", temp=0):
    """
    Input: prompts: List[String]
    Output: full_preds: List[String]
    """
    api_key = "your-api-key-here"  # Replace with your actual API key
    preds = []

    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        preds.append(response["choices"][0]["message"]["content"])

    return preds