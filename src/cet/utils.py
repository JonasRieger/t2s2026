import json
import openai

secrets = json.load(open("secrets.json", "r"))
API_KEY_OPENAI = secrets.get("open_api_key", None)


def get_topic_cleartext(
    topic_terms: list[str], model="gpt-5.1", api_key_openai=API_KEY_OPENAI
) -> str:
    """Generates english topic descriptions based on a list of strings

    Args:
        topic_terms (list[str]): List of topic terms in spanish for the topic
        model (str, optional): OpenAi Model. Defaults to "gpt-5.1".

    Returns:
        str: Generated english topic description
    """

    openai.api_key = API_KEY_OPENAI
    instr = f"Provide a short and clear english topic description based on the following spanish topic terms."
    response = openai.responses.create(
        model=model,
        instructions=instr,
        input=", ".join(topic_terms),
    )
    return response.output_text
