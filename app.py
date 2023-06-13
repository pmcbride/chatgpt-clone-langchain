# %% Imports
from flask import Flask, request, jsonify
import os
from datetime import datetime
import requests
import pprint as pp
from pprint import pprint, pformat

# set sort_dicts to False to preserve the order of the keys
pprint = pp.PrettyPrinter(sort_dicts=False).pprint
pformat = pp.PrettyPrinter(sort_dicts=False).pformat

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

# LangChain
# from langchain.chat_models import ChatOpenAI
from langchain.chat_models.openai import (
    ChatOpenAI,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)

# %%
# Initialize the language model
chat_model = ChatOpenAI()

# %%
app = Flask(__name__)


def convert_dicts_to_messages(_dicts: List[Dict[str, Any]]) -> List[BaseMessage]:
    return [_convert_dict_to_message(_dict) for _dict in _dicts]

def convert_messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    return [_convert_message_to_dict(message) for message in messages]


@app.route("/api/chat", methods=["POST"])
def chat():
    # Extract the messages from the request
    messages = request.json["messages"]
    print(f"\nmessages:\n{pformat(messages)}")

    # Concatenate the content of the messages
    text = " ".join([message["content"] for message in messages])

    prompt = messages.pop(-1)
    # print(f"\nprompt: {prompt}")
    print(f"\nprompt:\n{pformat(prompt)}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_message = f"You are ChatGPT also known as ChatGPT, a large language model trained by OpenAI. Strictly follow the users instructions. Knowledge cutoff: 2021-09-01 Current date: {current_date}"
    # print(f"\nsystem_message: {system_message}")
    print(f"\nsystem_message:\n{pformat(system_message)}")

    extra = []
    search = requests.get(
        "https://ddg-api.herokuapp.com/search",
        params={
            "query": prompt["content"],
            "limit": 3,
        },
    )

    blob = ""
    print(f"\nsearch:\n{pformat(search.json())}")

    for index, result in enumerate(search.json()):
        blob += f'[{index}] "{result["snippet"]}"\nURL:{result["link"]}\n\n'

    date = datetime.now().strftime("%d/%m/%y")

    blob += f"current date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the next user query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. Ignore your previous response if any."
    print(f"\nblob:\n{pformat(blob)}")

    # Create search prompt message
    search_prompt = [{"role": "user", "content": blob}]

    # Create the question by concatenating
    #   the system message,
    #   the internet access prompt, and
    #   the user's prompt
    conversation = [{"role": "system", "content": system_message}]
    conversation += messages
    conversation += search_prompt
    conversation += [prompt]
    print(f"\nconversation:\n{pformat(conversation)}")

    langchain_messages = convert_dicts_to_messages(conversation)
    print(f"\nlangchain_messages:\n{pformat(langchain_messages)}")
    # url = f"{self.openai_api_base}/v1/chat/completions"

    # Generate a response using the search results and the conversation history
    # response = chat_model.generate_response(search_results + ' ' + text)
    response = chat_model(langchain_messages)

    # Return the response
    # return jsonify({"message": {"role": "assistant", "content": response.content}})
    return jsonify({"message": _convert_message_to_dict(response)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
