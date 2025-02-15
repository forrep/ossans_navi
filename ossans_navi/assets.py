import datetime
import json
import re
import time
from typing import Any

from google.genai.types import Schema, Type

from ossans_navi import config
from ossans_navi.type.slack_type import SlackChannel


def get_now() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def dedent(text, indent: int = 8) -> str:
    return re.sub(f"^ {{{indent}}}", '', text, 0, re.MULTILINE)


ImageDescriptionItemSchema = Schema(
    type=Type.ARRAY,
    items=Schema(
        type=Type.OBJECT,
        properties={
            "text": Schema(type=Type.STRING),
            "description": Schema(type=Type.STRING),
        },
        required=["text", "description"],
        property_ordering=["text", "description"],
    ),
)


def get_image_description_system_prompt(channel: SlackChannel, settings: str):
    return dedent(
        f"""
        # Now
        {get_now()}

        # Precondition
        - You are an excellent assistant named {" or ".join([f'"{name}"' for name in config.ASSISTANT_NAMES])} that works as a bot on slack used by the "{config.WORKSPACE_NAME}"!
        - This message is exchanged on the "Slack channel" described below.
        {settings + ("\n" if settings else "")}
        # Slack channel
        name: {channel.name}
        topic: {channel.topic}
        purpose: {channel.purpose}

        # What I need you to do
        - Output a description and text for each of the images included in each message.
        - Outputs an array of image descriptions and text using the permalink of the message as a key
        - After considering the intent of the message, retrieve the necessary information from the image and output a summary in the "description" field.
        - Output the text information of the image to "text".
        - Output your response in JSON format according to the "Output format" described below.
        - Please think in {config.LANGUAGE} and respond in {config.LANGUAGE}.

        # Output format
        {{
            message1_permalink: [
                {{
                    "text": message1_image1_text,
                    "description": message1_image1_description
                }},
                {{
                    "text": message1_image2_text,
                    "description": message1_image2_description
                }},
                ...
            ],
            message2_permalink: [
                {{
                    "text": message2_image1_text,
                    "description": message2_image1_description
                }},
                {{
                    "text": message2_image2_text,
                    "description": message2_image2_description
                }},
                ...
            ],
            ...
        }}
        """
    ).strip()


ClassificationSchema = Schema(
    type=Type.OBJECT,
    properties={
        "response_text": Schema(type=Type.STRING),
        "message_type": Schema(type=Type.STRING),
        "slack_emoji_name": Schema(type=Type.STRING),
    },
    required=["response_text", "message_type", "slack_emoji_name"],
    property_ordering=["response_text", "message_type", "slack_emoji_name"],
)


def get_classification_system_prompt(channel: SlackChannel, settings: str):
    return dedent(
        f"""
        # Now
        {get_now()}

        # Precondition
        - You are an excellent assistant named {" or ".join([f'"{name}"' for name in config.ASSISTANT_NAMES])} that works as a bot on slack used by the "{config.WORKSPACE_NAME}"!
        - This message is exchanged on the "Slack channel" described below.
        {settings + ("\n" if settings else "")}
        # Slack channel
        name: {channel.name}
        topic: {channel.topic}
        purpose: {channel.purpose}

        # What I need you to do
        - Please output your response in JSON format according to the "Output format" described below.

        # Output format
        {{
        "response_text": contents,
        "message_type": contents,
        "slack_emoji_name": emoji_name
        }}

        # Rules for "response_text"
        - Output response to user.

        # Rules for "message_type"
        - Consider the intent of the last message sent by the user. Then output the message intent by selecting from the options below. Be sure to select from only the following options.
            - question
            - request
            - report
            - advice
            - agreement
            - empathy
            - confirmation
            - admiration
            - disappointment
            - task_list
            - other

        # Rules for "slack_emoji_name"
        - Output the name of the appropriate slack emoji to react to.
        - Anything other than :+1: or :thumbsup: would be great!
        """
    ).strip()


SlackSearchWordSchema = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING),
        "date_range": Schema(type=Type.STRING),
        "how_to_find_out": Schema(type=Type.STRING),
        "slack_search_words": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=["user_intent", "date_range", "how_to_find_out", "slack_search_words"],
    property_ordering=["user_intent", "date_range", "how_to_find_out", "slack_search_words"],
)


def get_slack_search_word_system_prompt(channel: SlackChannel, settings: str):
    return dedent(
        f"""
        # Now
        {get_now()}

        # Precondition
        - You are an excellent assistant named {" or ".join([f'"{name}"' for name in config.ASSISTANT_NAMES])} that works as a bot on slack used by the "{config.WORKSPACE_NAME}"!
        - This message is exchanged on the "Slack channel" described below.
        {settings + ("\n" if settings else "")}
        # Slack channel
        name: {channel.name}
        topic: {channel.topic}
        purpose: {channel.purpose}

        # What I need you to do
        - On my behalf, I would like you to come up with search keywords for full-text search in Slack.
        - Please output your response in JSON format according to the "Output format" described below.
        - Please think in {config.LANGUAGE} and respond in {config.LANGUAGE}.
        - Slack is used by all employees for work, chatting, etc., so you can retrieve information that the "{config.WORKSPACE_NAME}" has when you search for it.

        # Output format
        {{
        "user_intent": contents,
        "date_range": contents,
        "how_to_find_out": contents,
        "slack_search_words": [search_word_1, search_word_2, ... ]
        }}

        # Rules for "user_intent"
        - Organize and output the questions and intentions contained in the last message.
        - If the last message does not contain a question or intent, null is output.

        # Rules for "date_range"
        - If "user_intent" is limited to a date, please output that content.

        # Rules for "how_to_find_out"
        - If "user_intent" is null, output null.
        - What and how should I look into to resolve "user_intent"? Please organize and output.

        # Rules for "slack_search_words"
        - If "user_intent" is null, output an empty array in "slack_search_words".
        - If "user_intent" exists, think of many search words that might get an answer and output them in an array, I search once for each element of the array.
        - Separate search words with a space for AND search.
        - You can also add a date filter to the search words.
        - If "after:yyyy-mm-dd" is specified as a search word, it searches after that date.
        - If "before:yyyy-mm-dd" is specified as a search word, it searches before that date.
        - Specify "user_id" in the form "from:<@UXXXXXXXX>" to search messages by sender.
        - "in:#channel_name" to search within a channel.
        - We would like to search for words with similar meanings, as information may be recorded with different wording. Please provide as many search terms as you can.
        - If "user_intent" is time-related, please use the date filter, do not include relative dates and times in the string, such as "去年", "今年", or "最近".
        - To obtain a wide range of search results, please include search terms with only one word, not just words that use AND search.
        """
    ).strip()


def get_information_obtained_by_rag_prompt(info: list[dict[str, Any]], words: list[str] | None = None) -> str:
    return (
        "# Information obtained at RAG (JSON format)\n"
        + (
            (
                "```\n"
                + json.dumps(info, ensure_ascii=False, separators=(',', ':')) + "\n"
                + "```\n"
            ) if len(info) > 0 else (
                "No relevant information was found. Please respond appropriately.\n"
            )
        )
        + (
            (
                "\n"
                + "## Search words\n"
                + "- " + "\n- ".join(words)
            ) if isinstance(words, list) and len(words) > 0 else ""
        )
    )


RefineSlackSearchesSchema = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING),
        "permalinks": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "get_next_messages": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "get_messages": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "additional_search_words": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=["user_intent", "permalinks", "get_next_messages", "get_messages", "additional_search_words"],
    property_ordering=["user_intent", "permalinks", "get_next_messages", "get_messages", "additional_search_words"],
)


def get_refine_slack_searches_system_prompt(channel: SlackChannel, settings: str):
    return dedent(
        f"""
        # Now
        {get_now()}

        # Precondition
        - You are an excellent assistant named {" or ".join([f'"{name}"' for name in config.ASSISTANT_NAMES])} that works as a bot on slack used by the "{config.WORKSPACE_NAME}"!
        - "Information obtained at RAG" is the messages and threads exchanged in Slack in JSON format. "attachments" and "files" are not posted by the submitter themselves, but are quoted from other information. If the thread's parent message exists, it will be populated in "root_message"
        - User's message is exchanged on the "Slack channel" described below.
        {settings + ("\n" if settings else "")}
        # Slack channel
        name: {channel.name}
        topic: {channel.topic}
        purpose: {channel.purpose}

        # What I need you to do
        - I want to pick out useful information from "Information obtained at RAG". Please exclude information that is completely irrelevant to the user's question, and output permalinks for the other necessary information.
        - Mixing in information that is not necessary is not a problem, but never omit relevant information. The more permalinks you output, the better!
        - Output the permalink to "get_next_messages" if there is information needed for subsequent messages.
        - If you have a link in the text that may have the information you need, please specify the link in "get_messages".
        - Please output your response in JSON format according to the "Output format" described below.
        - Please think in {config.LANGUAGE} and respond in {config.LANGUAGE}.

        # Output format
        {{
        "user_intent": contents,
        "permalinks": [permalink_1, permalink_2, ... ],
        "get_next_messages": [permalink_1, permalink_2, ... ],
        "get_messages": [link_1, link_2, ... ],
        "additional_search_words": [search_word_1, search_word_2, ... ]
        }}

        # Rules for "user_intent"
        - Please organize and output the intent contained in the last message.
        - If the last message does not contain an intention, null is output.

        # Rules for "permalinks"
        - Output permalink as an array.
        - Super important, I'll say it again. Please do not leave out any permalinks for messages that may be of even the slightest relevance. It is OK to mix in extraneous information.
        - If "user_intent" is null or "Information obtained at RAG" does not contain the required information, an empty array is output.

        # Rules for "get_next_messages"
        - Retrieve messages replied to the specified permalinks and add them to "Information obtained at RAG".

        # Rules for "get_messages"
        - If you think you can find the information you need at a link in the text, please specify the link.

        # Rules for "additional_search_words"
        - If you need to retrieve additional information, output an array of new search words.
        """
    ).strip()


LastshotSchema = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "response_message": Schema(type=Type.STRING),
        "confirm_message": Schema(type=Type.STRING, nullable=True),
        "response_quality": Schema(type=Type.BOOLEAN),
    },
    required=["user_intent", "response_message", "confirm_message", "response_quality"],
    property_ordering=["user_intent", "response_message", "confirm_message", "response_quality"],
)


def get_lastshot_system_prompt(channel: SlackChannel, settings: str):
    return dedent(
        f"""
        # Now
        {get_now()}

        # Precondition
        - You are an excellent assistant named {" or ".join([f'"{name}"' for name in config.ASSISTANT_NAMES])} that works as a bot on slack used by the "{config.WORKSPACE_NAME}"!
        - "Information obtained at RAG" is the messages and threads exchanged in Slack in JSON format. "attachments" and "files" are not posted by the submitter themselves, but are quoted from other information. If the thread's parent message exists, it will be populated in "root_message"
        - This message is exchanged on the "Slack channel" described below.
        {settings + ("\n" if settings else "")}
        # Slack channel
        name: {channel.name}
        topic: {channel.topic}
        purpose: {channel.purpose}

        # What I need you to do
        - Respond to the user's questions or intentions.
        - Please refer to "Information obtained at RAG" and give priority to the {config.WORKSPACE_NAME}'s circumstances and internal rules in your answer.
        - The "Information obtained at RAG" contains outdated information; use the newer information.
        - The "Information obtained at RAG" includes information you yourself submitted, which may be incorrect.
        - If "Information obtained at RAG" does not provide valid information, please respond in general terms.
        - If you know someone who is knowledgeable about the question, please create a message to encourage that person to answer the question.
        - Please output your response in JSON format according to the "Output format" described below.
        - Mention "user_id" in the form of "<@UXXXXXXXXX>".
        - When you make a Mention, the other person will be notified, so please keep usage to a minimum!
        - Use a name, not an ID, to refer to a specific person.
        - Please think in {config.LANGUAGE} and respond in {config.LANGUAGE}.

        # Output format
        {{
        "user_intent": contents,
        "response_message": contents,
        "confirm_message": contents,
        "response_quality": contents
        }}

        # Rules for "user_intent"
        - Please organize and output the intent contained in the last message.
        - If the last message does not contain an intention, null is output.

        # Rules for "response_message"
        - You cannot do the actual action to the user, if you need action, ask someone else to do it!b
        - Used to reply to users, please output in markdown format.
        - If "user_intent" is null, then appropriate response is given.
        - When answering a "user_intent" from a "Information obtained at RAG", include a link to the referring permalink.

        # Rules for "confirm_message"
        - If "user_intent" is null, output null.
        - Outputs a message in markdown format to be confirmed by someone familiar with "user_intent".
        - Output following "response_message" as a reply to the user, do not repeat the same content.
        - Outputs null if no detailed person is known.

        # Rules for "response_quality"
        - Outputs true if both "user_intent exists" AND "Can be resolved with response_message" are satisfied.
        - Outputs false if other than above.
        """
    ).strip()
