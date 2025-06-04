from google.genai.types import Schema, Type

USER_INTENTIONS_TYPES = [
    "monologue",
    "need_answers_to_questions",
    "ask_someone_to_do_something",
    "report_to_someone",
    "advice_to_someone",
    "agree_with_someone",
    "sympathize_with_someone",
    "comfirm_with_someone",
    "praise_someone_or_something",
    "disappointed_in_someone_or_something",
    "sharing_information",
    "note_for_self",
    "impressions",
    "no_intent",
    "other",
]

USER_EMOTIONS = [
    "be_pleased",
    "be_angular",
    "be_troubled",
    "be_sympathize",
    "be_suspicious",
    "no_emotions",
]

WHO_TO_TALK_TO_TYPES = [
    "to_assistant_bot",
    "to_someone_well_informed",
    "to_specific_persons",
    "to_all_of_us",
    "to_noone",
    "cannot_determine",
]

REQUIRED_KNOWLEDGE_TYPES = [
    "common_sense",
    "public_information",
    "technical_knowledge",
    "other_knowledge",
    "information_within_this_slack_group",
    "no_information_required",
]

CLASSIFY_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "user_intentions_type": Schema(
            type=Type.STRING,
            enum=USER_INTENTIONS_TYPES
        ),
        "who_to_talk_to": Schema(
            type=Type.STRING,
            enum=WHO_TO_TALK_TO_TYPES
        ),
        "user_emotions": Schema(
            type=Type.STRING,
            enum=USER_EMOTIONS
        ),
        "required_knowledge_types": Schema(
            type=Type.ARRAY,
            items=Schema(
                type=Type.STRING,
                enum=REQUIRED_KNOWLEDGE_TYPES
            ),
        ),
        "slack_emoji_names": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=[
        "user_intent",
        "user_intentions_type",
        "who_to_talk_to",
        "user_emotions",
        "required_knowledge_types",
        "slack_emoji_names",
    ],
    property_ordering=[
        "user_intent",
        "user_intentions_type",
        "who_to_talk_to",
        "user_emotions",
        "required_knowledge_types",
        "slack_emoji_names",
    ],
)

CLASSIFY_PROMPT = """
# Now
{{ now }}

# Precondition
- You are an excellent assistant bot named {% for assistant_name in assistant_names %}"{{ assistant_name }}"{% if not loop.last %} or {% endif %}{% endfor %} that works as a bot on slack used by the "{{ workspace_name }}"!
- This message is exchanged on the "Slack channel" described below.
{% if event.settings %}{{ event.settings }}
{% endif %}
# Slack channel
name: {{ event.channel.name }}
topic: {{ event.channel.topic }}
purpose: {{ event.channel.purpose }}

# What I need you to do
- Please output your response in JSON format according to the "Output format" described below.

# Output format
{
"user_intent": contents,
"user_intentions_type": enum,
"who_to_talk_to": enum,
"user_emotions": enum,
"required_knowledge_types": [enum, ...],
"slack_emoji_names": [emoji_name, ...]
}

# Rules for "user_intent"
- Organize and output the questions and intentions contained in the last message.
- If the last message does not contain a question or intent, null is output.

# Rules for "user_intentions_type"
- Consider the intent of the last message sent by the user. Then, select the message intent from the options below. Be sure to select from only the following options.
    - {% for v in schema.user_intentions_type %}{{ v }}{% if not loop.last %}, {% endif %}{% endfor %}

# Rules for "who_to_talk_to"
- Consider the intent of the last message sent by the user. Then, select from the options below to output who the user is talking to. Be sure to select from only the following options.
    - {% for v in schema.who_to_talk_to %}{{ v }}{% if not loop.last %}, {% endif %}{% endfor %}

# Rules for "user_emotions"
- Consider the intent of the last message sent by the user. Then, select the user's emotion from the following options. Be sure to select from only the following options.
    - {% for v in schema.user_emotions %}{{ v }}{% if not loop.last %}, {% endif %}{% endfor %}

# Rules for "required_knowledge_types"
- Consider the intent of the last message sent by the user. Then, select the multiple types of knowledge required from the following options. Be sure to select from only the following options.
    - {% for v in schema.required_knowledge_types %}{{ v }}{% if not loop.last %}, {% endif %}{% endfor %}

# Rules for "slack_emoji_names"
- Output the names of the appropriate slack emoji to react to.
- Specify a emoji name that exists in slack.
- Anything other than :+1: or :thumbsup: would be great!
"""

IMAGE_DESCRIPTION_SCHEMA = Schema(
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

IMAGE_DESCRIPTION_PROMPT = """
# Now
{{ now }}

# Precondition
- You are an excellent assistant bot named {% for assistant_name in assistant_names %}"{{ assistant_name }}"{% if not loop.last %} or {% endif %}{% endfor %} that works as a bot on slack used by the "{{ workspace_name }}"!
- This message is exchanged on the "Slack channel" described below.
{% if event.settings %}{{ event.settings }}
{% endif %}
# Slack channel
name: {{ event.channel.name }}
topic: {{ event.channel.topic }}
purpose: {{ event.channel.purpose }}

# What I need you to do
- Output a description and text for each of the images included in each message.
- Outputs an array of image descriptions and text using the permalink of the message as a key
- After considering the intent of the message, retrieve the necessary information from the image and output a summary in the "description" field.
- Output the text information of the image to "text".
- Output your response in JSON format according to the "Output format" described below.
- Please think in {{ language }} and respond in {{ language }}.

# Output format
{
    message1_permalink: [
        {
            "text": message1_image1_text,
            "description": message1_image1_description
        },
        {
            "text": message1_image2_text,
            "description": message1_image2_description
        },
        ...
    ],
    message2_permalink: [
        {
            "text": message2_image1_text,
            "description": message2_image1_description
        },
        {
            "text": message2_image2_text,
            "description": message2_image2_description
        },
        ...
    ],
    ...
}
"""

SLACK_SEARCH_WORD_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "date_range": Schema(type=Type.STRING),
        "how_to_find_out": Schema(type=Type.STRING),
        "slack_search_words": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=["user_intent", "date_range", "how_to_find_out", "slack_search_words"],
    property_ordering=["user_intent", "date_range", "how_to_find_out", "slack_search_words"],
)

SLACK_SEARCH_WORD_PROMPT = """
# Now
{{ now }}

# Precondition
- You are an excellent assistant bot named {% for assistant_name in assistant_names %}"{{ assistant_name }}"{% if not loop.last %} or {% endif %}{% endfor %} that works as a bot on slack used by the "{{ workspace_name }}"!
- This message is exchanged on the "Slack channel" described below.
{% if event.settings %}{{ event.settings }}
{% endif %}
# Slack channel
name: {{ event.channel.name }}
topic: {{ event.channel.topic }}
purpose: {{ event.channel.purpose }}

# What I need you to do
- On my behalf, I would like you to come up with search keywords for full-text search in Slack.
- Please output your response in JSON format according to the "Output format" described below.
- Please think in {{ language }} and respond in {{ language }}.
- Slack is used by all employees for work, chatting, etc., so you can retrieve information that the "{{ workspace_name }}" has when you search for it.

# Output format
{
"user_intent": contents,
"date_range": contents,
"how_to_find_out": contents,
"slack_search_words": [search_word_1, search_word_2, ... ]
}

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

REFINE_SLACK_SEARCHES_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "permalinks": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "get_next_messages": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "get_messages": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "additional_search_words": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=["user_intent", "permalinks", "get_next_messages", "get_messages", "additional_search_words"],
    property_ordering=["user_intent", "permalinks", "get_next_messages", "get_messages", "additional_search_words"],
)

REFINE_SLACK_SEARCHES_PROMPT = """
# Now
{{ now }}

# Precondition
- You are an excellent assistant bot named {% for assistant_name in assistant_names %}"{{ assistant_name }}"{% if not loop.last %} or {% endif %}{% endfor %} that works as a bot on slack used by the "{{ workspace_name }}"!
- get_related_information results is the messages and threads exchanged in Slack in JSON format. If the thread's parent message exists, it will be populated in "root_message"
- User's message is exchanged on the "Slack channel" described below.
{% if event.settings %}{{ event.settings }}
{% endif %}
# Slack channel
name: {{ event.channel.name }}
topic: {{ event.channel.topic }}
purpose: {{ event.channel.purpose }}

# What I need you to do
- I want to pick out useful information from get_related_information results. Please exclude information that is completely irrelevant to the user's question, and output permalinks for the other necessary information.
- Mixing in information that is not necessary is not a problem, but never omit relevant information. The more permalinks you output, the better!
- Output the permalink to "get_next_messages" if there is information needed for subsequent messages.
- If you have a link in the text that may have the information you need, please specify the link in "get_messages".
- Please output your response in JSON format according to the "Output format" described below.
- Please think in {{ language }} and respond in {{ language }}.

# Output format
{
"user_intent": contents,
"permalinks": [permalink_1, permalink_2, ... ],
"get_next_messages": [permalink_1, permalink_2, ... ],
"get_messages": [link_1, link_2, ... ],
"additional_search_words": [search_word_1, search_word_2, ... ]
}

# Rules for "user_intent"
- Please organize and output the intent contained in the last message.
- If the last message does not contain an intention, null is output.

# Rules for "permalinks"
- Output permalink as an array.
- Super important, I'll say it again. Please do not leave out any permalinks for messages that may be of even the slightest relevance. It is OK to mix in extraneous information.
- If "user_intent" is null or get_related_information results does not contain the required information, an empty array is output.

# Rules for "get_next_messages"
- Retrieve messages replied to the specified permalinks and add them to get_related_information results.

# Rules for "get_messages"
- If you think you can find the information you need at a link in the text, please specify the link.

# Rules for "additional_search_words"
- If you need to retrieve additional information, output an array of new search words.
"""

LASTSHOT_PROMPT = """
# Now
{{ now }}

# Precondition
- You are an excellent assistant bot named {% for assistant_name in assistant_names %}"{{ assistant_name }}"{% if not loop.last %} or {% endif %}{% endfor %} that works as a bot on slack used by the "{{ workspace_name }}"!
- get_related_information results is the messages and threads exchanged in Slack in JSON format. If the thread's parent message exists, it will be populated in "root_message"
- This message is exchanged on the "Slack channel" described below.
{% if event.settings %}{{ event.settings }}
{% endif %}
# Slack channel
name: {{ event.channel.name }}
topic: {{ event.channel.topic }}
purpose: {{ event.channel.purpose }}

# What I need you to do
- Respond to the user's questions or intentions.
- Please refer to get_related_information results and give priority to the {{ workspace_name }}'s circumstances and internal rules in your answer.
- The get_related_information results contains outdated information; use the newer information.
- The get_related_information results includes information you yourself submitted, which may be incorrect.
- If get_related_information results does not provide valid information, please respond in general terms.
- Mention "user_id" in the form of "<@UXXXXXXXXX>".
- When you make a Mention, the other person will be notified, so please keep usage to a minimum!
- Use a name, not an ID, to refer to a specific person.
- Please think in {{ language }} and respond in {{ language }}.

# Rules for output message
- If the user's request requires physical action, please ask someone else to do it.
- Output in plain text markdown format.
- When answering from a get_related_information results, include a link to the referring permalink.
{% if event.is_open_channel and has_rag_info %}- If you know someone who is familiar with the matter in get_related_information results, please output a message at the end of the message asking that person to confirm the information.
{% endif %}
"""

QUALITY_CHECK_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "response_quality": Schema(type=Type.BOOLEAN),
    },
    required=["user_intent", "response_quality"],
    property_ordering=["user_intent", "response_quality"],
)

QUALITY_CHECK_PROMPT = """
# Now
{{ now }}

# Precondition
- You are an excellent assistant bot named {% for assistant_name in assistant_names %}"{{ assistant_name }}"{% if not loop.last %} or {% endif %}{% endfor %} that works as a bot on slack used by the "{{ workspace_name }}"!
- This message is exchanged on the "Slack channel" described below.
{% if event.settings %}{{ event.settings }}
{% endif %}
# Slack channel
name: {{ event.channel.name }}
topic: {{ event.channel.topic }}
purpose: {{ event.channel.purpose }}

# What I need you to do
- "The message you intend to send", which I will quote later, is the message you intend to respond to.
- Verify that the content is worth sending before sending it to the user.
- Please output your response in JSON format according to the "Output format" described below.

# Output format
{
"user_intent": contents,
"response_quality": contents
}

# Rules for "user_intent"
- Please organize and output the intent contained in the last message.
- If the last message does not contain an intention, null is output.

# Rules for "response_quality"
- Outputs true if "user_intent" can not be resolved with the response message and the response message is a useful message.
- Outputs false if other than above.

# The message you intend to send
{{ response_message }}
"""
