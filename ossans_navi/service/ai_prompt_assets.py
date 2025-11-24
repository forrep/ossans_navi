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
- Please respond in raw JSON format without including Markdown code blocks, in accordance with JSON Schema.
{% if event.has_image_video_audio %}- The attached image, video, and audio content will be analyzed and entered later. Please respond to this task on the assumption that they have been entered.
{% endif %}

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

# Now
{{ now }}
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
- Please respond in raw JSON format without including Markdown code blocks, in accordance with JSON Schema.
- Please think in {{ language }} and respond in {{ language }}.

# Now
{{ now }}
"""

SLACK_SEARCH_WORD_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "date_range": Schema(type=Type.STRING),
        "how_to_find_out": Schema(type=Type.STRING),
        "slack_search_words": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "external_urls": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=["user_intent", "date_range", "how_to_find_out", "slack_search_words", "external_urls"],
    property_ordering=["user_intent", "date_range", "how_to_find_out", "slack_search_words", "external_urls"],
)

SLACK_SEARCH_WORD_PROMPT = """
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
- Please respond in raw JSON format without including Markdown code blocks, in accordance with JSON Schema.
- Please think in {{ language }} and respond in {{ language }}.
- Slack is used by all employees for work, chatting, etc., so you can retrieve information that the "{{ workspace_name }}" has when you search for it.
{% if event.has_image_video_audio %}- The attached image, video, and audio content will be analyzed and entered later. Please respond to this task on the assumption that they have been entered.
{% endif %}

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
- If "user_intent" is time-related, please use the date filter, do not include relative dates and times in the string, such as "last year", "this year", or "recently".
- To obtain a wide range of search results, please include search terms with only one word, not just words that use AND search.

# Rules for "external_urls"
- If the message contains an external URL relevant to the "user_intent", output that URL as an array.
- If "user_intent" is null, output an empty array in "external_urls".

# Now
{{ now }}
"""

REFINE_SLACK_SEARCHES_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "user_intent": Schema(type=Type.STRING, nullable=True),
        "permalinks": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
        "additional_search_words": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
    },
    required=["user_intent", "permalinks", "additional_search_words"],
    property_ordering=["user_intent", "permalinks", "additional_search_words"],
)

REFINE_SLACK_SEARCHES_PROMPT = """
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
- Please respond in raw JSON format without including Markdown code blocks, in accordance with JSON Schema.
- Please think in {{ language }} and respond in {{ language }}.
{% if event.has_image_video_audio %}- The attached image, video, and audio content will be analyzed and entered later. Please respond to this task on the assumption that they have been entered.
{% endif %}

# Rules for "user_intent"
- Please organize and output the intent contained in the last message.
- If the last message does not contain an intention, null is output.

# Rules for "permalinks"
- Output permalink as an array.
- Super important, I'll say it again. Please do not leave out any permalinks for messages that may be of even the slightest relevance. It is OK to mix in extraneous information.
- If "user_intent" is null or get_related_information results does not contain the required information, an empty array is output.

# Rules for "additional_search_words"
- If you need to retrieve additional information, output an array of new search words.

# Now
{{ now }}
"""

URL_CONTEXT_PROMPT = """
- Retrieve the content from the provided URL and summarize it according to the "user_intent".
- Please think in {{ language }} and respond in {{ language }}.

## user_intent
{{ event.user_intent }}

# Now
{{ now }}
"""

LASTSHOT_PROMPT = """
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
- If get_related_information results does not provide related information, please respond in general terms.
- Mention "user_id" in the form of "<@UXXXXXXXXX>".
- When you make a Mention, the other person will be notified, so please keep usage to a minimum!
- Use a name, not an ID, to refer to a specific person.
- Please think in {{ language }} and respond in {{ language }}.
{% if not event.is_mention %}- Please respond briefly.
{% endif %}

# Rules for output message
- If the user's request requires physical action, please ask someone else to do it.
- Output in plain text markdown format.
- When answering from a get_related_information results, include a link to the referring permalink.
{% if event.is_open_channel and has_rag_info %}- If you know someone who is familiar with the matter in get_related_information results, please output a message at the end of the message asking that person to confirm the information.
{% endif %}

# Now
{{ now }}
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
- Please respond in raw JSON format without including Markdown code blocks, in accordance with JSON Schema.
- Please think in {{ language }} and respond in {{ language }}.

# Rules for "user_intent"
- Please organize and output the intent contained in the last message.
- If the last message does not contain an intention, null is output.

# Rules for "response_quality"
- Outputs true if "user_intent" can be resolved with the response message and the response message is a useful message.
- Outputs false if other than above.

# The message you intend to send
{{ response_message }}

# Now
{{ now }}
"""
