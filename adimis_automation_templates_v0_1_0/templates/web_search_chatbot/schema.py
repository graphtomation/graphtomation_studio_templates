from .template import web_search_chatbot

graph_schema = {
    "name": "Web Search Chatbot",
    "state_graph": web_search_chatbot,
    "compile_template_args": {
        "checkpointer": {"name": "postgres"},
        "store": {"name": "postgres"},
    },
    "template_schema": {
        "nodes": [
            {
                "label": "__start__",
                "data": {
                    "type": "start",
                    "description": "The entry point of the graph, initiating the chatbot workflow.",
                },
            },
            {
                "label": "chatbot",
                "data": {
                    "type": "ai",
                    "description": "Processes user messages using a chatbot and generates responses with optional web search integration.",
                },
            },
            {
                "label": "tools",
                "data": {
                    "type": "tool",
                    "description": "Handles web search queries using DuckDuckGo and provides results to the chatbot.",
                    "tools": ["duck-duck-go-search"],
                },
            },
            {
                "label": "__end__",
                "data": {
                    "type": "end",
                    "description": "Marks the completion of the chatbot's interaction session.",
                },
            },
        ],
        "edges": [
            {"source": "__start__", "target": "chatbot"},
            {"source": "chatbot", "target": "tools", "conditional": True},
            {"source": "tools", "target": "chatbot"},
            {"source": "chatbot", "target": "__end__", "conditional": True},
        ],
    },
    "metadata": {
        "description": "A chatbot that uses DuckDuckGo search to help users find information.",
        "version": "0.0.1",
        "is_published": True,
        "tags": ["search", "information"],
        "documentation_path": "templates/web_search_chatbot/README.md",
    },
    "input_form_fields": [
        {
            "key": "messages",
            "label": "Messages",
            "type": "messages",
            "description": "Enter your messages here.",
            "placeholder": "Type your messages...",
            "defaultValue": None,
            "options": None,
            "className": "message-input",
            "required": True,
            "required_error_message": "Messages field is required.",
            "needUserInput": True,
        }
    ],
    "config_form_fields": [
        {
            "key": "system_prompt",
            "label": "System Prompt",
            "type": "textarea",
            "description": "Enter the system prompt for the chatbot.",
            "placeholder": "Type system prompt...",
            "defaultValue": "You are a helpful assistant.",
            "required": False,
            "needUserInput": True,
        },
    ],
    "node_specific_configs": {
        "chatbot": [
            {
                "key": "llm_config",
                "label": "LLM Configuration",
                "type": "llm_builder",
                "defaultValue": {
                    "model_name": "gpt-4o",
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "top_p": 0.9,
                    "system_prompt": "You are a helpful assistant.",
                },
                "description": "Define llm configurations for the chatbot.",
                "options": [
                    {"label": "GPT-3.5 Turbo", "value": "gpt-3.5-turbo"},
                    {"label": "GPT-4o", "value": "gpt-4o"},
                    {"label": "GPT-4o", "value": "gpt-4o"},
                    {"label": "o1", "value": "o1"},
                    {"label": "o3", "value": "o3"},
                ],
                "required": False,
            },
        ]
    },
}
