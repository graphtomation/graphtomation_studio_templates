from .template import multi_agent_team_graph

graph_schema = {
    "name": "Multi Agents Team Graph",
    "state_graph": multi_agent_team_graph,
    "compile_template_args": {
        "checkpointer": {"name": "postgres"},
        "store": {"name": "postgres"},
    },
    "metadata": {
        "description": "A graph that coordinates multiple agents to execute complex tasks collaboratively.",
        "version": "0.0.1",
        "is_published": True,
        "tags": ["multi-agent", "automation", "AI-team"],
        "documentation_path": "templates/multi_agent_team_graph/README.md",
    },
    "input_form_fields": [
        {
            "key": "messages",
            "label": "Messages",
            "type": "messages",
            "description": "Enter the initial messages to guide the multi-agent execution.",
            "placeholder": "Type your messages...",
            "defaultValue": None,
            "options": None,
            "className": "message-input",
            "required": True,
            "required_error_message": "Messages field is required.",
            "needUserInput": True,
        }
    ],
    "node_specific_configs": {
        "team_inputs_generator": [
            {
                "key": "agent_config",
                "label": "Agent Configuration",
                "type": "agent_builder",
                "defaultValue": {
                    "role": "Team Input Generator",
                    "goal": (
                        "Analyze the provided team configuration, conversation history, and any user feedback "
                        "to generate precise inputs for the team node execution. Your output must conform exactly to the provided JSON schema."
                    ),
                    "backstory": (
                        "You are a configuration expert with full knowledge of the team structure (including its agents and tasks), "
                        "as described in the team configuration. You must generate a structured JSON input that reflects the user's requirements "
                        "as discussed in the conversation, and incorporate any improvement suggestions from previous feedback. "
                        "Output only valid JSON that adheres strictly to the schema."
                    ),
                    "llm_config": {
                        "model_name": "gpt-4o",
                        "temperature": 0.2,
                        "max_tokens": 2048,
                        "top_p": 0.9,
                        "system_prompt": (
                            "Output a JSON object that strictly adheres to the provided schema. "
                            "Incorporate the team configuration, the conversation history, and any prior user feedback. "
                            "Do not include any extra text—only the valid JSON object."
                        ),
                    },
                },
                "required": False,
                "needUserInput": False,
            },
            {
                "key": "task_config",
                "label": "Task Configuration",
                "type": "task_builder",
                "defaultValue": {
                    "name": "Generate Team Inputs",
                    "description": (
                        "Generate the inputs for the team node as a JSON object following the provided schema.\n\n"
                        "### Context:\n"
                        "**Team Configuration:**\n{team_config}\n\n"
                        "**Conversation History:**\n{conversation_history}\n\n"
                        "\n**User Feedback:**\n{human_feedback_on_last_execution}"
                        "Using the above, produce a JSON object that precisely captures the user requirements for the team execution. "
                        "Ensure the output strictly conforms to the provided JSON schema."
                    ),
                    "agent_role": "Team Input Generator",
                    "expected_output": "A JSON object that strictly adheres to the provided schema.",
                },
                "required": False,
                "needUserInput": False,
            },
        ],
        "team": [
            {
                "key": "team_config",
                "label": "Team Configuration",
                "type": "team_builder",
                "required": True,
                "needUserInput": False,
                "defaultValue": {
                    "agents": [],
                    "tasks": [],
                    "process": "sequential",
                },
            },
        ],
    },
    "template_schema": {
        "nodes": [
            {
                "label": "__start__",
                "data": {
                    "type": "start",
                    "description": "The entry point of the graph, initiating the workflow.",
                },
            },
            {
                "label": "team_inputs_generator",
                "data": {
                    "type": "ai",
                    "description": "Generates initial team inputs based on the provided messages and configuration.",
                },
            },
            {
                "label": "team_inputs_validator",
                "data": {
                    "type": "human",
                    "description": "Prompts the user to review and validate the generated team inputs before execution.",
                    "interrupt_schemas": [
                        {
                            "key": "new_team_input",
                            "has_question": True,
                            "field": {
                                "key": "new_team_input",
                                "label": "New Team Input",
                                "type": "json",
                                "description": "Enter the new team input JSON object.",
                                "placeholder": "Type the new team input JSON object...",
                                "defaultValue": None,
                                "options": None,
                                "required": True,
                                "required_error_message": "New team input field is required.",
                                "needUserInput": True,
                            },
                        }
                    ],
                },
            },
            {
                "label": "team",
                "data": {
                    "type": "ai",
                    "description": "Executes the multi-agent team based on the validated inputs and generates a response.",
                },
            },
            {
                "label": "team_result_validator",
                "data": {
                    "type": "human",
                    "description": "Asks the user to confirm whether the execution met their expectations or needs further adjustments.",
                    "interrupt_schemas": [
                        {
                            "key": "is_approved",
                            "has_question": True,
                            "field": {
                                "key": "is_approved",
                                "label": "Is Approved?",
                                "type": "approval",
                                "description": "Confirm whether the execution met your expectations.",
                                "placeholder": "Select...",
                                "required": True,
                                "required_error_message": "Approval field is required.",
                                "needUserInput": True,
                            },
                        }
                    ],
                },
            },
            {
                "label": "user_feedback_collector",
                "data": {
                    "type": "human",
                    "description": "Collects user feedback and determines whether to refine and rerun the execution or conclude the process.",
                    "interrupt_schemas": [
                        {
                            "key": "feedbacks",
                            "has_question": True,
                            "field": {
                                "key": "feedbacks",
                                "label": "Feedback",
                                "type": "textarea",
                                "description": "Provide feedback on the execution",
                                "placeholder": "Type your feedback...",
                                "defaultValue": None,
                                "options": None,
                                "required": False,
                                "required_error_message": "Feedback field is required.",
                                "needUserInput": True,
                            },
                        }
                    ],
                },
            },
            {
                "label": "__end__",
                "data": {
                    "type": "end",
                    "description": "The termination point of the graph, marking the completion of the process.",
                },
            },
        ],
        "edges": [
            {"source": "__start__", "target": "team_inputs_generator"},
            {
                "source": "team_inputs_generator",
                "target": "team_inputs_validator",
            },
            {"source": "team_inputs_validator", "target": "team"},
            {"source": "team", "target": "team_result_validator"},
            {
                "source": "team_result_validator",
                "target": "__end__",
                "conditional": True,
            },
            {
                "source": "team_result_validator",
                "target": "user_feedback_collector",
                "conditional": True,
            },
            {
                "source": "user_feedback_collector",
                "target": "__end__",
                "conditional": True,
            },
            {
                "source": "user_feedback_collector",
                "target": "team_inputs_generator",
                "conditional": True,
            },
        ],
    },
}
