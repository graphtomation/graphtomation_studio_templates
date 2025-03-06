import json
from typing_extensions import TypedDict
from typing import Optional,Literal, NotRequired


from langgraph.graph import StateGraph, START
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


from ..lib.utils import pydantic_model_generator
from ..lib.crewai import MultiAgentTeamBuilder, MultiAgentTeamBuilderConfig, AgentConfig, TaskConfig
from ..lib.core import (
    get_node_specific_config,
    transform_to_raw_message,
    BaseGraphState,
    BaseGraphConfig,
)


# SECTION: MultiAgentTeamGraph


class MultiAgentTeamGraphState(BaseGraphState):
    new_team_input: NotRequired[dict]
    last_team_input: NotRequired[dict]
    last_team_response: NotRequired[dict]
    feedbacks: NotRequired[str] = None


class TeamInputGeneratorConfig(TypedDict):
    agent_config: NotRequired[AgentConfig] = None
    task_config: NotRequired[TaskConfig] = None


class TeamConfig(TypedDict):
    team_config: MultiAgentTeamBuilderConfig


class NodeSpecificConfig(TypedDict):
    team: TeamConfig
    input_generator_config: NotRequired[TeamInputGeneratorConfig] = None


class MultiAgentTeamGraphConfig(BaseGraphConfig[NodeSpecificConfig]):
    pass


class MultiAgentTeamGraph:
    def team_inputs_generator(
        self, state: MultiAgentTeamGraphState, config: RunnableConfig
    ) -> MultiAgentTeamGraphState:
        import json

        team_config = get_node_specific_config(
            config, node="team", config_key="team_config"
        )

        if not team_config:
            return {
                "messages": [
                    AIMessage(
                        content="No team configuration was provided. Please ensure a valid configuration is set before execution."
                    )
                ]
            }

        team_input_schema = pydantic_model_generator(
            output_json_configs=team_config.get("input_schema", [])
        )

        # Define default for team_inputs_generator node.
        default_generator_config = {
            "agent_config": {
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
                },
            },
            "task_config": {
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
        }

        team_inputs_generator_config = get_node_specific_config(
            config,
            node="team_inputs_generator",
            default=default_generator_config,
        ) or default_generator_config

        if "task_config" in team_inputs_generator_config:
            team_inputs_generator_config["task_config"][
                "output_json"
            ] = team_input_schema
        else:
            team_inputs_generator_config["task_config"] = {
                "output_json": team_input_schema
            }

        input_generator_config: MultiAgentTeamBuilderConfig = {
            "agents": [team_inputs_generator_config.get("agent_config")],
            "tasks": [team_inputs_generator_config.get("task_config")],
        }

        # Build the input generator crew using the above configuration.
        input_generator = MultiAgentTeamBuilder(config=input_generator_config).build()
        response = input_generator.kickoff(
            inputs={
                "conversation_history": transform_to_raw_message(
                    state.get("messages", [])
                ),
                "human_feedback_on_last_execution": state.get("feedbacks", None),
                "team_config": json.dumps(team_config, indent=2),
            }
        ).json_dict

        # Return an updated state with informative messages and the generated team input.
        return {
            "messages": [
                AIMessage(
                    content="Alright, so i have generated the team inputs based on the provided team configuration."
                ),
            ],
            "new_team_input": response,
        }
    
    def team_inputs_validator(
        self, state: MultiAgentTeamGraphState
    ) -> MultiAgentTeamGraphState:

        value = interrupt(
            {
                "question": "Please review the generated team inputs and make any necessary modifications.",
                "new_team_input": state.get("new_team_input"),
            }
        )
        return {
            **value,
            "messages": [
                AIMessage("Final inputs for the team node have been generated."),
                AIMessage(
                    content=f"```json\n{json.dumps(value.get("new_team_input"), indent=2)}\n```",
                ),
            ],
        }

    def team(
        self, state: MultiAgentTeamGraphState, config: RunnableConfig
    ) -> MultiAgentTeamGraphState:
        team_config = get_node_specific_config(
            config, node="team", config_key="team_config"
        )

        if not team_config:
            return {
                "messages": [
                    AIMessage(
                        content="No team configuration was provided. Please ensure a valid configuration is set before execution."
                    )
                ]
            }

        team = MultiAgentTeamBuilder(team_config).build()
        response = team.kickoff(inputs=state.get("new_team_input"))
        return {
            "last_team_response": response.raw,
            "messages": [
                AIMessage(
                    content="The team has been executed based on the provided inputs."
                ),
                AIMessage(
                    content=response.raw,
                    usage_metadata={
                        "input_tokens": response.token_usage.prompt_tokens,
                        "output_tokens": response.token_usage.completion_tokens,
                        "total_tokens": response.token_usage.total_tokens,
                    },
                ),
            ],
            "last_team_input": state.get("new_team_input"),
            "new_team_input": None,
        }

    def team_result_validator(
        self, state: MultiAgentTeamGraphState
    ) -> Command[Literal["user_feedback_collector", "__end__"]]:
        value = interrupt(
            {
                "question": "Please confirm if the team's execution met your expectations. If not, provide feedback.",
                "last_team_response": state.get("last_team_response"),
            }
        )

        # Use equality check instead of identity check.
        if value.get("is_approved") == False:
            return Command(goto="user_feedback_collector")
        else:
            return Command(goto="__end__")

    def user_feedback_collector(
        self, state: MultiAgentTeamGraphState
    ) -> MultiAgentTeamGraphState:
        value = interrupt(
            {
                "question": "We appreciate your feedback. Please let us know how we can improve the team's performance.",
                "feedbacks": state.get("feedbacks", ""),
            }
        )

        return {
            **value,
            "messages": [
                HumanMessage(content=value.get("feedbacks")),
                SystemMessage(
                    content="Thank you for your feedback. We are refining the response based on your input."
                ),
            ],
        }

    def user_feedback_collector_edge(
        self, state: MultiAgentTeamGraphState
    ) -> Literal["__end__", "team_inputs_generator"]:
        return "team_inputs_generator" if state.get("feedbacks") else "__end__"

    def build(self):
        graph = StateGraph(
            MultiAgentTeamGraphState, config_schema=MultiAgentTeamGraphConfig
        )
        graph.add_node("team_inputs_generator", self.team_inputs_generator)
        graph.add_node("team_inputs_validator", self.team_inputs_validator)
        graph.add_node("team", self.team)
        graph.add_node("team_result_validator", self.team_result_validator)
        graph.add_node(
            "user_feedback_collector",
            self.user_feedback_collector,
        )
        graph.add_edge(start_key=START, end_key="team_inputs_generator")
        graph.add_edge(
            start_key="team_inputs_generator",
            end_key="team_inputs_validator",
        )
        graph.add_edge(start_key="team_inputs_validator", end_key="team")
        graph.add_edge(start_key="team", end_key="team_result_validator")
        graph.add_conditional_edges(
            source="user_feedback_collector", path=self.user_feedback_collector_edge
        )
        return graph


multi_agent_team = MultiAgentTeamGraph().build()
