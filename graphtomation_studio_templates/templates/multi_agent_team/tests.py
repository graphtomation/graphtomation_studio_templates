from django.test import TestCase
from unittest.mock import patch
from langgraph.types import Command as GraphCommand
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from .template import (
    MultiAgentTeamBuilderConfig,
    multi_agent_team_graph,
    MultiAgentTeamGraphConfig,
)

multi_agent_team_config: MultiAgentTeamBuilderConfig = {
    "name": "Marketing Strategy Automation",
    "agents": [
        {
            "role": "Lead Market Analyst",
            "goal": "Conduct amazing analysis of the products and competitors, providing in-depth insights to guide marketing strategies.",
            "backstory": "As the Lead Market Analyst at a premier digital marketing firm, you specialize in dissecting online business landscapes.",
            "llm_config": {
                "model_name": "gpt-4o",
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            "max_iter": 5,
            "allow_delegation": True,
        },
        {
            "role": "Chief Marketing Strategist",
            "goal": "Synthesize amazing insights from product analysis to formulate incredible marketing strategies.",
            "backstory": "You are the Chief Marketing Strategist at a leading digital marketing agency, known for crafting bespoke strategies that drive success.",
            "llm_config": {
                "model_name": "gpt-4o",
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            "max_iter": 5,
            "allow_delegation": True,
        },
        {
            "role": "Creative Content Creator",
            "goal": "Develop compelling and innovative content for social media campaigns, with a focus on creating high-impact ad copies.",
            "backstory": "As a Creative Content Creator at a top-tier digital marketing agency, you excel in crafting narratives that resonate with audiences.",
            "llm_config": {
                "model_name": "gpt-4o",
                "max_tokens": 2000,
                "temperature": 0.8,
            },
            "max_iter": 5,
            "allow_delegation": False,
        },
        {
            "role": "Chief Creative Director",
            "goal": "Oversee the work done by your team to make sure it is the best possible and aligned with the product goals, review, approve, ask clarifying questions or delegate follow-up work if necessary.",
            "backstory": "You are the Chief Content Officer at a leading digital marketing agency specializing in product branding.",
            "llm_config": {
                "model_name": "gpt-4o",
                "max_tokens": 2000,
                "temperature": 0.6,
            },
            "max_iter": 5,
            "allow_delegation": True,
        },
    ],
    "tasks": [
        {
            "name": "Research Task",
            "description": "Conduct a thorough research about the customer and competitors in the context of {customer_domain}.",
            "expected_output": "A complete report on the customer and their customers and competitors, including demographics, preferences, market positioning, and audience engagement.",
            "agent_role": "Lead Market Analyst",
            "max_retries": 3,
        },
        {
            "name": "Project Understanding Task",
            "description": "Understand the project details and the target audience for {project_description}.",
            "expected_output": "A detailed summary of the project and a profile of the target audience.",
            "agent_role": "Lead Market Analyst",
            "max_retries": 3,
        },
        {
            "name": "Marketing Strategy Task",
            "description": "Formulate a comprehensive marketing strategy for the project {project_description} of the customer {customer_domain}.",
            "expected_output": "A detailed marketing strategy document with goals, target audience, key messages, proposed tactics, and KPIs.",
            "agent_role": "Chief Marketing Strategist",
            "max_retries": 3,
        },
        {
            "name": "Campaign Idea Task",
            "description": "Develop creative marketing campaign ideas for {project_description}.",
            "expected_output": "A list of 5 campaign ideas, each with a brief description and expected impact.",
            "agent_role": "Creative Content Creator",
            "max_retries": 3,
        },
        {
            "name": "Copy Creation Task",
            "description": "Create marketing copies based on the approved campaign ideas for {project_description}.",
            "expected_output": "Marketing copies for each campaign idea.",
            "agent_role": "Creative Content Creator",
            "max_retries": 3,
        },
    ],
    "manager_agent_config": {
        "role": "Chief Creative Director",
        "goal": "Oversee and review all outputs to ensure quality and alignment with product goals.",
        "llm_config": {
            "model_name": "gpt-4o",
            "max_tokens": 2000,
            "temperature": 0.6,
        },
        "allow_delegation": True,
    },
    "memory": True,
    "cache": True,
    "max_rpm": 60,
    "planning": True,
    "chat_llm_config": {"model_name": "gpt-4o", "max_tokens": 2000, "temperature": 0.7},
    "input_schema": [
        {"field_name": "customer_domain", "field_type": "str"},
        {"field_name": "project_description", "field_type": "str"},
    ],
}


class MultiAgentTeamGraphTests(TestCase):
    def test_multi_agent_team_graph_execution(self):
        graph = multi_agent_team_graph.compile(checkpointer=MemorySaver())
        configurable: MultiAgentTeamGraphConfig = {
            "node_specific_configs": {"team": {"team_config": multi_agent_team_config}},
            "thread_id": "1",
        }
        runnable_config: RunnableConfig = {"configurable": configurable}

        outputs = []
        # Capture initial stream outputs.
        for output in graph.stream(
            input={
                "messages": [
                    (
                        "user",
                        "Help me craft a marketing strategy for my ai automation agency business.",
                    )
                ]
            },
            config=runnable_config,
            interrupt_before=["team_inputs_generator"],
        ):
            outputs.append(output)
            if isinstance(output, dict) and "__interrupt__" in output:
                # Simulate input prompt return "true" to resume execution.
                with patch("builtins.input", return_value="true"):
                    for resumed_output in graph.stream(
                        input=GraphCommand(resume=True),
                        config=runnable_config,
                        interrupt_before=["team_inputs_generator"],
                    ):
                        outputs.append(resumed_output)
                break

        # Assert that outputs were received.
        self.assertGreater(
            len(outputs), 0, "Expected outputs from the multi agent team graph stream."
        )
