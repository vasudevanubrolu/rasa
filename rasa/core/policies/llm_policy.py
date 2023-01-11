from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, POLICY_PRIORITY
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.policies.policy import Policy, PolicyPrediction, SupportedData
from rasa.engine.graph import ExecutionContext
from rasa.shared.core.events import BotUttered
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_SEND_TEXT
from rasa.shared.exceptions import FileIOException, RasaException
import rasa.shared.utils.io
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain, InvalidDomain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker

from langchain import LLMChain, OpenAI, ConversationChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversation.prompt import SUMMARY_PROMPT
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.schema import AgentAction, AgentFinish


logger = logging.getLogger(__name__)

_SKILL_PROMPT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

{skill_examples}

Current conversation:
{history}

Human: {input}
AI:"""


def noop(*args):
    """Does nothing."""
    return ""


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=False
)
class LLMPolicy(Policy):
    """Policy which uses a language model to generate the next action."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            POLICY_PRIORITY: DEFAULT_POLICY_PRIORITY,
        }

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        By default, this is only ML-based training data. If policies support rule data,
        or both ML-based data and rule data, they need to override this method.

        Returns:
            The data type supported by this policy (ML-based training data).
        """
        return SupportedData.ML_DATA

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
        skills: Optional[Dict] = None,
    ) -> None:
        """Constructs a new Policy object."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)
        self.skills = skills or {}
        self.max_iterations = 10

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        """Trains a policy.

        Args:
            training_trackers: The story and rules trackers from the training data.
            domain: The model's domain.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to train itself.

        Returns:
            A policy must return its resource locator so that potential children nodes
            can load the policy from the resource.
        """
        self.skills = self._create_skills_from_trackers(training_trackers, domain)
        self.persist()
        return self._resource

    def _create_skills_from_trackers(
        self, training_trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> Dict:
        """Creates a dictionary of skills from the training trackers."""
        skill_examples = defaultdict(list)

        for tracker in training_trackers:
            if tracker.is_augmented:
                continue
            skill_name = tracker.sender_id
            dialogue = self._tracker_as_readable_dialogue(tracker)
            skill_examples[skill_name].append(dialogue)

        skills = {}
        for skill_name, examples in skill_examples.items():
            if skill_name not in domain.skills:
                raise InvalidDomain(
                    f"Skill '{skill_name}' is used in the training data, but is not "
                    f"listed in the domain file."
                )

            skills[skill_name] = {
                "examples": examples,
                "description": domain.skills[skill_name].get("description", ""),
            }

        return skills

    def _tracker_as_readable_dialogue(self, tracker: TrackerWithCachedStates) -> Text:
        """Creates a readable dialogue from a tracker."""
        dialogue = []
        # TODO: need to remove any potential new lines from these strings
        for event in tracker.applied_events():
            if event.type_name == "user":
                dialogue.append(f"Human: {event.text}")
            if event.type_name == "bot":
                dialogue.append(f"AI: {event.text}")
        return "\n".join(dialogue)

    @classmethod
    def _skills_filename(cls) -> Text:
        return "skills.json"

    def _skills_info(self) -> Dict[Text, Any]:
        return {"skills": self.skills}

    def persist(self) -> None:
        """Persists the policy to storage."""
        with self._model_storage.write_to(self._resource) as path:
            file = Path(path) / self._skills_filename()

            rasa.shared.utils.io.create_directory_for_file(file)
            rasa.shared.utils.io.dump_obj_as_json_to_file(file, self._skills_info())

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
        """
        if tracker.events and tracker.events[-1].type_name == BotUttered.type_name:
            # if the last event was a bot utterance, we don't want to predict anything
            # futher as this is the end of the turn
            result = self._prediction_result(ACTION_LISTEN_NAME, tracker, domain)
            return self._prediction(result)

        dialogue = self._tracker_as_readable_dialogue(tracker)

        llm = OpenAI(temperature=0.7)
        summarizer = LLMChain(llm=llm, prompt=SUMMARY_PROMPT, verbose=True)
        # ideally we don't fully recompute but already store the previous summary somewhere
        # and use it for this prompt
        summary = summarizer.predict(summary="", new_lines=dialogue).strip()

        logger.info(f"Summary: '{summary}'")
        # need to handle skills:
        tools = [
            Tool(name=name, func=noop, description=skill.get("description", ""))
            for name, skill in self.skills.items()
        ]

        prompt = ConversationalAgent.create_prompt(tools)
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        agent = ConversationalAgent(llm_chain=llm_chain, ai_prefix="AI")

        # need to pass in proper prior steps. i guess based on tracker
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        iterations = 0
        chat_prediction = None
        # We now enter the agent loop (until it returns something).
        while iterations < self.max_iterations:

            agent_result = agent.plan(
                input=tracker.latest_message.text,
                intermediate_steps=intermediate_steps,
                chat_history=summary,
            )

            if isinstance(agent_result, AgentFinish):
                llm.callback_manager.on_agent_finish(agent_result, color="green")
                chat_prediction = agent_result.return_values.get("output")
                break

            if agent_result.tool in self.skills:
                llm.callback_manager.on_tool_start(
                    {"name": agent_result.tool}, agent_result, color="green"
                )
                observation = self._run_skill(
                    llm, summary, self.skills[agent_result.tool], tracker
                )
            else:
                llm.callback_manager.on_tool_start(
                    {"name": "N/A"}, agent_result, color="green"
                )
                observation = (
                    f"{agent_result.tool} is not a valid tool, try another one."
                )
            llm.callback_manager.on_tool_end(
                observation,
                color="pink",
                observation_prefix=agent.observation_prefix,
                llm_prefix=agent.llm_prefix,
            )
            intermediate_steps.append((agent_result, observation))
            iterations += 1

        logger.debug(f"Predicting action probabilities.")
        if chat_prediction:
            predicted_action_name = ACTION_SEND_TEXT
        else:
            predicted_action_name = None

        logger.debug(f"Predicted the next action '{predicted_action_name}'")
        result = self._prediction_result(predicted_action_name, tracker, domain)

        return self._prediction(result, action_metadata={"text": chat_prediction})

    def _run_skill(self, llm, summary, skill, tracker):
        prompt = PromptTemplate(
            input_variables=["history", "input", "skill_examples"],
            template=_SKILL_PROMPT_TEMPLATE,
        )

        chat_predictor = LLMChain(llm=llm, prompt=prompt, verbose=True)
        return chat_predictor.predict(
            input=tracker.latest_message.text,
            history=summary,
            skill_examples="\n\n".join(skill["examples"]),
        )

    def _prediction_result(
        self, action_name: Optional[Text], tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        result = self._default_predictions(domain)
        if action_name:
            result[domain.index_for_action(action_name)] = 1.0

        return result

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMPolicy":
        """Loads a trained policy (see parent class for full docstring)."""
        lookup = None

        try:
            with model_storage.read_from(resource) as path:
                skills_file = Path(path) / cls._skills_filename()
                skills_info = rasa.shared.utils.io.read_json_file(skills_file)
                skills = skills_info["skills"]

        except (ValueError, FileNotFoundError, FileIOException):
            logger.warning(
                f"Couldn't load metadata for policy '{cls.__name__}' as the persisted "
                f"metadata couldn't be loaded."
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            skills=skills,
        )
