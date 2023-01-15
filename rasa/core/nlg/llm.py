import copy
import logging

from langchain import LLMChain, OpenAI, PromptTemplate

from rasa.shared.core.trackers import DialogueStateTracker
from typing import Text, Tuple, Any, Dict, Optional, List

from rasa.core.nlg import interpolator
from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from rasa.shared.constants import RESPONSE_CONDITION, CHANNEL
from langchain.chains.conversation.prompt import SUMMARY_PROMPT
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import CallbackManager

logger = logging.getLogger(__name__)


_RESPONSE_VARIATION_PROMPT_TEMPLATE = """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. Rephrase the AI response staying close to the original message and retaining its meaning.

Summary of the conversation:
{history}

{current_input}
AI Response: {thought}
Rephrased Response:"""


class LoggerCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_end(self, response: LLMResult) -> None:
        """Do nothing."""
        pass

    def on_llm_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        logger.info(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Print out that we finished a chain."""
        logger.info(f"Finished chain.")

    def on_chain_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        action: AgentAction,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        logger.debug(action.log)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        logger.debug(observation_prefix)
        logger.debug(output)
        logger.debug(llm_prefix)

    def on_tool_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        logger.debug(f"on_text: {text}")

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        logger.debug(f"on_agent_finish: {finish.log}")


class LLMNaturalLanguageGenerator(TemplatedNaturalLanguageGenerator):
    """Generates responses based on modified templates."""

    def _tracker_as_readable_dialogue(self, tracker: DialogueStateTracker) -> Text:
        """Creates a readable dialogue from a tracker."""
        dialogue = []
        # TODO: need to remove any potential new lines from these strings
        for event in tracker.applied_events():
            if event.type_name == "user":
                dialogue.append(f"Human: {event.text}")
            if event.type_name == "bot":
                dialogue.append(f"AI: {event.text}")
        return "\n".join(dialogue)

    def summarize(self, tracker: DialogueStateTracker, llm: OpenAI):
        """Summarizes the dialogue using the LLM."""
        dialogue = self._tracker_as_readable_dialogue(tracker)
        manager = CallbackManager(handlers=[LoggerCallbackHandler()])
        summarizer = LLMChain(
            llm=llm,
            prompt=SUMMARY_PROMPT,
            verbose=True,
            callback_manager=manager,
        )
        # ideally we don't fully recompute but already store the previous summary somewhere
        # and use it for this prompt
        try:
            return summarizer.predict(summary="", new_lines=dialogue).strip()
        except Exception as e:
            logger.error(
                f"Error while summarizing dialogue: {dialogue} with error: {e}"
            )
            return dialogue

    def latest_message_from_tracker(
        self, tracker: DialogueStateTracker
    ) -> Tuple[Text, Text]:
        """Returns the latest message from the tracker.

        Returns:
            Tuple of (speaker, message).
        """
        for event in reversed(tracker.applied_events()):
            if event.type_name == "user":
                return ("Human", event.text)
            if event.type_name == "bot":
                return ("AI", event.text)
        return ("Human", "")

    def predict_variation(
        self,
        response: Optional[Dict[Text, Any]],
        tracker: DialogueStateTracker,
        llm: OpenAI,
    ):
        """Predicts a variation of the response."""
        if not response or "text" not in response:
            return response
        summary = self.summarize(tracker, llm)
        prompt = PromptTemplate(
            input_variables=["history", "current_input", "thought"],
            template=_RESPONSE_VARIATION_PROMPT_TEMPLATE,
        )

        speaker, latest_message = self.latest_message_from_tracker(tracker)
        if speaker == "Human":
            current_input = f"Human: {latest_message}"
        else:
            # if last message is from AI, we skip adding it
            current_input = ""
        manager = CallbackManager(handlers=[LoggerCallbackHandler()])
        chat_predictor = LLMChain(
            llm=llm, prompt=prompt, verbose=True, callback_manager=manager
        )
        updated_text = chat_predictor.predict(
            history=summary,
            thought=response["text"],
            current_input=current_input,
        )
        response["text"] = updated_text
        return response

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action."""
        llm = OpenAI(temperature=0.7)

        filled_slots = tracker.current_slot_values()
        templated_response = self.generate_from_slots(
            utter_action, filled_slots, output_channel, **kwargs
        )
        return self.predict_variation(templated_response, tracker, llm)
