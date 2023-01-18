from __future__ import annotations
import logging
import re
from typing import Any, Dict, Optional, Text, List

from langchain import LLMChain, OpenAI, PromptTemplate

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from langchain.callbacks.base import CallbackManager

from rasa.utils.llm_utils import LoggerCallbackHandler

logger = logging.getLogger(__name__)

_INTENT_CLASSIFICATION_PROMPT_TEMPLATE = """Label the last message from a human with a category.

{examples}

Message: {message}
Category:"""


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class LLMIntentClassifier(GraphComponent, IntentClassifier):
    """Intent classifier using the langchain library."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {"fallback_intent": "out_of_scope"}

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        example_embeddings: Optional[List[Dict]] = None,
    ) -> None:
        """Creates classifier."""
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        self.fallback_intent = self.component_config.get("fallback_intent")
        self.example_embeddings = example_embeddings or []
        self.available_intents = {e["intent"] for e in self.example_embeddings}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LLMIntentClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the intent classifier on a data set."""
        from langchain.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()

        embedding_results = embeddings.embed_documents(
            [ex.get(TEXT, "") for ex in training_data.intent_examples]
        )

        self.example_embeddings = [
            {"intent": ex.get(INTENT), "embedding": embedding, "text": ex.get(TEXT)}
            for embedding, ex in zip(embedding_results, training_data.intent_examples)
        ]

        self.persist()
        return self._resource

    def process(self, messages: List[Message]) -> List[Message]:
        """Sets the message intent and add it to the output if it exists."""
        llm = OpenAI(temperature=0.7)

        for message in messages:
            closest_examples = self._find_closest_examples(message)
            intent_name = self._classify(message, closest_examples, llm)

            logger.debug("LLM intent classifier predicted intent: " f"'{intent_name}'")
            if intent_name not in self.available_intents:
                intent_name = self.fallback_intent

            intent = {
                INTENT_NAME_KEY: intent_name,
                PREDICTED_CONFIDENCE_KEY: 1.0,
            }

            if intent_name:
                message.set(INTENT, intent, add_to_output=True)

        return messages

    def _classify(
        self, message: Message, examples: List[Dict], llm: OpenAI
    ) -> Optional[Text]:
        """Classify the message using an LLM."""
        manager = CallbackManager(handlers=[LoggerCallbackHandler()])
        prompt = PromptTemplate(
            input_variables=["examples", "message"],
            template=_INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
        )
        classifier = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            callback_manager=manager,
        )
        example_prompts = "\n\n".join(
            f"Message: {e['text']}\nCategory: {e['intent']}" for e in examples
        )

        try:
            return classifier.predict(
                examples=example_prompts, message=message.get(TEXT, "")
            ).strip()
        except Exception as e:
            logger.error(f"Error while classifying message: {message} with error: {e}")
            return None

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_dir:
            file_name = f"{self.__class__.__name__}.json"
            embedding_file = model_dir / file_name
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                embedding_file, self.example_embeddings
            )

    def _find_closest_examples(self, message: Message):
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores.utils import maximal_marginal_relevance

        embeddings = OpenAIEmbeddings()

        embedded_message = embeddings.embed_query(message.get(TEXT))

        embeddings = [e["embedding"] for e in self.example_embeddings]

        mmr_example_idxs = maximal_marginal_relevance(embedded_message, embeddings, k=3)

        return [self.example_embeddings[i] for i in mmr_example_idxs]

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> LLMIntentClassifier:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                embedding_file = model_dir / f"{cls.__name__}.json"
                example_embeddings = rasa.shared.utils.io.read_json_file(embedding_file)
        except ValueError:
            logger.warning(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            example_embeddings = None

        return cls(
            config, model_storage, resource, execution_context, example_embeddings
        )
