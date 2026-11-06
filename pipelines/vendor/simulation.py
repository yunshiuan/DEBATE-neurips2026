"""Adapted research simulator; see SOURCE_MANIFEST.json for provenance and edits."""
# prompt_version = "prompt_v6_ablation"
prompt_version = "prompt_v5"
simulation_version = "simulation_v2"

import os
from pathlib import Path
import random

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
MODEL_FAMILY_OVERRIDE = None
import pandas as pd
import argparse
import asyncio
import typing
import re
import ast
import json
import sys
import pathlib
from datetime import datetime


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



SAFE_HF_GENERATION_PATCH = True
BLOCK_MY_RESPONSE_LABEL = False

OUTPUT_LABEL_VARIANTS = {
    "response": ("response", "reponse", "responce", "respones", "responed"),
    "tweet": ("tweet", "twet", "tweeet", "tweat"),
}

BARE_LABEL_CONTENT_BLOCKLIST = ("to", "is", "was", "would", "will", "can", "could", "should", "about", "for")
ARTIFACT_PREFIX_RE = re.compile(r"^\s*my\s+a\s+(?P<content>.+)$", re.IGNORECASE | re.DOTALL)


def _strip_artifact_prefixes(text: str) -> str:
    artifact_match = ARTIFACT_PREFIX_RE.match(text)
    return artifact_match.group("content").strip() if artifact_match else text


def _extract_exact_label(text: str, expected_label: typing.Literal["response", "tweet"]) -> str:
    text = _strip_artifact_prefixes(text)

    variants = OUTPUT_LABEL_VARIANTS[expected_label]
    exact_label = variants[0]
    flags = re.IGNORECASE | re.DOTALL

    exact_match = re.match(
        rf"^\s*my\s+{re.escape(exact_label)}\s*(?:[:：\-–]\s*|\n+)(?P<content>.*)$",
        text,
        flags,
    )
    if exact_match:
        return _strip_artifact_prefixes(exact_match.group("content").strip())

    blocklist_pattern = "|".join(re.escape(word) for word in BARE_LABEL_CONTENT_BLOCKLIST)
    bare_exact_match = re.match(
        rf"^\s*my\s+{re.escape(exact_label)}\s+(?!(?:{blocklist_pattern})\b)(?P<content>.*)$",
        text,
        flags,
    )
    if bare_exact_match:
        return _strip_artifact_prefixes(bare_exact_match.group("content").strip())

    typo_pattern = "|".join(re.escape(variant) for variant in variants[1:])
    typo_match = re.match(
        rf"^\s*my\s+(?:{typo_pattern})\b\s*(?:[:：\-–]\s*)?(?P<content>.*)$",
        text,
        flags,
    )
    return _strip_artifact_prefixes(typo_match.group("content").strip()) if typo_match else text



class HFChatMessage(dict):
    @property
    def type(self):
        return self["role"]
    
    @property
    def content(self):
        return self["content"]


class MTChatMessageHistory(InMemoryChatMessageHistory):
    hf_messages: typing.List[typing.Dict[str, str]] = []
    hf_concat: bool = False
    api_type: str = "openai"
    strict_role_alternation: bool = False
    
    def __init__(self, api_type: str, strict_role_alternation: bool = False):
        super().__init__()
        # self.hf_concat = api_type == "huggingface"
        self.hf_concat = True
        self.api_type = api_type
        self.strict_role_alternation = strict_role_alternation
    
    def remove_last_message(self):
        self.messages.pop()
        self.hf_messages.pop()
    
    def clear_user_messages(self):
        """Remove all user/human messages from the conversation history."""
        # Remove user messages from the main messages list
        self.messages = [msg for msg in self.messages if not isinstance(msg, HumanMessage)]
        
        # Remove user messages from the HuggingFace messages list
        user_role = "user"
        self.hf_messages = [msg for msg in self.hf_messages if msg["role"] != user_role]
    
    def get_last_three_messages(self):
        return self.messages[-3:]
    
    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system" if isinstance(message, SystemMessage) else "unknown"
        self.hf_messages.append(HFChatMessage({
            "role": role,
            "content": message.content
        }))
    
    def hf_concat_user_messages(self):
        # Concatenate consecutive user messages into a single user message, while keeping other messages
        new_hf_messages = []
        current_user_message = ""
        for message in self.hf_messages:
            if message["role"] == "user":
                current_user_message += message["content"] + "\n"
            else:
                if current_user_message:
                    new_message = HFChatMessage({"role": "user", "content": current_user_message})
                    new_hf_messages.append(new_message)
                    current_user_message = ""
                new_hf_messages.append(message)
        if current_user_message != "":
            new_message = HFChatMessage({"role": "user", "content": current_user_message})
            new_hf_messages.append(new_message)

        if self.strict_role_alternation and new_hf_messages and new_hf_messages[-1]["role"] == "user":
            new_hf_messages.append(HFChatMessage({"role": "assistant", "content": "Acknowledged."}))

        return new_hf_messages
    
    def get_messages(self) -> typing.List[BaseMessage | HFChatMessage]:
        return self.hf_concat_user_messages()
    
    async def aget_messages(self) -> typing.List[BaseMessage | HFChatMessage]:
        return self.hf_concat_user_messages()


class Agent:
    def __init__(
            self, 
            agent_id, 
            agent_name, 
            all_agent_names, 
            persona, 
            initial_opinion_verbal, 
            initial_opinion_likert, 
            api_type, 
            model_name, 
            temperature, 
            seed, 
            max_tokens, 
            prompt_template_root, 
            topic, 
            log_path, 
            model = None, 
            tokenizer = None, 
            endpoint_url = None,
            is_dpo_breadth = False,
            ):
        self.api_type = api_type
        self.model_name = model_name
        self.agent_id = agent_id
        self.prompt_template_root = prompt_template_root
        self.persona = persona
        self.initial_opinion_verbal = initial_opinion_verbal
        self.initial_opinion_likert = initial_opinion_likert
        self.agent_name = agent_name
        self.topic = topic
        self.log_path = log_path  # Path to the log files
        self.conversation_log_path = os.path.join(self.log_path, "conversation_log.txt")
        self.conversation_log_memory_path = os.path.join(self.log_path, "conversation_log_memory.txt")
        self.payload_debug_path = os.path.join(self.log_path, "llm_payload_debug.jsonl")

        # print("max tokens: ", max_tokens, flush=True)

        persona_prompt = HumanMessagePromptTemplate.from_template(self.persona)

        if self.api_type == "openai":
            
            if temperature is None:
                temperature = 0.7
            if model_name == "gpt-5-mini":
                max_tokens = 2048  # hard code this because the default 100 tokens is not enough for reasoning models like gpt 5 mini

            chat_model = None
            if model_name == "gpt-5-mini":
                chat_model = ChatOpenAI(model_name=model_name, max_completion_tokens=max_tokens, reasoning_effort="medium") # hard code this because the default 100 tokens is not enough for reasoning models like gpt 5 mini
            else:
                chat_model = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, seed=seed, verbose=True)

            assert chat_model is not None, "Chat model initialization failed for OpenAI API"

        elif self.api_type == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if temperature is None:
                temperature = 0.7 # This is because, Gemini reasoning models work best at temperature of 1.0, using 0.7 with Gemini3.0+ can cause degraded reasoning performance, failure on complex tasks. More here: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
            chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_tokens=max_tokens, seed=seed, verbose=True, google_api_key=os.getenv("GEMINI_API_KEY"))
        elif self.api_type == "huggingface":
            model_family, model_type = self.get_model_family(model_name)

            if temperature is None:
                if model_type == "base":
                    temperature = 0.3
                else:
                    temperature = 0.7 # edit this
                    if is_dpo_breadth:
                        temperature = 0.2

            bad_words = [
                "You", 
                "User", 
                ] + [str(name) for name in all_agent_names]
            if not SAFE_HF_GENERATION_PATCH or BLOCK_MY_RESPONSE_LABEL:
                bad_words.append("My Response:")

            extra_body = {
                "bad_words": bad_words,
                "repetition_penalty": 1.2,
                "min_tokens": 5,
            }
            if SAFE_HF_GENERATION_PATCH:
                extra_body["top_p"] = 0.9
            if "qwen" in model_name.lower():
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                openai_api_base=endpoint_url,
                openai_api_key="EMPTY",
                verbose=True,
                extra_body=extra_body,
            )

        history_message_holder = MessagesPlaceholder(variable_name="history")
        question_placeholder = HumanMessagePromptTemplate.from_template("{input}")

        # Instead of changing the code here, self.get_sys_prompt dynamically parses system prompt based on the simulation version and prompt version defined at the top of the file.
        sys_prompt = self.get_sys_prompt(prompt_version)

        # Log the prompt to file
        log_interaction_to_file(self.conversation_log_path, f"Input (System Message): {sys_prompt}")
        systems_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)
        mistral = (
            self.api_type == "huggingface"
            and model_family == "mistral" and model_type == "instruct"
        )
        self.mistral = mistral
        self.sys_prompt = sys_prompt
        self.chat_model = chat_model
        self.debug_llm_payload = (
            mistral
            or os.getenv("MT_DEBUG_LLM_PAYLOAD", "0") == "1"
        )

        self.history = MTChatMessageHistory(api_type=self.api_type, strict_role_alternation=mistral)
        if mistral:
            chat_prompt = ChatPromptTemplate.from_messages(
                [systems_prompt, history_message_holder, question_placeholder]
            )
        else:
            chat_prompt = ChatPromptTemplate.from_messages(
                [systems_prompt, persona_prompt, history_message_holder, question_placeholder]
            )
        self.chat_prompt = chat_prompt

        chain = chat_prompt | chat_model | StrOutputParser()
        wrapped_chain = RunnableWithMessageHistory(
            chain,
            self.get_history,
            history_messages_key="history"
        )
        self.memory = wrapped_chain

        if self.api_type == "huggingface" and not mistral:
            if model_family == "mistral":
                sys_message = HumanMessage(content=sys_prompt)
                self.history.add_message(sys_message)
            else:
                sys_message = SystemMessage(content=sys_prompt)
                self.history.add_message(sys_message)

    def get_sys_prompt(self, prompt_version : str) -> str:
        sys_prompt = ""

        # Normal condition
        if prompt_version == "prompt_v5":
            sys_prompt = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step1_persona.md", self.api_type, self.model_name)
            sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            AGENT_PERSONA=self.persona, TOPIC=self.topic,
            INITIAL_OPINION_VERBAL=self.initial_opinion_verbal, INITIAL_OPINION_LIKERT=self.initial_opinion_likert
            )

        elif prompt_version == "prompt_v6_ablation":
            sys_prompt = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step1_persona.md", self.api_type, self.model_name)
            sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(TOPIC=self.topic)

        elif prompt_version == "prompt_v6_ablation_v3":
            sys_prompt = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step1_persona.md", self.api_type, self.model_name)
            sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            TOPIC=self.topic, INITIAL_OPINION_VERBAL=self.initial_opinion_verbal, INITIAL_OPINION_LIKERT=self.initial_opinion_likert
            )
        elif prompt_version == "prompt_v6_ablation_v4" or prompt_version == "prompt_v6_ablation_v5" or prompt_version == "prompt_v6_ablation_v6":
            sys_prompt = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step1_persona.md", self.api_type, self.model_name)
            sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            AGENT_PERSONA=self.persona, TOPIC=self.topic
            )
        
        return sys_prompt

    @staticmethod
    def get_model_family(model_name: str) -> typing.Tuple[typing.Literal["mistral", "llama", "tulu", "olmo"], typing.Literal["base", "instruct"]]:
        if MODEL_FAMILY_OVERRIDE is not None:
            return MODEL_FAMILY_OVERRIDE
        model_name_parts = model_name.split("/")
        if model_name_parts[0] == "HuggingFaceH4":
            return "mistral", "instruct"
        elif model_name_parts[0] == "RLHFlow":
            return "llama", "instruct"
        elif model_name_parts[0] == "allenai" and model_name_parts[1].startswith("OLMo"):
            if not ("SFT" in model_name_parts[1] or "DPO" in model_name_parts[1] or "Instruct" in model_name_parts[1]):
                return "olmo", "base"
            else:
                return "olmo", "instruct"
        elif model_name_parts[0] == "allenai" and model_name_parts[1].startswith("Llama"):
            return "tulu", "instruct"
        elif model_name_parts[0] == "mistralai" and "Instruct" in model_name_parts[1]:
            return "mistral", "instruct"
        elif model_name_parts[0] == "mistralai" and "Instruct" not in model_name_parts[1]:
            return "mistral", "base"
        elif model_name_parts[0] == "meta-llama" and "Instruct" in model_name_parts[1]:
            return "llama", "instruct"
        elif model_name_parts[0] == "meta-llama" and "Instruct" not in model_name_parts[1]:
            return "llama", "base"
        elif "mini-twitter/Llama-3.1-Tulu-3-8B-MT-DDPO" in model_name:
            return "tulu", "instruct"
        elif "llama_3.1_8b_QLoRA_left" in model_name or "llama_3.1_8b_QLoRA_May2" in model_name:
            return "llama", "instruct"
        elif "Qwen" in model_name:
            return "llama", "instruct"
        elif "mini-twitter/" in model_name and "Llama" in model_name:
            return "llama", "instruct"
        else:
            raise NotImplementedError(f"{model_name} is not supported for generation")


    @staticmethod
    def get_prompt_template(original_root: str, simulation_version: str, prompt_version: str, template_file: str, api_type: typing.Literal["openai", "huggingface", "gemini"], model_name: str) -> str:
        def content(file_path):
            with open(file_path, "r") as f:
                return f.read()

        if api_type == "huggingface":
            model_family, model_type = Agent.get_model_family(model_name)
            if model_type == "base" and template_file == "step2_generate_response.md":
                template = content(os.path.join(original_root, "base_models", f"step2_generate_response_{prompt_version}.md"))
                if model_family == "mistral" or model_family == "olmo":
                    return template.replace("{AFTER_PROMPT}", r"When you are finished, speak [END].")
                elif model_family == "llama":
                    return template.replace("{AFTER_PROMPT}", r"Start your response immediately after <|start_header_id|>assistant<|end_header_id|>.")
                
        return content(os.path.join(original_root, simulation_version, prompt_version, template_file))


    @staticmethod
    def get_model_archive_name(model_name: str) -> str:
        splits = model_name.split("/")
        if len(splits) == 2:
            if "llama_3.1_8b_QLoRA_left" in splits[1]:
                return "llama_3.1_8b_QLoRA_left"
            return splits[1]  # e.g., Mistral-7B-v0.3
        if "llama_3.1_8b_QLoRA_left" in model_name:
            return "llama_3.1_8b_QLoRA_left"
        if "llama_3.1_8b_QLoRA_May2" in model_name:
            return "llama_3.1_8b_QLoRA_May2"
        # Served names can be local absolute paths. Keep outputs under the
        # requested run directory rather than treating that name as a path.
        return re.sub(r"[^A-Za-z0-9_.:@+-]+", "_", model_name).strip("_")


    def get_history(self):
        """Get the chat history of the agent."""

        return self.history

    def _normalize_role(self, role: str) -> str:
        role_map = {
            "human": "user",
            "ai": "assistant",
        }
        return role_map.get(role, role)

    def _serialize_prompt_message(self, message):
        role = self._normalize_role(getattr(message, "type", "unknown"))
        return {
            "role": role,
            "content": message.content,
        }

    def _log_llm_payload(self, prompt: str, explicit_messages: typing.Optional[typing.List[typing.Dict[str, str]]] = None):
        if not self.debug_llm_payload:
            return

        try:
            if explicit_messages is not None:
                payload_messages = [
                    {
                        "role": self._normalize_role(msg.get("role", "unknown")),
                        "content": msg.get("content", ""),
                    }
                    for msg in explicit_messages
                ]
                payload_roles = [msg["role"] for msg in payload_messages]
            else:
                history_messages = self.history.hf_concat_user_messages()
                rendered_messages = self.chat_prompt.format_messages(
                    history=history_messages,
                    input=prompt,
                )
                payload_messages = [self._serialize_prompt_message(msg) for msg in rendered_messages]
                payload_roles = [msg["role"] for msg in payload_messages]

            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent_name": self.agent_name,
                "model_name": self.model_name,
                "api_type": self.api_type,
                "message_count": len(payload_messages),
                "roles": payload_roles,
                "messages": payload_messages,
            }
            with open(self.payload_debug_path, "a+") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            with open(os.path.join(self.log_path, "WARN.txt"), "a+") as f:
                f.write(f"[WARN] Failed to log llm payload: {e}\n")

    def _invoke_with_mistral_payload(self, prompt: str) -> str:
        history_messages = self.history.hf_concat_user_messages()
        payload_messages = [{"role": "system", "content": self.sys_prompt}]
        payload_messages.extend(
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            for msg in history_messages
        )
        payload_messages.append({"role": "user", "content": prompt})

        self._log_llm_payload(prompt, explicit_messages=payload_messages)
        response_message = self.chat_model.invoke(payload_messages)
        response_text = response_message.content if hasattr(response_message, "content") else str(response_message)

        self.history.add_user_message(prompt)
        self.history.add_ai_message(response_text)
        return response_text

    def inform_about_conversation(self, second_agent_name):
        prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step1_inform_about_conversation.md", self.api_type, self.model_name)
        prompt = prompt_template.format(
            SECOND_AGENT_NAME=second_agent_name
        )
        self.history.add_user_message(prompt)

        # Log the prompt to file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        self.log_memory_step("Inform About Conversation", prompt)
    
    def add_to_memory_tweet(self, tweet_written, tweet_received, second_agent_name):
        prompt_instructions = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step2_add_to_memory_tweet.md", self.api_type, self.model_name)
        prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
            AGENT_NAME=self.agent_name,
            TOPIC=self.topic,
            TWEET_WRITTEN=tweet_written,
            TWEET_RECEIVED=tweet_received,
            SECOND_AGENT_NAME=second_agent_name
        )

        self.history.add_user_message(prompt)
        
        # Log the prompt to file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        self.log_memory_step("Add Tweet to Memory", prompt)



    def add_response_to_memory(self, response, interaction_type, second_agent_name):
        if interaction_type == "read":
            prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step3_add_to_memory_read.md", self.api_type, self.model_name)
            prompt = prompt_template.format(
                MESSAGE_RECEIVED=response,
                TOPIC=self.topic,
                AGENT_NAME=self.agent_name,
                SECOND_AGENT_NAME=second_agent_name
            )
        elif interaction_type == "write":
            prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step3_add_to_memory_write.md", self.api_type, self.model_name)
            prompt = prompt_template.format(
                MESSAGE_WRITTEN=response,
                TOPIC=self.topic,
                AGENT_NAME=self.agent_name,
                SECOND_AGENT_NAME=second_agent_name
            )

        self.history.add_user_message(prompt)
        
        # Log the prompt to file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        self.log_memory_step(f"Add {interaction_type.capitalize()} Response to Memory", prompt)

    def generate_response(self, second_agent_name, max_length=100):
        """
        Generate a response from the agent.

        Args:
            list_historical_messages (list): A list of historical messages.
            second_agent_name (str): The name of the second agent.
            max_length (int): The maximum length of the response.   

        Returns:
            str: The generated response from the agent.
        """
        
        prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step2_generate_response.md", self.api_type, self.model_name)
        prompt = prompt_template.format(
            # TODO: Change the parameters when using the v1 prompt
            AGENT_NAME=self.agent_name,
            SECOND_AGENT_NAME=second_agent_name,
        )

        # Log the input prompt to the file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        if not self.mistral:
            self._log_llm_payload(prompt)

        if self.api_type == "huggingface":
            prompt = prompt.encode("ascii", errors="ignore").decode("ascii")
        if self.mistral:
            response = self._invoke_with_mistral_payload(prompt)
        else:
            response = self.memory.invoke({"input": prompt})

        new_response = response

        self.history.remove_last_message()  # remove the AI message for processing - required when we call external API which returns an AI message

        # Get the agent memory before removing the generation prompt
        memory = asyncio.run(self.history.aget_messages())
        memory_content = "\n".join([f"$${msg.type}$$: {msg.content}" for msg in memory])
        self.history.remove_last_message()  # remove generation prompt from history to avoid prompt repetition
        
        def get_error_message(reason: str, prompt: typing.Optional[str], response: typing.Optional[str], mem: bool) -> str:
            message = f"[WARN] {reason}\n"
            if prompt:
                message += f"[INFO] prompt=\n{prompt}\n\n"
            if response:
                message += f"[INFO] response=\n{response}\n\n"
            if mem:
                message += f"[INFO] last_3_mem=\n{self.history.get_last_three_messages()}\n"
            return message
        
        if new_response == "":
            err_message = get_error_message("Empty response", prompt, response, True)
            print(err_message, file=sys.stderr)
            with open(os.path.join(self.log_path, "WARN.txt"), "a+") as f:
                f.write(err_message)
        response_unfiltered = new_response
        response = new_response  # set for return

        response = _extract_exact_label(response, expected_label="response")

        # Log the output to the file
        log_interaction_to_file(self.conversation_log_path, f"Output (Assistant Message, {self.agent_name}): {response}")
        self.log_memory_step("Generate Response", f"Prompt: {prompt}\nResponse: {response}")

        return response, response_unfiltered, memory_content

    def generate_tweet(self, next_player):
        prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step2_generate_tweet.md", self.api_type, self.model_name)
        prompt = prompt_template.format(
            TOPIC=self.topic,
            NEXT_PLAYER=next_player
        )

        # Log the input prompt to the file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        if self.mistral:
            response = self._invoke_with_mistral_payload(prompt)
        else:
            self._log_llm_payload(prompt)
            response = self.memory.invoke({"input": prompt})


        self.history.remove_last_message()  # remove the AI message for processing
        self.history.remove_last_message()  # remove generation prompt from history to avoid prompt repetition

        response_unfiltered = response
        response = _extract_exact_label(response, expected_label="tweet")

        # Log the output to the file
        log_interaction_to_file(self.conversation_log_path, f"Output (Assistant Message, {self.agent_name}): {response}")
        self.log_memory_step("Generate Tweet", f"Prompt: {prompt}\nResponse: {response}")

        return response, response_unfiltered
        
    
    def generate_post_opinion(self, is_likert_scale: bool = False):
        """
        Generate a post opinion from the agent.

        Args:
            is_likert_scale (bool): Whether the post opinion is verbal or likert scale.
        Returns:
            str: The generated post opinion from the agent.
        """
        if is_likert_scale:
            prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step4_likert_scale.md", self.api_type, self.model_name)
            prompt = prompt_template.split("\n---------------------------\n")[0].format(
                TOPIC=self.topic
            )
        else:
            prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step4_generate_post_opinion.md", self.api_type, self.model_name)
            prompt = prompt_template.split("\n---------------------------\n")[0].format(
                TOPIC=self.topic
            )

        # Log the input prompt to the file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        if self.mistral:
            response = self._invoke_with_mistral_payload(prompt)
        else:
            self._log_llm_payload(prompt)
            response = self.memory.invoke({"input": prompt})

        self.history.remove_last_message()  # remove the AI message for processing
        self.history.remove_last_message()  # remove generation prompt from history to avoid prompt repetition

        response_unfiltered = response
        response = _extract_exact_label(response, expected_label="response")

        # Log the output to the file
        log_interaction_to_file(self.conversation_log_path, f"Output (Assistant Message, {self.agent_name}): {response}")
        self.log_memory_step("Generate Response", f"Prompt: {prompt}\nResponse: {response}")

        return response, response_unfiltered

    def change_round(self, previous_agent_name, next_agent_name):
        prompt_template = self.get_prompt_template(self.prompt_template_root, simulation_version, prompt_version, "step3_change_round.md", self.api_type, self.model_name)
        prompt = prompt_template.format(
            PREVIOUS_AGENT_NAME=previous_agent_name,
            NEXT_AGENT_NAME=next_agent_name
        )
        self.history.add_user_message(prompt)
        # Log the prompt to file
        log_interaction_to_file(self.conversation_log_path, f"Input (User Message): {prompt}")
        self.log_memory_step(f"Change Round", prompt)
    
    def log_memory_step(self, step_description, content):
        """
        Log a step in the agent's memory to a file.

        Args:
            step_description (str): A description of the current step.
            content (str): The content to be logged.
        """
        with open(self.conversation_log_memory_path, "a+") as f:
            f.write(f"Step: {step_description}\n")
            f.write(f"Content: {content}\n")
            f.write("-" * 50 + "\n\n")

def log_interaction_to_file(file_path, text: str):
    """
    Log an interaction to a file.

    This function appends the given text to the specified file, adding a newline
    after each interaction for better readability.

    Args:
        file_path (str): The path to the file where the interaction should be logged.
        text (str): The text of the interaction to be logged.

    Returns:
        None
    """
    with open(file_path, "a+") as f:
        f.write(text + "\n\n")  # Append the text to the file, separating each interaction with a newline

def get_sender_id_column_name(user_data):
    sender_list = user_data["sender_id"].dropna().unique()
    for possible_column_name in ["worker_id", "empirica_id"]:
        player_list = user_data[possible_column_name].dropna().unique()
        if len(set(sender_list).intersection(set(player_list))) > 0:
            return possible_column_name
    raise ValueError("Could not find the sender_id column name in the user data")

def get_tweet(user_data, player, round_number):
    """
    Retrieves a tweet from the user_data DataFrame based on the player and round number.

    Args:
        user_data (pd.DataFrame): The DataFrame containing tweet data.
        player (str): The name or identifier of the player who sent the tweet.
        round_number (int): The round number of the tweet.

    Returns:
        str: The text content of the tweet for the specified player and round.
             Returns the longest matching tweet if multiple matches are found.
    """
    try:
        matching_tweets = user_data[(user_data["sender_id"] == player) & (user_data["chat_round_order"] == round_number) & 
                                   (user_data["event_type"] == "tweet")]["text"]
        
        if len(matching_tweets) == 0:
            print(f"[WARN] No tweet found for player {player} in round {round_number}")
            return ""
        elif len(matching_tweets) == 1:
            tweet = matching_tweets.iloc[0]
        else:
            # If multiple matches, keep the longest one
            tweet = matching_tweets.loc[matching_tweets.str.len().idxmax()]
            print(f"[INFO] Multiple tweets found for player {player} in round {round_number}, keeping the longest one")
            
    except IndexError:
        print(f"[WARN] No tweet found for player {player} in round {round_number}")
        return ""
    return tweet


def get_initial_opinion(user_data, player, player_column_name):
    """
    Retrieves the initial opinion of a player from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        player (str): The name or identifier of the player.

    Returns:
        str: The initial opinion of the player.
    """
    opinion_slider_value_mapping = {
    "1": "Certainly disagree",
    "2": "Probably disagree",
    "3": "Lean disagree",
    "4": "Lean agree",
    "5": "Probably agree",
    "6": "Certainly agree",
    "unknown": "unknown"
}
    matching_opinions = user_data[(user_data[player_column_name] == player) & (user_data["event_type"] == "Initial Opinion")]["text"]
    
    if len(matching_opinions) == 0:
        print(f"[WARN] No initial opinion found for player {player}")
        return "", opinion_slider_value_mapping["unknown"]
    elif len(matching_opinions) == 1:
        initial_opinion = matching_opinions.iloc[0]
    else:
        # If multiple matches, keep the longest one
        matching_rows = user_data[(user_data[player_column_name] == player) & (user_data["event_type"] == "Initial Opinion")]
        longest_text_idx = matching_rows['text'].str.len().idxmax()
        initial_opinion = matching_rows.loc[longest_text_idx, 'text']
        print(f"[INFO] Multiple initial opinions found for player {player}, keeping the longest one")
    
    try:
        slider_value = str(int(user_data[(user_data[player_column_name] == player) & (user_data["event_type"] == "Initial Opinion")]["sliderValue"].iloc[0]))
    except:
        slider_value = "unknown"
    return initial_opinion, opinion_slider_value_mapping[slider_value]

def get_demographic_background(user_data, api_type, model_name, log_path: str, player_column_name):
    """
    Retrieves the demographic background of each player from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        api_type (str): The type of API to use for the agents (openai or huggingface).
        model_name (str): The name of the model to use for generating responses.

    Returns:
        dict: A dictionary where keys are player names and values are their demographic backgrounds.
    """
    # Read the markdown file once and store its content
    demographic_template = Agent.get_prompt_template(str(PROMPT_ROOT), simulation_version, prompt_version, "step0_demographics.md", api_type, model_name)
    player_list = user_data[player_column_name].dropna().unique()
    demographic_backgrounds = {}
    df_fields = ["age", "gender", "education", "ethnicity", "income", "politicalIdentity", "politicalViews", "childrenSchool", "residence", "maritalStatus", "bibleBelief", "evangelical", "religion", "occupation"]
    for player in player_list:
        player_survey = user_data[(user_data['event_type'] == 'exit_survey') & (user_data[player_column_name] == player)]
        player_survey_fields = player_survey['field'].tolist()
        if len(player_survey) == 0:
            demographic_backgrounds[player] = ""
            print(f"[WARN] Player {player} missing demographic data, skip demographics in simulation", file=sys.stderr)
            log_interaction_to_file(os.path.join(log_path, "WARN.txt"), f"[WARN] Player {player} missing demographic data, skip demographics in simulation")
            continue
        elif len(player_survey) < len(df_fields):
            print(f"[WARN] Player {player} missing some demographic data, plug in 'unknown' in demographics of simulation", file=sys.stderr)
            log_interaction_to_file(os.path.join(log_path, "WARN.txt"), f"[WARN] Player {player} missing some demographic data, plug in 'unknown' in demographics of simulation")
        player_data = {field: player_survey[player_survey['field'] == field]['text'].iloc[0] if field in player_survey_fields else "unknown" for field in df_fields}
        try:
            player_data["ethnicity"] = ", ".join(ast.literal_eval(player_data["ethnicity"]))
        except (ValueError, SyntaxError, TypeError):
            print(f"[WARN] Player {player} has invalid ethnicity {player_data['ethnicity']}, skip formatting it", file=sys.stderr)
        camel_to_snake = lambda s: re.sub(r'([A-Z])', r'_\1', s).upper()  # for example, 'politicalIdentity' -> 'POLITICAL_IDENTITY'
        template_fill = {camel_to_snake(field): player_data[field] for field in df_fields}
        demographic_background = demographic_template.format(**template_fill)  # format the demographic template with the player's data
        demographic_backgrounds[player] = demographic_background

    return demographic_backgrounds

def get_conversation_history(user_data, player1, player2, player_column_name):
    """
    Retrieves the conversation history between two players from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing message data.
        player1 (str): The name or identifier of the first player.
        player2 (str): The name or identifier of the second player.
    Returns:
        list: A list of tuples, where each tuple contains (sender, message_text).
              The list represents the conversation history between the two players.
    """
    # The number of messages are sent and received (tweet and initial opinion not included)
    conversation_history = []
    sent_messages = user_data[
        (user_data["event_type"] == "message_sent") & 
        (((user_data["sender_id"] == player1) & (user_data["recipient_id"] == player2)) | 
         ((user_data["sender_id"] == player2) & (user_data["recipient_id"] == player1)))
    ]

    if sent_messages.empty:
        return conversation_history

    for index, row in sent_messages.iterrows():
        current_player = player1 if row["sender_id"] == player1 or row[player_column_name] == player1 else player2
        current_message = row["text"]
        if pd.isna(current_message) or current_message.strip() == "":
            continue
        
        conversation_history.append((current_player, current_message))
    return conversation_history

def get_pairs(user_data, round_number, agents_v2) -> typing.List[typing.Tuple[Agent, Agent]]:
    """
    Retrieves the pairs of players for a given round number.

    Args:
        user_data (pd.DataFrame): The DataFrame containing message data.
        round_number (int): The round number to retrieve pairs for.
        agents_v2 (list): The list of agents.
    Returns:
        list: A list of tuples, where each tuple contains (sender_agent, recipient_agent).
              The list represents the pairs of agents for the specified round.
    """
    pairs = user_data[(user_data["chat_round_order"] == round_number) & (user_data["event_type"] != "tweet")][["sender_id", "recipient_id"]].dropna().drop_duplicates().values.tolist()
    pairs = list(set(tuple(sorted(pair)) for pair in pairs))
    agent_pairs = []
    for sender_id, recipient_id in pairs:
        sender_agent = next((agent for agent in agents_v2 if agent.agent_name == sender_id), None)
        recipient_agent = next((agent for agent in agents_v2 if agent.agent_name == recipient_id), None)
        if sender_agent and recipient_agent:
            agent_pairs.append((sender_agent, recipient_agent))
    return agent_pairs

def get_previous_agent(user_data, agent_name, round_number):
    previous_agent = user_data[(user_data["sender_id"] == agent_name) & (user_data["chat_round_order"] == round_number)]["recipient_id"].iloc[0]
    return previous_agent

def get_event_order(user_data, event_type, text):
    """
    Retrieves the event order of a message from the user data.
    """
    event_order = int(user_data[(user_data['event_type'] == event_type) & (user_data['text'] == text)]["event_order"].iloc[0])
    return event_order

def get_invalid_players(user_data):
    """
    Retrieves the invalid players from the user data.
    """
    # Get players without tweets in all 3 rounds
    # players_with_tweets = user_data[(user_data['event_type'] == 'tweet') & (user_data['text'].notna()) & (user_data['text'].str.strip() != '')]['sender_id'].apply(lambda x: x[:5]).value_counts()
    # players_missing_tweets = players_with_tweets[players_with_tweets < 3].index.tolist()

    # Get players who don't have a tweet in each round
    all_players = user_data['sender_id'].dropna().apply(lambda x: x[:5]).unique()
    players_missing_tweets = []
    
    for player in all_players:
        player_tweets = user_data[(user_data['event_type'] == 'tweet') & 
                                (user_data['sender_id'].str.startswith(player)) &
                                (user_data['text'].notna()) & 
                                (user_data['text'].str.strip() != '')]
        
        # Check if player has a tweet in each round
        rounds_with_tweets = player_tweets['chat_round_order'].unique()
        if len(rounds_with_tweets) < 3:
            players_missing_tweets.append(player)
    
    # Get players without exit survey
    all_players = user_data['sender_id'].dropna().apply(lambda x: x[:5]).unique()
    players_with_survey = user_data[user_data['event_type'] == 'exit_survey']['worker_id'].dropna().apply(lambda x: x[:5]).unique()
    players_missing_survey = [p for p in all_players if p not in players_with_survey]
    
    # Get players without any message_sent
    players_with_messages = user_data[user_data['event_type'] == 'message_sent']['sender_id'].dropna().apply(lambda x: x[:5]).unique()
    players_missing_messages = [p for p in all_players if p not in players_with_messages]
    
    # Combine all invalid players
    invalid_players = list(set(players_missing_tweets + players_missing_survey + players_missing_messages))
    return invalid_players


def concatenate_messages(user_data):
    """
    Concatenates consecutive messageSent entries and updates messageReceived entries.
    
    Args:
        user_data (pd.DataFrame): The DataFrame containing message data.

    Returns:
        pd.DataFrame: The updated DataFrame with concatenated messages.
    """

    df = user_data.copy()
    # first_message_sent_index = df[df['type'] == 'messageSent'].index[0]
    previous_sender, previous_sender_recipient = None, None
    previous_recipient, previous_recipient_sender = None, None
    last_recipient_index = 0
    last_write_index = 0
    
    for index, row in df.iterrows():
        if row['event_type'] == 'message_sent':
            if previous_sender == row['sender_id'] and previous_sender_recipient == row['recipient_id']:
                df.at[last_write_index, 'text'] += ' ' + row['text']
                df.at[index, 'text'] = ''

                df.at[last_write_index, 'time'] = row['time']
                df.at[last_write_index, 'end_time'] = row['end_time']
            else:
                last_write_index = index
                previous_sender = row['sender_id']
                previous_sender_recipient = row['recipient_id']
        elif row['event_type'] == 'message_recieved':
            if previous_recipient == row['recipient_id'] and previous_recipient_sender == row['sender_id']:
                df.at[last_recipient_index, 'text'] += ' ' + row['text']
                df.at[index, 'text'] = ''

                df.at[last_recipient_index, 'time'] = row['time']
                df.at[last_recipient_index, 'end_time'] = row['end_time']
            else:
                last_recipient_index = index
                previous_recipient = row['recipient_id']
                previous_recipient_sender = row['sender_id']
    return df

def concat_with_pairs(user_data, agents_v2):
    """
    Concatenates consecutive messageSent entries and updates messageReceived entries.
    """
    num_rounds = int(user_data["chat_round_order"].max())
    
    for round in range(1, num_rounds + 1):
        pairs = get_pairs(user_data, round, agents_v2)
        for pair in pairs:
            sender, recipient = pair
            pair_mask = (
                ((user_data['sender_id'] == sender.agent_name) & (user_data['recipient_id'] == recipient.agent_name)) | 
                ((user_data['sender_id'] == recipient.agent_name) & (user_data['recipient_id'] == sender.agent_name))
            )
            pair_data = user_data[pair_mask]
            previous_sender = None
            previous_recipient = None

            last_recipient_event_order = 0
            last_write_event_order = 0

            for index, row in pair_data.iterrows():
                if row['event_type'] == 'message_sent':
                    if previous_sender == row['sender_id'] and previous_sender_recipient == row['recipient_id']:
                        user_data.loc[(user_data['event_order'] == last_write_event_order), 'text'] += ' ' + row['text']
                        user_data.loc[(user_data['event_order'] == row['event_order']), 'text'] = ''

                        user_data.loc[(user_data['event_order'] == last_write_event_order), 'time'] = row['time']
                        user_data.loc[(user_data['event_order'] == last_write_event_order), 'end_time'] = row['end_time']
                    else:
                        last_write_event_order = row['event_order']
                        previous_sender = row['sender_id']
                        previous_sender_recipient = row['recipient_id']
                elif row['event_type'] == 'message_recieved':
                    if previous_recipient == row['recipient_id'] and previous_recipient_sender == row['sender_id']:
                        user_data.loc[(user_data['event_order'] == last_recipient_event_order), 'text'] += ' ' + row['text']
                        user_data.loc[(user_data['event_order'] == row['event_order']), 'text'] = ''

                        user_data.loc[(user_data['event_order'] == last_recipient_event_order), 'time'] = row['time']
                        user_data.loc[(user_data['event_order'] == last_recipient_event_order), 'end_time'] = row['end_time']
                    else:
                        last_recipient_event_order = row['event_order']
                        previous_recipient = row['recipient_id']
                        previous_recipient_sender = row['sender_id']

    return user_data

def write_conversation_history_to_dataframe(conversation_output, user_data, version, data_prefix):
    """
    Write the conversation history to a dataframe.

    Args:
        conversation_output (dict): The conversation history.
        user_data (pd.DataFrame): The user data.
    Returns:
        pd.DataFrame: Updated user data with LLM-generated text.
    """
    user_data_copy = user_data.copy()
    # drop all nan values in event_order
    user_data_copy = user_data_copy[user_data_copy['event_order'].notna()]
    column_name = "llm_text"
    unfiltered_column_name = "llm_text_unfiltered"
    user_data_copy[column_name] = pd.Series(index=user_data_copy.index, dtype='object').fillna('')
    user_data_copy[unfiltered_column_name] = pd.Series(index=user_data_copy.index, dtype='object').fillna('')


    user_data_copy['event_order'] = user_data_copy['event_order'].astype(int)
    
    for round in conversation_output:
        round_number = int(round.split('_')[1])
        for pair in conversation_output[round]:
            for message_index, message in enumerate(conversation_output[round][pair]):
                if len(message) == 3:
                    sender, message_text, message_text_unfiltered = message
                else:
                    sender, message_text = message
                    message_text_unfiltered = message_text
                recipient = pair[0] if sender == pair[1] else pair[1]
                
                # Find the corresponding row in user_data_copy
                mask = (
                    ((user_data_copy['sender_id'] == sender) | (user_data_copy['sender_id'] == recipient)) &
                    (user_data_copy['chat_round_order'] == round_number) &
                    (user_data_copy['text'].notna()) &
                    (user_data_copy['text'] != '') &
                    (user_data_copy['event_type'] == 'message_sent')
                )
                
                matching_rows = user_data_copy[mask]

                if message_index >= len(matching_rows):
                    print(f"[WARN] Data Prefix: {data_prefix}")
                    print(f"[WARN] message_index {message_index} out of bounds for {len(matching_rows)} matching rows")
                    continue

                message_row = matching_rows.iloc[message_index]
                message_row_event_order = int(message_row['event_order'])
                    

                if message_row['event_type'] == 'tweet':
                    user_data_copy.loc[user_data_copy['event_order'] == message_row_event_order, column_name] = message_row['text']
                    user_data_copy.loc[user_data_copy['event_order'] == message_row_event_order, unfiltered_column_name] = message_row['text']
                else:
                    user_data_copy.loc[user_data_copy['event_order'] == message_row_event_order, column_name] = message_text
                    user_data_copy.loc[user_data_copy['event_order'] == message_row_event_order, unfiltered_column_name] = message_text_unfiltered
    
    # Append the exit_survey to the end of the dataframe
    exit_survey = user_data[user_data['event_type'] == 'exit_survey']
    user_data_copy = user_data_copy._append(exit_survey, ignore_index=True)

    return user_data_copy

def write_step_memory(user_data, event_order, agent, col_name, memory_content=None):
    """
    Write the agent's current memory state to the dataframe for a specific step.
    
    Args:
        user_data_copy (pd.DataFrame): The dataframe to update
        event_order (int): The event order of the current step
        agent (Agent): The agent whose memory to write
        isSent (bool): Whether the message is sent or received
        memory_content (str): The memory content to write
    """
    if memory_content is None:
        memory = asyncio.run(agent.history.aget_messages())
        memory_content = "\n".join([f"$${msg.type}$$: {msg.content}" for msg in memory])
    user_data.loc[
        (user_data['event_order'] == event_order) & 
        (user_data['event_type'] == col_name), 
        'input_prompt'
    ] = memory_content
    return user_data

def write_tweets_to_dataframe(tweet_dict, user_data):
    """
    Write the tweets to the dataframe.
    """
    if not tweet_dict:
        return user_data
    if 'llm_text_unfiltered' not in user_data.columns:
        user_data['llm_text_unfiltered'] = ''
    for tweet in tweet_dict:
        tweet_value = tweet_dict[tweet]
        if isinstance(tweet_value, tuple):
            tweet_text, tweet_text_unfiltered = tweet_value
        else:
            tweet_text = tweet_value
            tweet_text_unfiltered = tweet_value
        user_data.loc[user_data['text'] == tweet, 'llm_text'] = tweet_text
        user_data.loc[user_data['text'] == tweet, 'llm_text_unfiltered'] = tweet_text_unfiltered
    return user_data

def write_post_opinions_to_dataframe(post_opinion_dict, agreement_dict, user_data, player_column_name):
    """
    Write the post opinions to the dataframe.
    """
    user_data['agreement_level'] = ''
    if 'llm_text_unfiltered' not in user_data.columns:
        user_data['llm_text_unfiltered'] = ''
    for post_agent in post_opinion_dict:
        post_opinion_value = post_opinion_dict[post_agent]
        if isinstance(post_opinion_value, tuple):
            post_opinion_text, post_opinion_text_unfiltered = post_opinion_value
        else:
            post_opinion_text = post_opinion_value
            post_opinion_text_unfiltered = post_opinion_value
        end_memory = asyncio.run(post_agent.history.aget_messages())
        memory_content = "\n".join([f"$${msg.type}$$: {msg.content}" for msg in end_memory])
        user_data.loc[
            (user_data['event_type'] == 'Post Opinion') & 
            (user_data['worker_id'] == post_agent.agent_name), 
            'input_prompt'
        ] = memory_content
        user_data.loc[(user_data['event_type'] == 'Post Opinion') & (user_data[player_column_name] == post_agent.agent_name), 'llm_text'] = post_opinion_text
        user_data.loc[(user_data['event_type'] == 'Post Opinion') & (user_data[player_column_name] == post_agent.agent_name), 'llm_text_unfiltered'] = post_opinion_text_unfiltered
        user_data.loc[(user_data['event_type'] == 'Post Opinion') & (user_data[player_column_name] == post_agent.agent_name), 'agreement_level'] = agreement_dict[post_agent]
    return user_data

def post_processing(user_data, version, invalid_players):
    """
    Post-process the user data.
    """
    if version == "v1":
        column_name = "llm_text"
    elif version in ("v0", "v1_5", "v2"):
        column_name = "llm_text"
    unfiltered_column_name = "llm_text_unfiltered"
    if unfiltered_column_name not in user_data.columns:
        user_data[unfiltered_column_name] = ''
    for index, row in user_data.iterrows():
        if row['worker_id'] in invalid_players:
            continue
        if row['event_type'] == 'tweet':
            if version in ("v0", "v1_5"):
                if row['chat_round_order'] == 1:
                    user_data.at[index, column_name] = row['text']
                    user_data.at[index, unfiltered_column_name] = row['text']
                else:
                    user_data.at[index, column_name] = ''
                    user_data.at[index, unfiltered_column_name] = ''
            else:
                user_data.at[index, column_name] = row['text']
                user_data.at[index, unfiltered_column_name] = row['text']
        elif row['event_type'] == 'Initial Opinion' or row['event_type'] == 'Post Opinion':
            user_data.at[index, column_name] = row['text']
            user_data.at[index, unfiltered_column_name] = row['text']
        elif row['text'] == '':
            user_data.at[index, column_name] = ''
            user_data.at[index, unfiltered_column_name] = ''
        else:
            if row['event_type'] == 'message_sent':
                if row['text'] == '':
                    continue
                llm_text = row[column_name]
                llm_text_unfiltered = row[unfiltered_column_name]
                message = row['text']
                user_data.loc[(user_data['event_type'] == 'message_recieved') & (user_data['text'] == message), column_name] = llm_text
                user_data.loc[(user_data['event_type'] == 'message_recieved') & (user_data['text'] == message), unfiltered_column_name] = llm_text_unfiltered
    return user_data

def fix_names(user_data, player_column_name):
    """
    Fix the names of the agents.
    """
    # The released sessions have unique five-character participant prefixes.
    # Reject collisions in new inputs instead of merging distinct participants.
    full_names = user_data[player_column_name].dropna().astype(str).unique()
    if len({name[:5] for name in full_names}) != len(full_names):
        raise ValueError("Participant IDs must have distinct first five characters within each session")
    # Convert sender_id and recipient_id to str type and keep first 5 digits
    user_data['sender_id'] = user_data['sender_id'].fillna('').apply(lambda x: str(x)[:5] if x else '')
    user_data['recipient_id'] = user_data['recipient_id'].fillna('').apply(lambda x: str(x)[:5] if x else '')
    user_data[player_column_name] = user_data[player_column_name].fillna('').apply(lambda x: str(x)[:5] if x else '')
    return user_data


def main(
    api_type: typing.Literal["openai", "huggingface", "gemini"] = "openai", 
    model_name: str = "gpt-4o-mini-2024-07-18", 
    temperature: typing.Optional[float] = None, 
    seed: int = 1, 
    max_tokens: int = 100, 
    topic: typing.Optional[str] = None, 
    data_prefix: str = "20241015_020620_Economic_growth_always_occurs_when_taxes_are_lowered_01JA6XECGBRVBCQC7GF94VZQQ8", 
    version: str = "v2", 
    model = None, 
    tokenizer = None, 
    endpoint_url: typing.Optional[str] = None,
    is_dpo_breadth: bool = False,
    input_path=None,
    output_root=None,
    prompt_root=None,
    model_family=None,
    ):

    """
    Main function to simulate a conversation between agents.
    
    Args:
        api_type (str): The type of API to use for the agents (openai or huggingface).
        model_name (str): The name of the model to use for generating responses.
        temperature (float): The temperature parameter for the model.
        seed (int): The seed for reproducibility (only for OpenAI API).
        max_tokens (int): The maximum number of tokens for the model's responses.
        conversation (list): The initial conversation history.
        prompt_template_root (str): The root directory for prompt templates.
        topic (str): The topic of the conversation.
        data_prefix (str): The name of user data file without the CSV extension.
        version (str): Version of simulation to run ("v1" or "v2")
    Returns:
        None
    """
    global simulation_version, PROMPT_ROOT, MODEL_FAMILY_OVERRIDE
    # The retained v0/v2 prompt_v5 trees are byte-identical. The research
    # checkout has no simulation_v1_5 directory; its teacher-forcing rule is
    # implemented below, independently of the shared prompt wording.
    simulation_version = "simulation_v2" if version == "v1_5" else f"simulation_{version}"
    if prompt_root is not None:
        PROMPT_ROOT = Path(prompt_root)
    MODEL_FAMILY_OVERRIDE = (model_family, "instruct") if model_family else None
    prompt_template_root = str(PROMPT_ROOT)
    if input_path is None or output_root is None:
        raise ValueError("input_path and output_root are required")
    input_path = Path(input_path)
    output_dir = str(Path(output_root))

    print(f"Simulating with model {model_name} on data prefix {data_prefix} version {version}.", flush=True)
    print("is dpo breadth:", is_dpo_breadth, flush=True)
    
    if api_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment")
    if api_type == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Set GEMINI_API_KEY in the environment")
    if api_type not in ("openai", "gemini", "huggingface"):
        raise ValueError("Unsupported API type")
    random.seed(seed)

    if topic is None:
        topic = re.search(r'\d{8}_\d{6}_(.*)_.{26}', data_prefix).group(1).replace('_', ' ')
        topic = re.sub(r' +', ' ', topic)
    
    user_data = pd.read_csv(input_path, dtype={"worker_id": str, "empirica_id": str, "sender_id": str, "recipient_id": str})
    # user_data = concatenate_messages(user_data) # Handled in concat_with_pairs

    log_path = os.path.join(
        os.path.join(output_dir, "logs", version),
        data_prefix, Agent.get_model_archive_name(model_name))
    Path(log_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "simulation", data_prefix, Agent.get_model_archive_name(model_name))).mkdir(parents=True, exist_ok=True)
    if os.path.exists(os.path.join(log_path, "conversation_log.txt")):
        os.remove(os.path.join(log_path, "conversation_log.txt"))
    if os.path.exists(os.path.join(log_path, "conversation_log_memory.txt")):
        os.remove(os.path.join(log_path, "conversation_log_memory.txt"))
    if os.path.exists(os.path.join(log_path, "WARN.txt")):
        os.remove(os.path.join(log_path, "WARN.txt"))

    player_column_name = get_sender_id_column_name(user_data)
    user_data = fix_names(user_data, player_column_name)
    players = user_data[player_column_name].dropna().unique()
    demographic_backgrounds = get_demographic_background(user_data, api_type, model_name, log_path, player_column_name)

    # Get the initial opinions text and likert scale
    initial_opinions_verbal = {}
    initial_opinions_likert = {}
    for player in players:
        initial_opinions_verbal[player], initial_opinions_likert[player] = get_initial_opinion(user_data, player, player_column_name)
    initial_opinions = {player: get_initial_opinion(user_data, player, player_column_name) for player in players}
    invalid_players = get_invalid_players(user_data)

    agents = []
    for i in range(len(players)):
        persona = demographic_backgrounds[players[i]]
        initial_opinion_verbal = initial_opinions_verbal[players[i]]
        initial_opinion_likert = initial_opinions_likert[players[i]]
        agent = Agent(i+1, players[i], players.tolist(), persona, initial_opinion_verbal, initial_opinion_likert, api_type, model_name, temperature, seed, max_tokens, prompt_template_root, topic, log_path, model, tokenizer, endpoint_url=endpoint_url, is_dpo_breadth=is_dpo_breadth)
        agents.append(agent)
        with open(agent.conversation_log_memory_path, "w") as f:
            f.write(f"Memory Log for {agent.agent_name} (Agent ID: {agent.agent_id})\n\n")

    # user_data = concat_with_pairs(user_data, agents)
    conversation_output = {}
    tweet_dict = {}
    for round in range(len(players) - 1):
        pairs = get_pairs(user_data, round + 1, agents)
        tweets = {agent: get_tweet(user_data, agent, round + 1) for agent in players}
        # Dict of dict of list of tuples
        conversation_output[f'round_{round + 1}'] = {}

        for pair in pairs:
            agent1, agent2 = pair
            conversation_history = get_conversation_history(user_data, agent1.agent_name, agent2.agent_name, player_column_name)
            # Skip only after informing about changing round
            if len(conversation_history) == 0:
                print(f"[WARN] No conversation history found for player {agent1.agent_name} and {agent2.agent_name} in round {round + 1}")
                continue
            if agent1.agent_name in invalid_players or agent2.agent_name in invalid_players:
                print(f"[WARN] Invalid player {agent1.agent_name} or {agent2.agent_name} in round {round + 1}")
                continue
            if round != 0:
                change_round_agent1 = True
                if agent1.agent_name in invalid_players:
                    change_round_agent1 = False
                else:
                    try:
                        previous_agent_agent1 = get_previous_agent(user_data, agent1.agent_name, round)
                        if previous_agent_agent1 in invalid_players:
                            if round == 1:
                                change_round_agent1 = False
                            else:
                                previous_agent_agent1 = get_previous_agent(user_data, agent1.agent_name, round-1) # Only for round 3
                    except Exception as e:
                        print(f"Exception: {e}")
                        previous_agent_agent1 = ""
                if change_round_agent1:
                    # TODO ablation v2
                    # pass
                    agent1.change_round(previous_agent_agent1, agent2.agent_name)
                
                change_round_agent2 = True
                if agent2.agent_name in invalid_players:
                    change_round_agent2 = False
                else:
                    try:
                        previous_agent_agent2 = get_previous_agent(user_data, agent2.agent_name, round)
                        if previous_agent_agent2 in invalid_players:
                            if round == 1:
                                change_round_agent2 = False
                            else:
                                previous_agent_agent2 = get_previous_agent(user_data, agent2.agent_name, round-1) # Only for round 3
                    except Exception as e:
                        print(f"Inside previous_agent_agent2. Round: {round}, Current Agent: {agent2.agent_name}")
                        print(f"Exception: {e}")
                        previous_agent_agent2 = ""
                if change_round_agent2:
                    # TODO ablation v2
                    # pass
                    agent2.change_round(previous_agent_agent2, agent1.agent_name)

            agent1.inform_about_conversation(agent2.agent_name)
            agent2.inform_about_conversation(agent1.agent_name)
            
            if version in ("v0", "v1_5") and round != 0:
                agent1_round_tweet, agent1_round_tweet_unfiltered = agent1.generate_tweet(agent2.agent_name)
                agent2_round_tweet, agent2_round_tweet_unfiltered = agent2.generate_tweet(agent1.agent_name)

                tweet_dict[tweets[agent1.agent_name]] = (agent1_round_tweet, agent1_round_tweet_unfiltered)
                tweet_dict[tweets[agent2.agent_name]] = (agent2_round_tweet, agent2_round_tweet_unfiltered)

                agent1.add_to_memory_tweet(agent1_round_tweet, agent2_round_tweet, agent2.agent_name)
                event_order = get_event_order(user_data, "tweet", tweets[agent1.agent_name])
                user_data = write_step_memory(user_data, event_order, agent1, "tweet")

                agent2.add_to_memory_tweet(agent2_round_tweet, agent1_round_tweet, agent1.agent_name)
                event_order = get_event_order(user_data, "tweet", tweets[agent2.agent_name])
                user_data = write_step_memory(user_data, event_order, agent2, "tweet")

            else:
                agent1.add_to_memory_tweet(tweets[agent1.agent_name], tweets[agent2.agent_name], agent2.agent_name)
                try:
                    event_order = get_event_order(user_data, "tweet", tweets[agent1.agent_name])
                    user_data = write_step_memory(user_data, event_order, agent1, "tweet")
                except:
                    print(f"[WARN] No tweet found for player {agent1.agent_name} in round {round + 1}")
                agent2.add_to_memory_tweet(tweets[agent2.agent_name], tweets[agent1.agent_name], agent1.agent_name)
                try:
                    event_order = get_event_order(user_data, "tweet", tweets[agent2.agent_name])
                    user_data = write_step_memory(user_data, event_order, agent2, "tweet")
                except:
                    print(f"[WARN] No tweet found for player {agent2.agent_name} in round {round + 1}")
            conversation_output[f'round_{round + 1}'][(agent1.agent_name, agent2.agent_name)] = []

            for message in conversation_history:
                event_order_sent = get_event_order(user_data, "message_sent", message[1])
                if message[0] == agent1.agent_name:
                    agent1_output, agent1_output_unfiltered, agent1_memory = agent1.generate_response(agent2.agent_name)
                    user_data = write_step_memory(user_data, event_order_sent, agent1, "message_sent", agent1_memory)
                    conversation_output[f'round_{round + 1}'][(agent1.agent_name, agent2.agent_name)].append((agent1.agent_name, agent1_output, agent1_output_unfiltered))
                    if version in ("v1_5", "v2"):
                        agent1.add_response_to_memory(message[1], "write", agent2.agent_name)
                        agent2.add_response_to_memory(message[1], "read", agent1.agent_name)
                        # user_data = write_step_memory(user_data, event_order_sent, agent1, "message_sent")
                        # user_data = write_step_memory(user_data, event_order_recieved, agent2, "message_recieved")
                    else:
                        agent1.add_response_to_memory(agent1_output, "write", agent2.agent_name)
                        agent2.add_response_to_memory(agent1_output, "read", agent1.agent_name)
                        # user_data = write_step_memory(user_data, event_order_sent, agent1, "message_sent")
                        # user_data = write_step_memory(user_data, event_order_recieved, agent2, "message_recieved")
                else:
                    agent2_output, agent2_output_unfiltered, agent2_memory = agent2.generate_response(agent1.agent_name)
                    user_data = write_step_memory(user_data, event_order_sent, agent2, "message_sent", agent2_memory)
                    conversation_output[f'round_{round + 1}'][(agent1.agent_name, agent2.agent_name)].append((agent2.agent_name, agent2_output, agent2_output_unfiltered))
                    if version in ("v1_5", "v2"):
                        agent2.add_response_to_memory(message[1], "write", agent1.agent_name)
                        agent1.add_response_to_memory(message[1], "read", agent2.agent_name)
                        # user_data = write_step_memory(user_data, event_order_sent, agent2, "message_sent")
                        # user_data = write_step_memory(user_data, event_order_recieved, agent1, "message_recieved")
                    else:
                        agent2.add_response_to_memory(agent2_output, "write", agent1.agent_name)
                        agent1.add_response_to_memory(agent2_output, "read", agent2.agent_name)
                        # user_data = write_step_memory(user_data, event_order_sent, agent2, "message_sent")
                        # user_data = write_step_memory(user_data, event_order_recieved, agent1, "message_recieved")

            # TODO ablation v2
            # Clear user messages from both agents' memory after completing the round
            # agent1.history.clear_user_messages()
            # agent2.history.clear_user_messages()

    post_opinion_dict = {}
    agreement_dict = {}
    for agent in agents:
        if agent.agent_name in invalid_players:
            print(f"[WARN] Invalid player {agent.agent_name} in round {round + 1}. Skipping Post Opinion.")
            continue
        post_opinion_dict[agent] = agent.generate_post_opinion(is_likert_scale=False)
        agreement_text, _ = agent.generate_post_opinion(is_likert_scale=True)
        agreement_dict[agent] = agreement_text

    # Log agent histories
    for agent in agents:
        agent_history = asyncio.run(agent.history.aget_messages())
        with open(os.path.join(log_path, f"{agent.agent_name}_history.txt"), "w") as log_file:
            for message in agent_history:
                log_file.write(f"{message.type}: {message.content}\n")
                log_file.write("-" * 50 + "\n")
        print(f"{agent.agent_name}'s conversation history has been logged to {agent.agent_name}_history.txt")

    # Write results
    user_data = write_conversation_history_to_dataframe(conversation_output, user_data, version=version, data_prefix=data_prefix)
    user_data = post_processing(user_data, version=version, invalid_players=invalid_players)
    user_data = write_post_opinions_to_dataframe(post_opinion_dict, agreement_dict, user_data, player_column_name)
    user_data = write_tweets_to_dataframe(tweet_dict, user_data)
    
    # TODO change name for normal/ablation
    output_filename = f"simulation-{version}.csv"
    user_data.to_csv(os.path.join(output_dir, "simulation", data_prefix, Agent.get_model_archive_name(model_name), output_filename), index=False)
    print("length of data written to csv:")
    print(len(user_data))

    result = {
        "api_type": api_type,
        "model_name": model_name,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "version": version
    }
    with open(os.path.join(output_dir, "simulation", data_prefix, Agent.get_model_archive_name(model_name), output_filename.replace(".csv", ".json")), "w") as f:
        json.dump(result, f, indent=4)


