# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import copy
import json
import logging
import time
from collections import defaultdict
from typing import Any, Optional, Union

from ... import OpenAIWrapper
from ...llm_config import LLMConfig
from ...oai.openai_utils import create_gpt_assistant, retrieve_assistants_by_name, update_gpt_assistant, OAI_PRICE1K
from ...runtime_logging import log_new_agent, logging_enabled
from ..agent import Agent
from ..assistant_agent import AssistantAgent, ConversableAgent
from ...cmbagent_utils import cmbagent_debug, file_search_max_num_results, cmbagent_disable_display
from ...agentchat.conversable_agent import UpdateSystemMessage

import re
import sys
import pandas as pd
from IPython.display import display

logger = logging.getLogger(__name__)


class GPTAssistantAgent(ConversableAgent):
    """An experimental AG2 agent class that leverages the OpenAI Assistant API for conversational capabilities.
    This agent is unique in its reliance on the OpenAI Assistant for state management, differing from other agents like ConversableAgent.
    """

    DEFAULT_MODEL_NAME = "gpt-4o-mini"

    def __init__(
        self,
        name="GPT Assistant",
        instructions: Optional[str] = None,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], bool]] = None,
        assistant_config: Optional[dict[str, Any]] = None,
        overwrite_instructions: bool = False,
        overwrite_tools: bool = False,
        **kwargs: Any,
    ):
        """Args:
        name (str): name of the agent. It will be used to find the existing assistant by name. Please remember to delete an old assistant with the same name if you intend to create a new assistant with the same name.
        instructions (str): instructions for the OpenAI assistant configuration.
        When instructions is not None, the system message of the agent will be
        set to the provided instructions and used in the assistant run, irrespective
        of the overwrite_instructions flag. But when instructions is None,
        and the assistant does not exist, the system message will be set to
        AssistantAgent.DEFAULT_SYSTEM_MESSAGE. If the assistant exists, the
        system message will be set to the existing assistant instructions.
        llm_config (LLMConfig or dict or False): llm inference configuration.
            - model: Model to use for the assistant (gpt-4-1106-preview, gpt-3.5-turbo-1106).
        assistant_config
            - assistant_id: ID of the assistant to use. If None, a new assistant will be created.
            - check_every_ms: check thread run status interval
            - tools: Give Assistants access to OpenAI-hosted tools like Code Interpreter and Knowledge Retrieval,
                    or build your own tools using Function calling. ref https://platform.openai.com/docs/assistants/tools
            - file_ids: (Deprecated) files used by retrieval in run. It is Deprecated, use tool_resources instead. https://platform.openai.com/docs/assistants/migration/what-has-changed.
            - tool_resources: A set of resources that are used by the assistant's tools. The resources are specific to the type of tool.
        overwrite_instructions (bool): whether to overwrite the instructions of an existing assistant. This parameter is in effect only when assistant_id is specified in llm_config.
        overwrite_tools (bool): whether to overwrite the tools of an existing assistant. This parameter is in effect only when assistant_id is specified in llm_config.
        kwargs (dict): Additional configuration options for the agent.
            - verbose (bool): If set to True, enables more detailed output from the assistant thread.
            - Other kwargs: Except verbose, others are passed directly to ConversableAgent.
        """
        self._verbose = kwargs.pop("verbose", False)
        openai_client_cfg, openai_assistant_cfg = self._process_assistant_config(llm_config, assistant_config)
        ## cmbagent debug print: 
        # print('in gpt_assistant_agent.py openai_client_cfg: ', openai_client_cfg)
        # print('in gpt_assistant_agent.py openai_assistant_cfg: ', openai_assistant_cfg)

        super().__init__(
            name=name, 
            # system_message=instructions, 
            update_agent_state_before_reply=[UpdateSystemMessage(instructions),], ## added for cmbagent
            human_input_mode="NEVER", 
            llm_config=openai_client_cfg, 
            **kwargs
        )
        if logging_enabled():
            log_new_agent(self, locals())

        # GPTAssistantAgent's azure_deployment param may cause NotFoundError (404) in client.beta.assistants.list()
        # See: https://github.com/microsoft/autogen/pull/1721
        model_name = self.DEFAULT_MODEL_NAME
        if openai_client_cfg.get("config_list") is not None and len(openai_client_cfg["config_list"]) > 0:
            model_name = openai_client_cfg["config_list"][0].pop("model", self.DEFAULT_MODEL_NAME)
        else:
            model_name = openai_client_cfg.pop("model", self.DEFAULT_MODEL_NAME)

        logger.warning("OpenAI client config of GPTAssistantAgent(%s) - model: %s", name, model_name)

        oai_wrapper = OpenAIWrapper(**openai_client_cfg)
        if len(oai_wrapper._clients) > 1:
            logger.warning("GPT Assistant only supports one OpenAI client. Using the first client in the list.")

        self._openai_client = oai_wrapper._clients[0]._oai_client
        openai_assistant_id = openai_assistant_cfg.get("assistant_id", None)
        if openai_assistant_id is None:
            # try to find assistant by name first
            candidate_assistants = retrieve_assistants_by_name(self._openai_client, name)
            if len(candidate_assistants) > 0:
                # Filter out candidates with the same name but different instructions, file IDs, and function names.
                candidate_assistants = self.find_matching_assistant(
                    candidate_assistants,
                    instructions,
                    openai_assistant_cfg.get("tools", []),
                )

            if len(candidate_assistants) == 0:
                logger.warning("No matching assistant found, creating a new assistant")
                # create a new assistant
                if instructions is None:
                    logger.warning(
                        "No instructions were provided for new assistant. Using default instructions from AssistantAgent.DEFAULT_SYSTEM_MESSAGE."
                    )
                    instructions = AssistantAgent.DEFAULT_SYSTEM_MESSAGE
                self._openai_assistant = create_gpt_assistant(
                    self._openai_client,
                    name=name,
                    instructions=instructions,
                    model=model_name,
                    assistant_config=openai_assistant_cfg,
                )
            else:
                logger.warning(
                    "Matching assistant found, using the first matching assistant: %s",
                    candidate_assistants[0].__dict__,
                )
                self._openai_assistant = candidate_assistants[0]
        else:
            # retrieve an existing assistant
            self._openai_assistant = self._openai_client.beta.assistants.retrieve(openai_assistant_id)
            ## cmbagent debug print: 
#             print('in gpt_assistant_agent.py self._openai_assistant.response_format: ', self._openai_assistant.response_format)
#             if self._openai_assistant.response_format != openai_client_cfg['config_list'][0]['response_format']:
#                 print("""
# The response format of the found assistant does not match the one provided in the config. 
# You must delete the assistant and create a new one with the correct response format.
# Terminating the program.""")
#                 sys.exit()
            # if no instructions are provided, set the instructions to the existing instructions
            if instructions is None:
                logger.warning(
                    "No instructions were provided for given assistant. Using existing instructions from assistant API."
                )
                instructions = self.get_assistant_instructions()
            elif overwrite_instructions is True:
                logger.warning(
                    "overwrite_instructions is True. Provided instructions will be used and will modify the assistant in the API"
                )
                self._openai_assistant = update_gpt_assistant(
                    self._openai_client,
                    assistant_id=openai_assistant_id,
                    assistant_config={
                        "instructions": instructions,
                    },
                )
            else:
                logger.warning(
                    "overwrite_instructions is False. Provided instructions will be used without permanently modifying the assistant in the API."
                )

            # Check if tools are specified in assistant_config
            specified_tools = openai_assistant_cfg.get("tools", None)

            if specified_tools is None:
                # Check if the current assistant has tools defined
                if self._openai_assistant.tools:
                    logger.warning(
                        "No tools were provided for given assistant. Using existing tools from assistant API."
                    )
                else:
                    logger.info(
                        "No tools were provided for the assistant, and the assistant currently has no tools set."
                    )
            elif overwrite_tools is True:
                # Tools are specified and overwrite_tools is True; update the assistant's tools
                logger.warning(
                    "overwrite_tools is True. Provided tools will be used and will modify the assistant in the API"
                )
                ## cmbagent debug print: 
                if cmbagent_debug:
                    print('in gpt_assistant_agent.py specified_tools: ', specified_tools)
                    print('in gpt_assistant_agent.py openai_assistant_cfg: ', openai_assistant_cfg)
                # this seems to not be called as of 18 dec 2024
                # print("in gpt_assistant_agent.py specified_tools: ", specified_tools)
                self._openai_assistant = update_gpt_assistant(
                    self._openai_client,
                    assistant_id=openai_assistant_id,
                    assistant_config={
                        "tools": specified_tools,
                        # "tool_choice": {"type": "function", "function": {"name": "file_search"}}, ## force required tool call for file_search doesnt work here
                        "tool_resources": openai_assistant_cfg.get("tool_resources", None),
                        "temperature": openai_assistant_cfg.get("temperature",None),
                        "top_p": openai_assistant_cfg.get("top_p",None),
                        # "check_every_ms":500, ## cmbagent added this to check every 500ms
                    },
                )
                ## cmbagent debug print: 
                # print('in gpt_assistant_agent.py assistant_config: ', openai_assistant_cfg)
            else:
                # Tools are specified but overwrite_tools is False; do not update the assistant's tools
                logger.warning("overwrite_tools is False. Using existing tools from assistant API.")

        # relay error message if assistant not set-up properly
        if 'error' in self._openai_assistant:
            ## cmbagent debug print: 
            if cmbagent_debug:
                print('assistant not set-up properly, relaying error message: ', self._openai_assistant)
            self._assistant_error = self._openai_assistant

        else:
            self.update_system_message(self._openai_assistant.instructions)
            # lazily create threads
            self._openai_threads = {}
            self._unread_index = defaultdict(int)
            self.register_reply([Agent, None], GPTAssistantAgent._invoke_assistant, position=2)
            self._assistant_error = None

        # set up dictionary attribute for cost summary
        self.cost_dict = {'Agent': [], 'Cost': [], 'Prompt Tokens': [], 'Completion Tokens': [], 'Total Tokens': []}

    def _invoke_assistant(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """Invokes the OpenAI assistant to generate a reply based on the given messages.

        Args:
            messages: A list of messages in the conversation history with the sender.
            sender: The agent instance that sent the message.
            config: Optional configuration for message processing.

        Returns:
            A tuple containing a boolean indicating success and the assistant's reply.
        """
        if messages is None:
            messages = self._oai_messages[sender]
        unread_index = self._unread_index[sender] or 0
        pending_messages = messages[unread_index:]

        # Check and initiate a new thread if necessary
        if self._openai_threads.get(sender, None) is None:
            self._openai_threads[sender] = self._openai_client.beta.threads.create(
                messages=[],
            )
        assistant_thread = self._openai_threads[sender]
        # Process each unread message
        for message in pending_messages:
            ## cmbagent debug print: 
            # print('in gpt_assistant_agent.py message in pending_messages: ', message)
            # print('in gpt assistant agent.py:', self.llm_config)
            if message["content"].strip() == "":
                continue
            # Convert message roles to 'user' or 'assistant', by calling _map_role_for_api, to comply with OpenAI API spec
            api_role = self._map_role_for_api(message["role"])
            ## cmbagent debug print: 
            if cmbagent_debug:
                print('in gpt_assistant_agent.py api_role: ', api_role)
            self._openai_client.beta.threads.messages.create(
                thread_id=assistant_thread.id,
                content=message["content"],
                role=api_role,
            )

        ## cmbagent debug print: 
        if cmbagent_debug:
            print('in gpt_assistant_agent.py running with system message: ', self.system_message)
            print('in gpt_assistant_agent.py tool_resources: ', self._openai_assistant.tool_resources)
        # Create a new run to get responses from the assistant
        run = self._openai_client.beta.threads.runs.create(
            thread_id=assistant_thread.id,
            assistant_id=self._openai_assistant.id,
            # pass the latest system message as instructions
            instructions=self.system_message,
            # tool_resources=self._openai_assistant.tool_resources, ## doesnt work
            tools=[{ ## cmbagent added this
                'type': 'file_search', ## cmbagent added this
                'file_search': {'max_num_results': file_search_max_num_results} ## cmbagent added this
            }], ## cmbagent added this
            tool_choice={"type": "file_search", "function": {"name": "file_search"}} ## cmbagent added this to force tool call
        )


        ## cmbagent debug print: 
        # print('in gpt_assistant_agent.py run done. calling _get_run_response')

        run_response_messages = self._get_run_response(assistant_thread, run)

        ## cmbagent debug print: 
        if cmbagent_debug:
        # if cmbagent_debug:
            print('in gpt_assistant_agent.py run: ', run)
            # print('in gpt_assistant_agent.py run_response_messages: ', run_response_messages)
            print('in gpt_assistant_agent.py run_response_messages json depth 3: ', json.dumps(run_response_messages, indent=3))
            print('debug here if you want to print the chunks etc')
            # print('in gpt_assistant_agent.py system_message: ', self.system_message)
            run_steps = self._openai_client.beta.threads.runs.steps.list(
                thread_id=assistant_thread.id,
                run_id=run.id
            )
            # print('in gpt_assistant_agent.py run_steps: ', run_steps)
            try:
                print('in gpt_assistant_agent.py run_steps json depth 3: ', json.dumps(run_steps, indent=3))
            except:
                print('in gpt_assistant_agent.py run_steps: ', run_steps)

            i = 0
            for step in run_steps.data:
                print("i: ", i)
                try:

                    retrieved_step = self._openai_client.beta.threads.runs.steps.retrieve(
                        thread_id=step.thread_id,
                        run_id=run.id,
                        step_id=step.id,
                        include=["step_details.tool_calls[*].file_search.results[*].content"]
                    )
                    r = 0
                    for result in retrieved_step.step_details.tool_calls[0].file_search.results:
                        print("\n\nr: ", r)
                        print("\n\nresult: ", result)
                        r += 1

                except:
                    print("step.step_details.tool_calls: None")
                print("\n\nstep done\n\n")
                i += 1
            # sys.exit()

        assert len(run_response_messages) > 0, "No response from the assistant."

        response = {
            "role": run_response_messages[-1]["role"],
            "content": "",
        }
        for message in run_response_messages:
            # just logging or do something with the intermediate messages?
            # if current response is not empty and there is more, append new lines
            if len(response["content"]) > 0:
                response["content"] += "\n\n"
            response["content"] += message["content"]

        self._unread_index[sender] = len(self._oai_messages[sender]) + 1
        return True, response

    def _map_role_for_api(self, role: str) -> str:
        """Maps internal message roles to the roles expected by the OpenAI Assistant API.

        Args:
            role (str): The role from the internal message.

        Returns:
            str: The mapped role suitable for the API.
        """
        if role in ["function", "tool"]:
            return "assistant"
        elif role == "system":
            return "system"
        elif role == "user":
            return "user"
        elif role == "assistant":
            return "assistant"
        else:
            # Default to 'assistant' for any other roles not recognized by the API
            return "assistant"

    def remove_numerical_references(self,text):
        # Remove numerical references of format [0], [1], etc.
        cleaned_text = re.sub(r'\[\d+\]', '', text)
        return cleaned_text
    
    def cost(self, run):
        """Calculate the cost of the run."""
        model = run.model
        if model not in OAI_PRICE1K:
            # log warning that the model is not found
            logger.warning(
                f'Model {model} is not found. The cost will be 0. In your config_list, add field {{"price" : [prompt_price_per_1k, completion_token_price_per_1k]}} for customized pricing.'
            )
            return 0

        n_input_tokens = run.usage.prompt_tokens if run.usage is not None else 0  # type: ignore [union-attr]
        n_output_tokens = run.usage.completion_tokens if run.usage is not None else 0  # type: ignore [union-attr]
        if n_output_tokens is None:
            n_output_tokens = 0
        tmp_price1K = OAI_PRICE1K[model]
        # First value is input token rate, second value is output token rate
        if isinstance(tmp_price1K, tuple):
            return (tmp_price1K[0] * n_input_tokens + tmp_price1K[1] * n_output_tokens) / 1000  # type: ignore [no-any-return]
        return tmp_price1K * (n_input_tokens + n_output_tokens) / 1000  # type: ignore [operator]

    def print_usage_summary(self, tokens_dict):
        """
        Prints a summary of token usage and costs for the current run.

        Args:
            tokens_dict (dict): Dictionary containing token usage information with keys:
                - model: Name of the model used
                - prompt_tokens: Number of prompt tokens used
                - completion_tokens: Number of completion tokens used  
                - total_tokens: Total number of tokens used
                - cost: Total cost of the run

        Updates the agent's cost_dict attribute with the usage information and prints
        a formatted summary to stdout.
        """
        # Extracting values from the dictionary
        model = tokens_dict["model"]
        prompt_tokens = tokens_dict["prompt_tokens"]
        completion_tokens = tokens_dict["completion_tokens"]
        total_tokens = tokens_dict["total_tokens"]
        cost = tokens_dict["cost"]
        

        # Restructure tokens_dict to create a DataFrame
        df = pd.DataFrame([{
            "Model": model,
            "Cost": f"{cost:.5f}",
            "Prompt Tokens": prompt_tokens,
            "Completion Tokens": completion_tokens,
            "Total Tokens": total_tokens,
        }])
        if not cmbagent_disable_display:
            display(df.style.hide(axis="index"))

        # Update dictionary containing all costs
        self.cost_dict['Agent'].append(self.name.replace('_agent', ''))
        self.cost_dict['Cost'].append(cost) 
        self.cost_dict['Prompt Tokens'].append(prompt_tokens)
        self.cost_dict['Completion Tokens'].append(completion_tokens)
        self.cost_dict['Total Tokens'].append(total_tokens)


    def _get_run_response(self, thread, run):
        """Waits for and processes the response of a run from the OpenAI assistant.

        Args:
            thread: The thread object initiated with the OpenAI assistant.
            run: The run object initiated with the OpenAI assistant.

        Returns:
            Updated run object, status of the run, and response messages.
        """
        ## cmbagent debug print: 
        # print('in gpt_assistant_agent.py _get_run_response')
        while True:
            run = self._wait_for_run(run.id, thread.id)
            if run.status == "completed":
                response_messages = self._openai_client.beta.threads.messages.list(thread.id, order="asc")

                # register cost 
                prompt_tokens = run.usage.prompt_tokens
                completion_tokens = run.usage.completion_tokens
                total_tokens = run.usage.total_tokens

                cost = self.cost(run)
                tokens_dict = {
                    "model": run.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
                self.print_usage_summary(tokens_dict)

                ## cmbagent debug print check rag steps: 
                # print("run.id: ", run.id)
                # run_steps = self._openai_client.beta.threads.runs.steps.list(
                #     thread_id=thread.id,
                #     run_id=run.id
                # )
                # # print("run_steps list: ", run_steps)
                # # from pprint import pprint
                # # pprint(run_steps, depth=3, width=120)

                # # print("run_steps.data: ", run_steps.data)
                # i = 0
                # for step in run_steps.data:
                #     print("i: ", i)
                #     # print("step: ", step)
                #     # print("step.step_details: ", step.step_details)
                #     # print("step.thread_id: ", step.thread_id)

                #     try:
                #         # print("step.step_details.tool_calls: ", step.step_details.tool_calls)

                #         retrieved_step = self._openai_client.beta.threads.runs.steps.retrieve(
                #             thread_id=step.thread_id,
                #             run_id=run.id,
                #             step_id=step.id,
                #             include=["step_details.tool_calls[*].file_search.results[*].content"]
                #         )
                #         # print("\n\nretrieved_step: ", retrieved_step)
                        
                #         # print("\n\nstep.step_details.tool_calls[0].file_search.results: ",retrieved_step.step_details.tool_calls[0].file_search.results)
                #         r = 0
                #         for result in retrieved_step.step_details.tool_calls[0].file_search.results:
                #             print("\n\nr: ", r)
                #             print("\n\nresult: ", result)
                #             # print("result.content: ", result.content)
                #             r += 1

                #     except:
                #         print("step.step_details.tool_calls: None")
                #     # print("step.step_details.tool_calls[0].file_search.results: ", step.step_details.tool_calls[0].file_search.results)
                #     print("\n\nstep done\n\n")
                #     i += 1





                new_messages = []
                for msg in response_messages:
                    if msg.run_id == run.id:
                        for content in msg.content:
                            if content.type == "text":
                                # Remove numerical references from the content
                                cleaned_content = self.remove_numerical_references(self._format_assistant_message(content.text))
                                new_messages.append(
                                    {"role": msg.role, 
                                     "content": cleaned_content}
                                )
                                # new_messages.append({
                                #     "role": msg.role,
                                #     "content": self._format_assistant_message(content.text),
                                # })
                            elif content.type == "image_file":
                                new_messages.append({
                                    "role": msg.role,
                                    "content": f"Received file id={content.image_file.file_id}",
                                })
                return new_messages
            elif run.status == "requires_action":
                print('in gpt_assistant_agent.py run.status == "requires_action"')
                actions = []
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    function = tool_call.function
                    tool_call_id = tool_call.id
                    is_exec_success, tool_response = self.execute_function(
                        function.dict(), call_id=tool_call_id, verbose=self._verbose
                    )
                    tool_response["metadata"] = {
                        "tool_call_id": tool_call.id,
                        "run_id": run.id,
                        "thread_id": thread.id,
                    }

                    logger.info(
                        "Intermediate executing(%s, Success: %s) : %s",
                        tool_response["name"],
                        is_exec_success,
                        tool_response["content"],
                    )
                    actions.append(tool_response)

                submit_tool_outputs = {
                    "tool_outputs": [
                        {"output": action["content"], "tool_call_id": action["metadata"]["tool_call_id"]}
                        for action in actions
                    ],
                    "run_id": run.id,
                    "thread_id": thread.id,
                }

                run = self._openai_client.beta.threads.runs.submit_tool_outputs(**submit_tool_outputs)
            else:
                run_info = json.dumps(run.dict(), indent=2)
                raise ValueError(f"Unexpected run status: {run.status}. Full run info:\n\n{run_info})")

    def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
        """Waits for a run to complete or reach a final state.

        Args:
            run_id: The ID of the run.
            thread_id: The ID of the thread associated with the run.

        Returns:
            The updated run object after completion or reaching a final state.
        """
        in_progress = True
        while in_progress:
            run = self._openai_client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                if cmbagent_debug:  
                    print('in gpt_assistant_agent.py _wait_for_run: in_progress')
                    print('in gpt_assistant_agent.py llm_config:  ', self.llm_config)
                time.sleep(self.llm_config.get("check_every_ms", 1000) / 1000)
        return run

    def _format_assistant_message(self, message_content):
        """Formats the assistant's message to include annotations and citations."""
        annotations = message_content.annotations
        citations = []

        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the text with a footnote
            message_content.value = message_content.value.replace(annotation.text, f" [{index}]")

            # Gather citations based on annotation attributes
            if file_citation := getattr(annotation, "file_citation", None):
                try:
                    cited_file = self._openai_client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}: {file_citation.quote}")
                except Exception as e:
                    logger.error(f"Error retrieving file citation: {e}")
            elif file_path := getattr(annotation, "file_path", None):
                try:
                    cited_file = self._openai_client.files.retrieve(file_path.file_id)
                    citations.append(f"[{index}] Click <here> to download {cited_file.filename}")
                except Exception as e:
                    logger.error(f"Error retrieving file citation: {e}")
                # Note: File download functionality not implemented above for brevity

        # Add footnotes to the end of the message before displaying to user
        message_content.value += "\n" + "\n".join(citations)
        return message_content.value

    def can_execute_function(self, name: str) -> bool:
        """Whether the agent can execute the function."""
        return False

    def reset(self):
        """Resets the agent, clearing any existing conversation thread and unread message indices."""
        super().reset()
        for thread in self._openai_threads.values():
            # Delete the existing thread to start fresh in the next conversation
            self._openai_client.beta.threads.delete(thread.id)
        self._openai_threads = {}
        # Clear the record of unread messages
        self._unread_index.clear()

    def clear_history(self, agent: Optional[Agent] = None):
        """Clear the chat history of the agent.

        Args:
            agent: the agent with whom the chat history to clear. If None, clear the chat history with all agents.
        """
        super().clear_history(agent)
        try:
            if self._openai_threads.get(agent, None) is not None:
                # Delete the existing thread to start fresh in the next conversation
                thread = self._openai_threads[agent]
            logger.info("Clearing thread %s", thread.id)
            self._openai_client.beta.threads.delete(thread.id)
            self._openai_threads.pop(agent)
            self._unread_index[agent] = 0
        except:
            pass
        
    def pretty_print_thread(self, thread):
        """Pretty print the thread."""
        if thread is None:
            print("No thread to print")
            return
        # NOTE: that list may not be in order, sorting by created_at is important
        messages = self._openai_client.beta.threads.messages.list(
            thread_id=thread.id,
        )
        messages = sorted(messages.data, key=lambda x: x.created_at)
        print("~~~~~~~THREAD CONTENTS~~~~~~~")
        for message in messages:
            content_types = [content.type for content in message.content]
            print(f"[{message.created_at}]", message.role, ": [", ", ".join(content_types), "]")
            for content in message.content:
                content_type = content.type
                if content_type == "text":
                    print(content.type, ": ", content.text.value)
                elif content_type == "image_file":
                    print(content.type, ": ", content.image_file.file_id)
                else:
                    print(content.type, ": ", content)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    @property
    def oai_threads(self) -> dict[Agent, Any]:
        """Return the threads of the agent."""
        return self._openai_threads

    @property
    def assistant_id(self):
        """Return the assistant id"""
        return self._openai_assistant.id

    @property
    def openai_client(self):
        return self._openai_client

    @property
    def openai_assistant(self):
        return self._openai_assistant

    def get_assistant_instructions(self):
        """Return the assistant instructions from OAI assistant API"""
        return self._openai_assistant.instructions

    def delete_assistant(self):
        """Delete the assistant from OAI assistant API"""
        logger.warning("Permanently deleting assistant...")
        self._openai_client.beta.assistants.delete(self.assistant_id)

    def find_matching_assistant(self, candidate_assistants, instructions, tools):
        """Find the matching assistant from a list of candidate assistants.
        Filter out candidates with the same name but different instructions, and function names.
        """
        matching_assistants = []

        # Preprocess the required tools for faster comparison
        required_tool_types = {
            "file_search" if tool.get("type") in ["retrieval", "file_search"] else tool.get("type") for tool in tools
        }

        required_function_names = {
            tool.get("function", {}).get("name")
            for tool in tools
            if tool.get("type") not in ["code_interpreter", "retrieval", "file_search"]
        }

        for assistant in candidate_assistants:
            # Check if instructions are similar
            if instructions and instructions != getattr(assistant, "instructions", None):
                logger.warning(
                    "instructions not match, skip assistant(%s): %s",
                    assistant.id,
                    getattr(assistant, "instructions", None),
                )
                continue

            # Preprocess the assistant's tools
            assistant_tool_types = {
                "file_search" if tool.type in ["retrieval", "file_search"] else tool.type for tool in assistant.tools
            }
            assistant_function_names = {tool.function.name for tool in assistant.tools if hasattr(tool, "function")}

            # Check if the tool types, function names match
            if required_tool_types != assistant_tool_types or required_function_names != assistant_function_names:
                logger.warning(
                    "tools not match, skip assistant(%s): tools %s, functions %s",
                    assistant.id,
                    assistant_tool_types,
                    assistant_function_names,
                )
                continue

            # Append assistant to matching list if all conditions are met
            matching_assistants.append(assistant)

        return matching_assistants

    def _process_assistant_config(self, llm_config, assistant_config):
        """Process the llm_config and assistant_config to extract the model name and assistant related configurations."""
        if llm_config is False:
            raise ValueError("llm_config=False is not supported for GPTAssistantAgent.")

        openai_client_cfg = {} if llm_config is None else copy.deepcopy(llm_config)

        openai_assistant_cfg = {} if assistant_config is None else copy.deepcopy(assistant_config)

        # Move the assistant related configurations to assistant_config
        # It's important to keep forward compatibility
        assistant_config_items = ["assistant_id", "tools", "file_ids", "tool_resources", "check_every_ms"]
        for item in assistant_config_items:
            if openai_client_cfg.get(item) is not None and openai_assistant_cfg.get(item) is None:
                openai_assistant_cfg[item] = openai_client_cfg[item]
            openai_client_cfg.pop(item, None)

        ## cmbagent debug print check: 
        if cmbagent_debug:
            print('in gpt_assistant_agent.py openai_client_cfg: ', openai_client_cfg)
            print('in gpt_assistant_agent.py openai_assistant_cfg: ', openai_assistant_cfg)
        # import sys; sys.exit()
        return openai_client_cfg, openai_assistant_cfg
