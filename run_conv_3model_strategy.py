import re
import time
from typing import List
from transformers import LogitsProcessor, LogitsProcessorList
import os
import argparse

import torch

from fastchat.model import load_model, get_conversation_template, add_model_args

from tqdm.auto import tqdm
import json
import copy
import openai
from pathlib import Path
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel


AGENT_PREFIX = "Dialogue History:"
AGENT_SUFFIX = "Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. Output Format: \nThought: <thought>\nResponse: <response>"
AGENT_SUFFIX_LLAMA = "Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. You should response as a real conversation.\n If you think user has explicitly mentioned the above intent, you should say \"Proceed to task oriented dialog agent.\""
USER_SUFFIX = "Imagine you are a real person. You are having chat with a online agent, so the repsonse do not include any expresssions. Remember, maintain a natural tone. Your response should be only your text resposne without any other expressions and emojis. Keep it as short as possible. Again, NO EMOJIS\n"
USER_SUFFIX_NEG_ALL = "You are not interested in FindAttraction, FindRestaurants, FindMovie, LookUpMusic, SearchHotel, FindEvents, if the agent ask any about one of them, donot ask for any recommendations and you should say, \"I don't want to talk about this. Let's talk about something else\". Note that you should be more firm."
USER_SUFFIX_NEG = "You are not interested in {intents}, if the agent ask any about one of them, donot ask for any recommendations and you should say, \"I don't want to talk about this. Let's talk about something else\". Note that you should be more firm."
# USER_SUFFIX_NEG = "You are not interested in 'FindAttraction', 'FindRestaurants'. Do not continue these topics."

SYSREM_PROMPT = "<|begin_of_text|> A chat  between a  curious  user  and  an  artificial  intelligence  assistant. USER: <value>"

SALESAGENT_SYSTEM_PROMPT = """\
You are a professional sales agent. Your objective is to identify the user's underlying intent, strategically guide the conversation toward areas of interest, and elicit the user's explicit expression of that intent.\
"""
SALESAGENT_PROMPT = """\
# Dialogue History:
{history}

# Strategy
According to statistics about the user, there is a high propability that the user is interested in these: {intents}
Rationale: {rationale}

# Internal Reflection:
Based on the above dialogue, your current reasoning is:
{thought}

If the current thought indicates the user has implicitly expressed interest in a specific topic, continue the conversation by following that topic naturally.
If the user has not shown a clear interest or has declined previous suggestions, pivot by using the strategy that best fits their likely occupation or background to guide the next part of the conversation.
Try to avoid repetition with the previous dialogue, and keep your response short, matching the user's length.
Now, continue the conversation with an appropriate response.

Output Format:
{
    "response": <response>
}\
"""

section_strategy = {
    'A': {
        'intents': 'FindRestaurants, FindAttraction',
        'rationale': 'These users often value relaxation and leisure experiences when off work.'
    },
    'J': {
        'intents': 'SearchHotel, FindRestaurants',
        'rationale': 'Tech workers frequently travel for work and value reliable accommodations and good dining options.'
    },
    'K': {
        'intents': 'SearchHotel, FindRestaurants',
        'rationale': 'These users may have business travel needs and typically prefer higher-end services.'
    },
    'P': {
        'intents': 'FindRestaurants, FindEvents',
        'rationale': 'Educators often enjoy social or cultural activities and group-friendly dining.'
    },
    'Q': {
        'intents': 'FindRestaurants, FindEvents',
        'rationale': 'These users often seek stress relief through leisure activities and social events.'
    },
    'R': {
        'intents': 'FindEvents, FindRestaurants',
        'rationale': 'Creatives are usually interested in events and venues that provide inspiration or entertainment, along with unique dining experiences.'
    }
}

occupation2section = {
    'farmer': 'A',
    'woodcutter': 'A',
    'fisherman': 'A',
    'horticulturist': 'A',
    'software_engineer': 'J',
    'cybersecurity_specialist': 'J',
    'data_scientist': 'J',
    'telecommunications_technician': 'J',
    'investment_analyst': 'K',
    'actuary': 'K',
    'insurance_claims_adjuster': 'K',
    'financial_advisor': 'K',
    'primary_school_teacher': 'P',
    'university_professor': 'P',
    'vocational_trainer': 'P',
    'special_education_teacher': 'P',
    'doctor': 'Q',
    'nurse': 'Q',
    'physical_therapist': 'Q',
    'psychologist': 'Q',
    'actor': 'R',
    'musician': 'R',
    'artist': 'R',
    'writer': 'R'
}

SALESAGENT_PROMPT_NO_STRATEGY = """\
# Dialogue History:
{history}

# Internal Reflection:
Based on the above dialogue, your current reasoning is:
{thought}

If the current thought indicates the user has implicitly expressed interest in a specific topic, continue the conversation by following that topic naturally.
If the user has not shown a clear interest or has declined previous suggestions, pivot to guide the next part of the conversation.
Try to avoid repetition with the previous dialogue, and keep your response short, matching the user's length.
Now, continue the conversation with an appropriate response.

Output Format:
{
    "response": <response>
}\
"""


class BlockThoughtAfterResponse(LogitsProcessor):
    def __init__(self, tokenizer, target_block="Thought:", trigger="Response:"):
        """
        :param tokenizer: Your model tokenizer.
        :param target_block: The string we want to block after the trigger occurs.
        :param trigger: The string that activates the blocking logic once it appears.
        """
        self.tokenizer = tokenizer
        # Encode the string that we want to block
        self.block_ids = tokenizer.encode(
            target_block, add_special_tokens=False)
        # Encode the trigger string that enables the blocking
        self.trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)

        self.blocking_on = False  # whether we've seen the trigger yet

    def __call__(self, input_ids, logits):
        """
        input_ids: shape [batch_size, sequence_length]
        logits: shape [batch_size, vocab_size] (raw next-token logits)
        """
        # 1) Check whether the trigger has appeared in the *already-generated* sequence
        if not self.blocking_on:
            # naive approach: scan the last portion of input_ids to see if trigger_ids is a suffix anywhere
            # (or do a substring search if you want more robust detection)
            for i in range(len(input_ids[0]) - len(self.trigger_ids) + 1):
                if all(
                    input_ids[0][i + j] == self.trigger_ids[j]
                    for j in range(len(self.trigger_ids))
                ):
                    self.blocking_on = True
                    break

        # 2) If the trigger has appeared, ban continuing with the sequence "Thought:"
        if self.blocking_on:
            # We need to handle partial matches. For example, if the model has started generating
            # the first token of "Thought:", we detect that partial match and set the next correct token(s) to -∞.

            # figure out how many tokens at the end of input_ids match the beginning of self.block_ids
            block_len = len(self.block_ids)
            # up to block_len-1 tokens can already be matched in the end
            max_overlap = min(block_len - 1, input_ids.shape[1])
            overlap = 0
            for k in range(max_overlap):
                # compare slice of input_ids (the last k+1 tokens) to the first k+1 tokens of block_ids
                if list(input_ids[0][-(k + 1):]) == self.block_ids[: (k + 1)]:
                    overlap = k + 1
                else:
                    break

            # If overlap < block_len, the next token that would complete the next character
            # in "Thought:" should be banned
            if overlap < block_len:
                next_token_to_block = self.block_ids[overlap]
                logits[0, next_token_to_block] = float("-inf")

        return logits


console = Console()


def print_with_tqdm(s, style=""):
    with console.capture() as capture:
        console.print(s, style=style)
    str_output = capture.get().strip()
    tqdm.write(str_output)


def get_response(
    api_base: str,
    model: str,
    messages: List[dict],
    top_p: float = 0.75,
    temperature: float = 0.4,
    max_retry: int = 1000000,
    repetition_penalty: float = 1.2,
    enable_thinking: bool = True,
):
    assert len(messages) > 0
    openai.api_key = "EMPTY"
    openai.api_base = api_base

    if model == 'Qwen/Qwen3-8B' and not enable_thinking:
        if top_p != 0.8 or temperature != 0.7:
            print_with_tqdm(
                f'Warning: not using recommended decode strategy.', style='red')
        messages = copy.deepcopy(messages)
        messages[-1]['content'] = messages[-1]['content'] + ' /no_think'

    while max_retry > 0:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                # stop=["\n"],
                request_timeout=10,
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            tqdm.write(f"Exception: {e}")
            max_retry -= 1
            time.sleep(3)

    import ipdb
    ipdb.set_trace()


def main(args):
    max_conv = 15

    # load persona from persona.json
    # if output file exitst, load it and continue:wq
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            personas = json.load(f)
    else:
        with open(args.input_file, "r") as f:
            personas = json.load(f)

    if all(len(p.get('conversations', [])) == max_conv for p in personas):
        print(f'{Path(args.input_file).name}: Done')
        exit()

    occupation = Path(args.output_file).stem
    section = occupation2section[occupation]

    # Load model
    agent_model, agent_tokenizer = load_model(
        args.agent_model,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        # load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    agent_model.generation_config.pad_token_id = agent_tokenizer.pad_token_id

    logits_processor = LogitsProcessorList()
    logits_processor.append(
        BlockThoughtAfterResponse(
            agent_tokenizer, target_block=" Thought:", trigger="Response:"
        )
    )

    for i, persona in enumerate(tqdm(personas,
                                     desc=Path(args.input_file).name,
                                     position=0)):
        if "conversations" in persona and \
                len(persona["conversations"]) == max_conv:
            continue

        print_with_tqdm(Panel(persona['persona']))

        if 'conversations' not in persona:
            persona["conversations"] = {}
        if 'terminate_reason' not in persona:
            persona["terminate_reason"] = []
        if 'num_turns' not in persona:
            persona["num_turns"] = []

        for current_conv in tqdm(range(max_conv), desc='Conversation', position=1):
            if current_conv < len(persona['terminate_reason']):
                continue
            print_with_tqdm(Rule(f'{current_conv + 1}'))
            num_turn = 0
            history = []
            history.extend(
                [
                    {
                        "role": "system",
                        "content": "You are not an AI. " + persona["persona"],
                    },
                    {
                        "role": "user",
                        "content": "Hi."
                    },
                ]
            )

            history[0]["content"] += "\n" + USER_SUFFIX

            while True:
                user_response = get_response(
                    api_base='http://your-vllm-server/v1',
                    model='Qwen/Qwen3-8B',
                    messages=history,
                    top_p=0.8,
                    temperature=0.7,
                    enable_thinking=False,
                )

                if "Assistant: " in user_response:
                    user_response = user_response.split("Assistant: ")[1]
                if "ASSISTANT: " in user_response:
                    user_response = user_response.split("ASSISTANT: ")[1]
                if "User: " in user_response:
                    user_response = user_response.split("User: ")[1]
                if "USER: " in user_response:
                    user_response = user_response.split("USER: ")[1]

                # Remove <think></think>
                matches = re.findall(r"<think>(.*?)</think>", user_response,
                                     flags=re.DOTALL)
                if matches and matches[0].strip() != '':
                    print_with_tqdm(
                        f'User is thinking: {matches[0].strip()}', style='red')

                user_response = re.sub(r"<think>.*?</think>", "",
                                       user_response, flags=re.DOTALL)
                user_response = user_response.strip()

                print_with_tqdm(f'User: {user_response}', style='yellow')

                history.append(
                    {
                        "role": "assistant",
                        "content": user_response,
                    }
                )

                history_string = ""
                for turn in history[2:]:
                    if turn["role"] == "assistant":
                        role = "User"
                    else:
                        role = "Agent"
                    history_string += role + ": " + turn["content"] + "\n"

                conv = get_conversation_template(args.model_path)
                conv.append_message(
                    conv.roles[0],
                    SYSREM_PROMPT.replace(
                        "<value>",
                        AGENT_PREFIX + history_string + AGENT_SUFFIX
                    )
                )
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                prompt = prompt.split("<|begin_of_text|>")[-1]
                prompt = "<|begin_of_text|> " + prompt
                prompt = prompt.replace("### Assistant:", "")
                # print(f"Prompt: {prompt}")

                # Run inference
                inputs = agent_tokenizer(
                    [prompt], return_tensors="pt").to(args.device)
                # print(f"Token len: {len(inputs['input_ids'][0])}")
                if len(inputs["input_ids"][0]) > 2048:
                    print_with_tqdm("Input length exceeds 2048, break",
                                    style='red')
                    persona["terminate_reason"].append(
                        "Input length exceeds 2048")
                    persona["num_turns"].append(num_turn)
                    break

                while True:
                    # if outputs does not include "Response" generate til it has
                    output_ids = agent_model.generate(
                        **inputs,
                        do_sample=True if args.agent_temperature > 1e-5 else False,
                        temperature=args.agent_temperature,
                        repetition_penalty=args.agent_repetition_penalty,
                        top_p=args.agent_topp,
                        top_k=args.agent_topk,
                        max_new_tokens=args.agent_max_new_tokens,
                        logits_processor=logits_processor,
                    )
                    if agent_model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(
                            inputs["input_ids"][0]):]
                    agent_response = agent_tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )

                    # Fix wrong format
                    if agent_response.count("Thought") == 2:
                        agent_response = agent_response.replace(
                            '\nThought:', '')
                    # if "Response" occur 1 time in agent_response break
                    if agent_response.count("Thought") == 1 and \
                            agent_response.count("Response") == 1:
                        break
                    else:
                        print_with_tqdm(f'Sales agent outputed in wrong format, retry:\n{agent_response}',
                                        style='red')

                thought, agent_response = agent_response.split("Response")
                thought = thought.split("Thought: ")[1].strip()
                agent_response = agent_response.strip()
                print_with_tqdm(f"Thought: {thought}", style='blue')

                if "The user has explicitly shown" in thought:
                    pass
                elif "bye" in agent_response.lower() or \
                    "goodbye" in agent_response.lower() or \
                        "good bye" in agent_response.lower():
                    # I don't know why </s> shows up
                    agent_response = agent_response.removesuffix(
                        '</s>').strip()
                else:
                    while True:
                        try:
                            agent_response = get_response(
                                api_base='http://your-vllm-server/v1',
                                model='mistralai/mistral-7b-instruct-v0.3',
                                messages=[
                                    {
                                        "role": "system",
                                        "content": SALESAGENT_SYSTEM_PROMPT,
                                    },
                                    {
                                        "role": "user",
                                        "content": SALESAGENT_PROMPT.replace(
                                            '{history}', history_string
                                        ).replace(
                                            '{thought}', thought
                                        ).replace(
                                            '{intents}', section_strategy[section]['intents']
                                        ).replace(
                                            '{rationale}', section_strategy[section]['rationale']
                                        )
                                    },
                                ],
                                top_p=0.75,
                                temperature=0.4,
                            )
                            agent_response = json.loads(
                                agent_response)['response']
                            break
                        except Exception as e:
                            print_with_tqdm(f'Error: {e}', style='red')

                print_with_tqdm(
                    f"Sales Agent: {agent_response}", style='green')

                history.append(
                    {
                        "role": "user",
                        "content": agent_response,
                        "thought": thought,
                    }
                )
                num_turn += 2

                if "The user has explicitly shown" in thought:
                    persona["terminate_reason"].append("Success")
                    persona["num_turns"].append(num_turn)
                    break

                if "bye" in agent_response.lower()\
                        or "goodbye" in agent_response.lower()\
                        or "good bye" in agent_response.lower():
                    persona["terminate_reason"].append("Conversation End")
                    persona["num_turns"].append(num_turn)
                    break

                if num_turn == args.max_turns:
                    tqdm.write("Reach max turns: break")
                    persona["terminate_reason"].append("Reach Max Turns")
                    persona["num_turns"].append(num_turn)
                    break
            persona["conversations"][f"conv_{current_conv}"] = copy.deepcopy(
                history)
            # clear cuda memory
            torch.cuda.empty_cache()
            with open(args.output_file, "w") as f:
                json.dump(personas, f, indent=4)


def arg_parser():
    # agent's param
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--agent_model",
        type=str,
        default="miulab/SalesBot2_CoT_lora_w_neg_wo_dup_chitchat_e10",
        help="agents model",
    )
    parser.add_argument(
        "--agent_temperature", type=float, default=0.5, help="agents temperature"
    )
    parser.add_argument("--agent_topk", type=int,
                        default=50, help="agents topk")
    parser.add_argument("--agent_topp", type=float,
                        default=1, help="agents topp")
    parser.add_argument(
        "--agent_max_new_tokens", type=int, default=200, help="agents max new tokens"
    )
    parser.add_argument(
        "--agent_repetition_penalty",
        type=float,
        default=1.0,
        help="agents repetition penalty",
    )
    parser.add_argument(
        "--agent_do_sample", type=bool, default=True, help="agents no sample"
    )
    # user's param
    parser.add_argument("--user_model", type=str,
                        default=None, help="users model")
    parser.add_argument(
        "--user_temperature", type=float, default=0.5, help="users temperature"
    )
    parser.add_argument("--user_topk", type=int, default=50, help="users topk")
    parser.add_argument("--user_topp", type=float,
                        default=1, help="users topp")
    parser.add_argument(
        "--user_max_new_tokens", type=int, default=100, help="users max new tokens"
    )
    parser.add_argument(
        "--user_repetition_penalty",
        type=float,
        default=1.0,
        help="users repetition penalty",
    )
    parser.add_argument(
        "--user_do_sample", type=bool, default=True, help="users no sample"
    )
    # common param
    parser.add_argument("--max_turns", type=int, default=20, help="max turns")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file", type=str,
                        default="persona_with_conv.json")
    parser.add_argument("--input_file", type=str, default="persona.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    main(args)
