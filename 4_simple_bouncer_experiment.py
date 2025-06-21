import os
# import httpx
# original_sync_send = httpx.Client.send

# def patched_sync_send(self, request, *args, **kwargs):
#     if str(request.url) == "https://openrouter.ai/api/v1/v1/messages":
#         request.url = httpx.URL("https://openrouter.ai/api/v1/chat/completions")
#         api_key = os.getenv("OPENROUTER_API_KEY")
#         request.headers["Authorization"] = f"Bearer {api_key}"
#     return original_sync_send(self, request, *args, **kwargs)

# httpx.Client.send = patched_sync_send
from textwrap import dedent
from openai import OpenAI, AzureOpenAI
from ollama import Client
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import json
import subprocess
import argparse
import tempfile
import pygit2
import shutil
from pathlib import Path
import instructor
# from anthropic import Anthropic

load_dotenv()
max_tokens = 4000
CACHE_DIR = Path("repo_cache")
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
access_token = os.getenv("GITHUB_TOKEN")
or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
# ant_client = Anthropic(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY")
# )
az_client = AzureOpenAI(  
    azure_endpoint=os.getenv("ENDPOINT_URL"),  
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2025-01-01-preview",
)
client = Client(
    host=os.getenv("OLLAMA_ENDPOINT")
)
anthropic_client = instructor.from_openai(or_client, mode=instructor.Mode.TOOLS)
# anthropic_reasoning_client = instructor.from_anthropic(ant_client, mode=instructor.Mode.ANTHROPIC_JSON)

model_list = [
    # {
    #     "client": or_client,
    #     "deployment": "meta-llama/llama-4-maverick:free"
    # },
    {
        "client": client,
        "deployment": "qwen3:32b-q8_0"
    },
    {
        "client": client,
        "deployment": "gemma3:27b-it-q8_0"
    },
    {
        "client": az_client,
        "deployment": "gpt-4.1"
    },
    {
        "client": az_client,
        "deployment": "o4-mini"
    },
    {
        "client": anthropic_client,
        "deployment": "anthropic/claude-3.7-sonnet"
    },
    # {
    #     "client": anthropic_reasoning_client,
    #     "deployment": "anthropic/claude-3.7-sonnet:thinking"
    # }
]

def preclone_repos(repos):
    for repo in tqdm(repos, desc="Pre-cloning repos"):
        dest = CACHE_DIR / repo.replace("/", "_")
        if not dest.exists():
            pygit2.clone_repository(
                f"https://github.com/{repo}.git",
                str(dest),
                # callbacks=GitRemoteCallbacks("x-access-token", access_token),
            )

def run_codex(
        prompt: str,
        model: str = "o4-mini",
        provider: str = "azure",
        reasoning: bool = "low",
        cwd: str = None,
        # approval_mode: str = "full-auto"
    ) -> dict:
    """
    Call Codex CLI non-interactively, return parsed JSON.
    """
    cmd = [
        "codex", "-q", "--json",
        # "--approval-mode", approval_mode,
        "--reasoning", reasoning,
        "--provider", provider,
        "--model", model,
        prompt
    ]
    if cwd is None:
        proc = subprocess.run(cmd,
                            capture_output=True,
                            text=True)
    else:
        proc = subprocess.run(cmd,
                            capture_output=True,
                            text=True,
                            cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex failed ({provider}/{model}): {proc.stderr.strip()}"
        )
    # convert each line of the output to a JSON object
    # and return the list of JSON objects
    trace = []
    for line in proc.stdout.splitlines():
        if line.strip() == "":
            continue
        try:
            _json = json.loads(line)
            trace.append(_json)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"codex output is not valid JSON ({provider}/{model}): {line}"
            )
    return trace

def get_response(client, deployment, system_prompt, user_prompt, response_format):
    if isinstance(client, instructor.Instructor):
        if "thinking" in deployment:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                # reasoning = {
                #     "max_tokens": max_tokens,
                # },
                # not using temperature here because reasoning models should not be run with greedy decoding
                temperature=1,
                response_model=response_format,
                thinking={"budget_tokens": max_tokens},
                
                extra_body={"provider": {"require_parameters": True}},
            )
        else:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0,
                response_model=response_format,
                extra_body={"provider": {"require_parameters": True}},
            )
        return resp
    elif isinstance(client, Client):
        # ollama client
        if "qwen" in deployment:
            # reasoning models should not be run with greedy decoding
            temperature = 0.6
        else:
            temperature = 0
        response = client.chat(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options = {
            "seed": 44,
            "temperature": temperature,
            "num_ctx": 32000
            },
            format=response_format.model_json_schema(),
        )
        return response_format.model_validate_json(response.message.content)
    else:
        if "mini" in deployment:
            # only temp 1 is supported for reasoning family models
            temperature = 1
            completion = client.beta.chat.completions.parse(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning_effort="low",
                max_completion_tokens=max_tokens,
                temperature=temperature,
                seed=44,
                response_format=response_format,
            )
        else:
            completion = client.beta.chat.completions.parse(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                temperature=0,
                seed=44,
                response_format=response_format,
            )
        return completion.choices[0].message.parsed
    # except Exception as e:
    #     print(f"Error: {e}")
    #     completion = client.chat.completions.create(
    #         model=deployment,
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt},
    #         ],
    #         max_tokens=800,
    #         temperature=0,
    #         seed=44,
    #     )
    #     return completion.choices[0].message.content

class InputBouncerResponse(BaseModel):
    explanation: str = Field(
        description="Critical analysis if issue description is well-specified enough for a meaningful attempt at a solution"
    )
    label: Literal[
        "WELL_SPECIFIED",
        "REASONABLY_SPECIFIED",
        "VAGUE",
        "IMPOSSIBLE_TO_SOLVE"
    ]

class OutputBouncerResponse(BaseModel):
    explanation: str = Field(
        description="Critical analysis if the patch correctly addresses the issue description"
    )
    label: Literal[
        "CORRECT_AND_PRECISE",
        "CORRECT_BUT_INCOMPLETE",
        "BROAD_MISSING_KEY_ASPECTS",
        "INCORRECT"
    ]

def input_bouncer(client, deployment, instance_id, issue_description, temp_dir=None):
    if client is None:
        _deployment_simple = "codex"
        _prompt_file_loc = "prompts/input_bouncer_codex.txt"
    else:
        _prompt_file_loc = "prompts/input_bouncer.txt"
        _deployment_simple = deployment.split("/")[-1].replace(":", "_")
    output_file = os.path.join(OUTPUT_DIR, f"input_bouncer_{_deployment_simple}.json")
    data = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
        if instance_id in data:
            print(f"Instance {instance_id} already exists in {output_file}. Skipping.")
            return
    with open(_prompt_file_loc, "r") as _prompt_file:
        system_prompt = _prompt_file.read().strip()
    user_prompt = dedent(
        """
        Is the following issue well-specified enough for a meaningful attempt at a solution?

        <ISSUE_DESCRIPTION>
        {issue_description}
        </ISSUE_DESCRIPTION>
        """
    ).format(issue_description=issue_description)
    _trace = None
    if client is None:
        _trace = run_codex(
            prompt=system_prompt + "\n" + user_prompt + "\n" + "Please first provide a concise reasoning for your classification, and then clearly state your classification. Feel free to navigate the codebase and read files as needed.",
            cwd=temp_dir
        )
        if "content" not in _trace[-1]:
            raise RuntimeError(
                f"codex output is not valid JSON ({_trace[-1]})"
            )
        _agent_final_output = _trace[-1]["content"]
        if isinstance(_agent_final_output, list):
            _collected_text = ""
            for _item in _agent_final_output:
                _text = _item.get("text", "")
                if _text:
                    _collected_text += _text + "\n"
            if not _collected_text:
                _agent_final_output = _trace[-1]["content"]
            else:
                _agent_final_output = _collected_text
        _agent_final_output = str(_agent_final_output).strip()
        input_bouncer_response = get_response(
            az_client,
            "o4-mini",
            "Your objective is to convert the user's reasoning about classifying a software ticket into a JSON object without modifying or changing any data. The JSON object should contain the following fields: 'explanation' and 'label'.",
            "Please extract the reasoning and label based on the following data alone:\n\n" + _agent_final_output,
            InputBouncerResponse
        )
    else:
        input_bouncer_response = get_response(
            client,
            deployment,
            system_prompt,
            user_prompt,
            InputBouncerResponse
        )
    data[instance_id] = {
        "explanation": input_bouncer_response.explanation,
        "label": input_bouncer_response.label
    }
    if _trace is not None:
        data[instance_id]["trace"] = _trace
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    

def output_bouncer(client, deployment, instance_id, issue_description, generated_response, temp_dir=None):
    if client is None:
        _prompt_file_loc = "prompts/output_bouncer_codex.txt"
    else:
        _prompt_file_loc = "prompts/output_bouncer.txt"
    _deployment_simple = deployment.split("/")[-1].replace(":", "_")
    output_file = os.path.join(OUTPUT_DIR, f"output_bouncer_{_deployment_simple}.json")
    data = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
        if instance_id in data:
            print(f"Instance {instance_id} already exists in {output_file}. Skipping.")
            return
    with open(_prompt_file_loc, "r") as _prompt_file:
        system_prompt = _prompt_file.read().strip()
    user_prompt = dedent(
        """
        <PATCH>
        {generated_response}
        </PATCH>

        Does this patch correctly address the issue described below?

        <ISSUE_DESCRIPTION>
        {issue_description}
        </ISSUE_DESCRIPTION>
        """
    ).format(issue_description=issue_description, generated_response=generated_response)
    _trace = None
    if client is None:
        try:
            _trace = run_codex(
                prompt=system_prompt + "\n" + user_prompt + "\n" + "Please first provide a concise reasoning for your classification, and then clearly state your classification in the format: <EXPLANATION>reasoning</EXPLANATION><LABEL>label</LABEL>.",
                cwd=temp_dir
            )
            _agent_final_output = _trace[-1]["content"]
            if isinstance(_agent_final_output, list):
                _collected_text = ""
                for _item in _agent_final_output:
                    _text = _item.get("text", "")
                    if _text:
                        _collected_text += _text + "\n"
                if not _collected_text:
                    _agent_final_output = _trace[-1]["content"]
                else:
                    _agent_final_output = _collected_text
            _agent_final_output = str(_agent_final_output).strip()
            output_bouncer_response = get_response(
                az_client,
                "o4-mini",
                "Your objective is to convert the user's reasoning about classifying a code patch into a JSON object without modifying or changing any data. The JSON object should contain the following fields: 'explanation' and 'label'.",
                "Please extract the reasoning and label based on the following data alone:\n\n" + _agent_final_output,
                OutputBouncerResponse
            )
        except OSError as e:
            if e.errno != 7:
                raise
            print(f"Error running Codex: {e}")
            output_bouncer_response = OutputBouncerResponse(
                explanation="ERROR: FAILED TO PROCESS",
                label="INCORRECT"
            )
    else:
        output_bouncer_response = get_response(
            client,
            deployment,
            system_prompt,
            user_prompt,
            OutputBouncerResponse
        )
    data[instance_id] = {
        "explanation": output_bouncer_response.explanation,
        "label": output_bouncer_response.label
    }
    if _trace is not None:
        data[instance_id]["trace"] = _trace
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def run_input_bouncer(codex=False):
    df = pd.read_csv("./dataset/input_bouncer.csv")
    preclone_repos(sorted(set(df["repo"])))
    # Input Bouncer with tqdm progress bar
    print("Running input_bouncer...")
    with tqdm(total=len(df), desc="Issues", position=0) as pbar_outer:
        for _, row in df.iterrows():
            instance_id = row["instance_id"]

            issue_description = row["problem_statement"]
            if codex:
                _deployment_simple = "codex"
                output_file = os.path.join(OUTPUT_DIR, f"input_bouncer_{_deployment_simple}.json")
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        data = json.load(f)
                    if instance_id in data:
                        tqdm.write(f"Instance {instance_id} already exists in {output_file}. Skipping.")
                        pbar_outer.update(1)
                        continue

                repo_name = row["repo"]
                base_commit = row["base_commit"]
                cached = CACHE_DIR / repo_name.replace("/", "_")
                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.copytree(str(cached), temp_dir, dirs_exist_ok=True)

                    repo = pygit2.Repository(str(temp_dir))
                    commit = repo.get(base_commit)
                    repo.set_head(commit.id)
                    repo.reset(commit.id, pygit2.GIT_RESET_HARD)
                    try:
                        input_bouncer(None, None, instance_id, issue_description, temp_dir)
                    except Exception as e:
                        print(f"Error running Codex: {e}")
                        continue
            else:
                for model in tqdm(model_list, desc=f"Models for {instance_id[:10]}...", position=1, leave=False):
                    client = model["client"]
                    deployment = model["deployment"]
                    try:
                        input_bouncer(client, deployment, instance_id, issue_description)
                    except Exception as e:
                        continue
            pbar_outer.update(1)

def run_output_bouncer(codex=False, data_loc="./dataset/random_sample_bouncer.csv", ignored_instances=None):
    df = pd.read_csv(data_loc)
    preclone_repos(sorted(set(df["repo"])))
    print("Running output_bouncer...")
    with tqdm(total=len(df), desc="Issues", position=0) as pbar_outer:
        for _, row in df.iterrows():
            instance_id = row["instance_id"]
            if ignored_instances is not None and instance_id in ignored_instances:
                tqdm.write(f"Instance {instance_id} is in ignored instances. Skipping.")
                pbar_outer.update(1)
                continue
            issue_description = row["problem_statement"]
            generated_response = row["patch"]
            if codex:
                if data_loc == "./dataset/random_sample_bouncer.csv":
                    _deployment_simple = "codex"
                else:
                    _deployment_simple = "codex_" + row["agent"]

                output_file = os.path.join(OUTPUT_DIR, f"output_bouncer_{_deployment_simple}.json")
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        data = json.load(f)
                    if instance_id in data:
                        tqdm.write(f"Instance {instance_id} already exists in {output_file}. Skipping.")
                        pbar_outer.update(1)
                        continue
                
                repo_name = row["repo"]
                base_commit = row["base_commit"]
                cached = CACHE_DIR / repo_name.replace("/", "_")
                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.copytree(str(cached), temp_dir, dirs_exist_ok=True)
                    repo = pygit2.Repository(str(temp_dir))
                    commit = repo.get(base_commit)
                    repo.set_head(commit.id)
                    repo.reset(commit.id, pygit2.GIT_RESET_HARD)
                    try:
                        output_bouncer(None, _deployment_simple, instance_id, issue_description, generated_response, temp_dir)
                    except Exception as e:
                        print(f"Error running Codex: {e}")
                        continue
            else:
                for model in tqdm(model_list, desc=f"Models for {instance_id[:10]}...", position=1, leave=False):
                    client = model["client"]
                    deployment = model["deployment"]
                    try:
                        output_bouncer(
                            client,
                            deployment,
                            instance_id,
                            issue_description,
                            generated_response,
                            temp_dir=temp_dir
                        )
                    except Exception as e:
                        continue
            pbar_outer.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run input or output bouncer.")
    parser.add_argument(
        "--input",
        action="store_true",
        help="Run input bouncer"
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="Run output bouncer"
    )
    parser.add_argument(
        "--codex",
        action="store_true",
        help="Run codex bouncer"
    )
    parser.add_argument(
        "--agents",
        action="store_true",
        help="Run output bouncer for all agents"
    )
    args = parser.parse_args()
    if args.input:
        run_input_bouncer(args.codex)
    elif args.output:
        run_output_bouncer(args.codex)
    elif args.agents:
        selected_input_bouncer = "outputs/input_bouncer_o4-mini.json"
        with open(selected_input_bouncer) as f:
            selected_input_bouncer_data = json.load(f)
        ignored_instances = []
        for instance_id, info in selected_input_bouncer_data.items():
            if info["label"] not in ["WELL_SPECIFIED", "REASONABLY_SPECIFIED"]:
                ignored_instances.append(instance_id)
        for _agent in ["amazon-q_bouncer.csv", "sweagent_bouncer.csv", "OpenHands_bouncer.csv"]:
            print(f"Running output bouncer for {_agent}...")
            run_output_bouncer(True, data_loc=f"./dataset/agent_output/{_agent}", ignored_instances=ignored_instances)
    else:
        print("Please specify --input or --output or --agents to run the bouncer.")
        exit(1)