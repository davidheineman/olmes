"""DeepCoder: Competitive programming evaluation using the TACO subset of DeepCoder-Preview-Dataset.
https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset

Homepage: https://github.com/agentica-project/DeepCoder
"""

import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union

import datasets
import numpy as np
from tqdm import tqdm

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.metrics.metric import CodePassAtK
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template
from oe_eval.utilities.extraction_utils import extract_code
from oe_eval.utilities.lm_styles import LanguageModelStore, LMStyle
from oe_eval.utilities.testing_util import run_test

sys.set_int_max_str_digits(50000)


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    run_test_result = run_test(sample, test=generation, debug=debug, timeout=int(timeout))
    if run_test_result:
        res, metadata = run_test_result
    else:
        in_outs = json.loads(sample["input_output"])
        num_tests = len(in_outs["inputs"])
        res = [-4] * num_tests
        metadata = {
            "error_code": -4,
            "error_message": "Function not found or compilation error.",
        }
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=False):
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        metadata_list = [{"error_code": -1, "error_message": "global timeout"}]
        if debug:
            print("global timeout")
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(sample, o, timeout=timeout, debug=debug)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            if debug:
                print(f"Compilation failed: {repr(e)}")
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            assert isinstance(curr_metadata, dict), curr_metadata
            res.append(curr_res)
            metadata.append(curr_metadata)
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }
            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(inputs), f"results={len(results)} inputs={len(inputs)}"
    return results, metadata


class DeepCoder(Task):
    """DeepCoder competitive programming evaluation (TACO subset, stdin/stdout format)."""

    VERSION = 0.1
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "/datasets/pretraining_data/dhei/hf_datasets/agentica-org__DeepCoder-Preview-Dataset",
        "dataset_name": "taco",
        "native_id_field": "task_id",
        "primary_metric": "pass_at_1",
        "split": "train",
        "generation_kwargs": {
            "max_gen_toks": 2048,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
            "repeats": 10,
        },
        "context_kwargs": {
            "answer_prefix": "",
        },
        "metric_kwargs": {
            "pass_at_ks": [1, 5, 10],
            "timeout": 10.0,
            "n_exe_workers": 20,
        },
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset_path = self.task_config["dataset_path"]
        dataset_name = self.task_config.get("dataset_name", "taco")

        if os.path.exists(dataset_path):
            subdir = os.path.join(dataset_path, dataset_name)
            if os.path.exists(subdir):
                self.dataset = datasets.load_dataset("parquet", data_dir=subdir)
            else:
                self.dataset = datasets.load_dataset("parquet", data_dir=dataset_path)
        else:
            self.dataset = datasets.load_dataset(dataset_path, dataset_name)

    def make_metrics(self):
        self._metrics = [
            CodePassAtK(
                process_code_results_fn=self._process_code_results,
                code_exe_fn=self._custom_code_eval,
                extra_metric_names=["metadata"],
                **self.task_config["metric_kwargs"],
            )
        ]
        return self._metrics

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return []

    def test_docs(self):
        return []

    def doc_to_text(self, doc):
        return doc["query"]

    def _process_doc(self, doc):
        out_doc = {}
        out_doc["task_id"] = doc.get("task_id", str(hash(doc["problem"])))
        out_doc["problem"] = doc["problem"]

        tests = doc["tests"] if isinstance(doc["tests"], dict) else json.loads(doc["tests"])
        out_doc["input_output"] = json.dumps({
            "inputs": tests["inputs"],
            "outputs": tests["outputs"],
            "fn_name": None,
        })
        out_doc["num_tests"] = len(tests["inputs"])

        format_instruction = (
            "Read the inputs from stdin, solve the problem, and write the answer to stdout. "
            "Enclose your code within delimiters as follows.\n"
            "```python\n# YOUR CODE HERE\n```"
        )

        if self.task_config.get("use_chat_format", False):
            out_doc["problem_statement"] = doc["problem"]
            out_doc["format_instruction"] = format_instruction
            out_doc["query"] = doc["problem"]
        else:
            system_message = (
                "You are an expert Python programmer. You will be given a question "
                "(problem specification) and will generate a correct Python program "
                "that matches the specification and passes all tests."
            )
            query = (
                f"{system_message}\n\n"
                f"### Question:\n{doc['problem']}\n\n"
                f"### Format: {format_instruction}\n"
                f"### Answer: (use the provided format with backticks)\n\n"
            )
            query += self.task_config["context_kwargs"].get("answer_prefix", "")
            out_doc["query"] = query

        out_doc = apply_prompt_template(out_doc, self.task_config)
        return out_doc

    def doc_to_target(self, doc):
        pass

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc=doc, ctx=ctx, doc_id=doc_id)

    def _process_code_results(self, results: List[dict]) -> List[dict]:
        output = []
        for res in results:
            completion = res["model_resps"]["continuation"]
            lm_style = LMStyle.OpenAIChat
            model_name = res.get("model_resps", {}).get("llm_response", {}).get("model")
            if model_name:
                model_obj = LanguageModelStore.get(model_name, None)
                if model_obj is not None:
                    lm_style = model_obj.model_style
            extracted_code = extract_code(completion, lm_style)
            output.append({
                "res_id": res["res_id"],
                "doc": res["doc"],
                "completion": extracted_code,
            })
        return output

    def _custom_code_eval(
        self, batched_code_test: List[dict], timeout_sec: int = 10, n_exe_workers: int = 1
    ) -> List[dict]:
        generations_by_doc = defaultdict(list)
        for item in batched_code_test:
            doc_id = item["doc"]["task_id"]
            generations_by_doc[doc_id].append(item)

        unique_docs = {item["doc"]["task_id"]: item["doc"] for item in batched_code_test}
        sorted_unique_docs = sorted(unique_docs.items())

        samples_list = [doc for _, doc in sorted_unique_docs]
        generations_list = [
            [item["completion"] for item in generations_by_doc[tid]]
            for tid, _ in sorted_unique_docs
        ]
        doc_id_to_index = {tid: i for i, (tid, _) in enumerate(sorted_unique_docs)}

        results, metadata = evaluate_generations(
            samples_list=samples_list,
            generations_list=generations_list,
            timeout=timeout_sec,
            num_process_evaluate=n_exe_workers,
        )

        outcomes = []
        for tid, items in generations_by_doc.items():
            sample_idx = doc_id_to_index.get(tid)
            if sample_idx is None or sample_idx not in results:
                for item in items:
                    outcomes.append({"res_id": item["res_id"], "passed": False})
                continue

            generation_results = results[sample_idx]
            generation_metadata = metadata[sample_idx]

            for i, item in enumerate(items):
                passed = all(r is True for r in generation_results[i])
                outcome = {
                    "res_id": item["res_id"],
                    "passed": bool(passed),
                    "metadata": generation_metadata[i],
                }
                outcomes.append(outcome)

        return outcomes
