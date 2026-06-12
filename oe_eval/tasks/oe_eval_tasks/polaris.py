"""
Polaris Dataset - Competition math problems from POLARIS-Project/Polaris-Dataset-53K.
"""

import re
from typing import List, Union

from lm_eval.tasks.hendrycks_math.utils import is_equiv as hendrycks_is_equiv
from lm_eval.tasks.minerva_math.utils import (
    get_unnormalized_answer,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
)
from lm_eval.tasks.minerva_math.utils import remove_boxed

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.metrics.metric import GenericMetric
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template, map_indexed


class Polaris(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    DATASET_PATH = "/datasets/pretraining_data/dhei/hf_datasets/POLARIS-Project__Polaris-Dataset-53K"
    TASK_CONFIG_DEFAULTS = {
        "native_id_field": "index",
        "split": "train",
        "primary_metric": "exact_match_flex",
        "context_kwargs": {
            "use_cot": True,
            "cot_style": "plain",
        },
        "generation_kwargs": {
            "max_gen_toks": 4096,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": [],
        },
        "chat_overrides": {
            "generation_kwargs": {
                "stop_sequences": [],
            },
        },
        "metric_kwargs": {},
    }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        return map_indexed(self._process_doc, self.dataset["train"])

    def _process_doc(self, doc, index=1):
        query = doc["problem"].strip()
        answer = doc["answer"].strip()
        out_doc = {
            "index": index,
            "problem": doc["problem"],
            "query": query,
            "answer": answer,
            "difficulty": doc.get("difficulty"),
        }
        out_doc = apply_prompt_template(out_doc, self.task_config)
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        doc.update({"choices": [doc["answer"]]})
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answer"])

    def extract_answers(self, results):
        raw_answer = results[0] if isinstance(results, list) else results
        all_answers = []
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
        if boxed_answer is not None:
            all_answers.append(normalize_final_answer(boxed_answer))
        if len(all_answers) == 0:
            dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)
        if len(all_answers) == 0:
            all_answers.append(normalize_final_answer(raw_answer))
        return all_answers

    def make_metrics(self):
        self._metrics = [
            GenericMetric(
                process_results_fn=self.process_results,
                metric_names=["exact_match", "exact_match_flex"],
                **self.task_config["metric_kwargs"],
            ),
        ]
        return self._metrics

    def process_results(self, doc, results):
        raw_answer = results[0] if isinstance(results, list) else results
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                extracted = normalize_final_answer(remove_boxed(boxed_answer))
            except AssertionError:
                extracted = normalize_final_answer(raw_answer)
        else:
            extracted = normalize_final_answer(raw_answer)

        gold = doc["answer"]
        exact_match = 1 if is_equiv(extracted, gold) else 0
        if exact_match == 0 and hendrycks_is_equiv(extracted, gold):
            exact_match = 1

        max_flex_match = exact_match
        if max_flex_match == 0:
            all_extracted = self.extract_answers(results)
            for ans in all_extracted:
                if is_equiv(ans, gold) or hendrycks_is_equiv(ans, gold):
                    max_flex_match = 1
                    break

        metrics = {
            "exact_match": exact_match,
            "exact_match_flex": max_flex_match,
        }
        all_extracted = self.extract_answers(results)
        if all_extracted:
            metrics["model_answer"] = all_extracted[0]
        return metrics
