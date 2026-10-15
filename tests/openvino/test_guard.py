#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import unittest
from types import SimpleNamespace

import torch
from parameterized import parameterized
from transformers import AutoTokenizer, set_seed
from utils_tests import MODEL_NAMES, OPENVINO_DEVICE, SEED

from optimum.intel import (
    OVGuard,
    OVGuardConfig,
    OVGuardReport,
    OVGuardVerdict,
    OVGuardViolationError,
    OVModelForCausalLM,
    OVModelForTokenClassification,
)
from optimum.intel.openvino.generation_guard import (
    GENERATIVE_GUARD_SPECS,
    PROMPT_SCOPE,
    RESPONSE_SCOPE,
    UNPARSABLE_LABEL,
    GenerativeGuardSpec,
    get_generative_guard_spec,
)


def _spec(name: str) -> GenerativeGuardSpec:
    return next(spec for spec in GENERATIVE_GUARD_SPECS if spec.name == name)


#: Matches any answer and blocks nothing, so a randomly initialized guard model always clears the
#: content. Doubles as the test of `guard_spec=`, the extension point for unlisted guard families.
ALWAYS_CLEARING_SPEC = GenerativeGuardSpec(
    name="test-always-clearing",
    template_marker=_spec("Llama Guard").template_marker,
    label_pattern=r"(?P<label>.*)",
    labels=("safe",),
    default_blocking_labels=(),
    content_as_parts=True,
    max_new_tokens=4,
)


class GenerativeGuardSpecTest(unittest.TestCase):
    """Parses answers recorded from the real guard models, without running any model."""

    LLAMA_GUARD_ANSWERS = (
        ("\n\nsafe", "safe", ()),
        ("\n\nunsafe\nS1", "unsafe", ("S1",)),
        ("\n\nunsafe\nS1,S9", "unsafe", ("S1", "S9")),
    )
    QWEN3_GUARD_GEN_ANSWERS = (
        ("Safety: Safe\nCategories: None", "Safe", ()),
        ("Safety: Unsafe\nCategories: Violent", "Unsafe", ("Violent",)),
        (
            "Safety: Unsafe\nCategories: Non-violent Illegal Acts\nRefusal: No",
            "Unsafe",
            ("Non-violent Illegal Acts",),
        ),
        (
            "Safety: Controversial\nCategories: Politically Sensitive Topics",
            "Controversial",
            ("Politically Sensitive Topics",),
        ),
    )

    @parameterized.expand([("Llama Guard",), ("Qwen3Guard-Gen",)])
    def test_spec_parses_recorded_answers(self, name):
        spec = _spec(name)
        answers = self.LLAMA_GUARD_ANSWERS if name == "Llama Guard" else self.QWEN3_GUARD_GEN_ANSWERS
        for answer, expected_label, expected_categories in answers:
            self.assertEqual(spec.parse(answer), (expected_label, expected_categories), msg=answer)
        # Every label the spec declares has to be covered by a recorded answer, so that adding one
        # without a matching sample is caught here rather than in production.
        self.assertEqual({label for _, label, _ in answers}, set(spec.labels))
        self.assertLessEqual(set(spec.default_blocking_labels), set(spec.labels))

    @parameterized.expand([("Llama Guard",), ("Qwen3Guard-Gen",)])
    def test_spec_reports_unparsable_answers(self, name):
        with self.assertLogs("optimum.intel.openvino.generation_guard", level="WARNING"):
            label, categories = _spec(name).parse("I am not a guard model.")
        self.assertEqual(label, UNPARSABLE_LABEL)
        self.assertEqual(categories, ())

    @parameterized.expand([("Llama Guard",), ("Qwen3Guard-Gen",)])
    def test_spec_is_detected_from_the_chat_template(self, name):
        spec = _spec(name)
        tokenizer = SimpleNamespace(chat_template=f"whatever ... {spec.template_marker} ... more")
        self.assertIs(get_generative_guard_spec(tokenizer), spec)

    def test_unknown_chat_template_is_not_detected(self):
        self.assertIsNone(get_generative_guard_spec(SimpleNamespace(chat_template="{{ messages }}")))
        self.assertIsNone(get_generative_guard_spec(SimpleNamespace(chat_template=None)))


class OVGuardConfigAndReportTest(unittest.TestCase):
    """Unit tests for the model-independent pieces of `optimum.intel.openvino.generation_guard`."""

    def test_default_config_is_valid(self):
        config = OVGuardConfig()
        self.assertIsNone(config.chunk_size)
        self.assertEqual(config.prompt_mode, "async")
        self.assertEqual(config.on_violation, "stop")
        self.assertTrue(config.emit_before_check)

    @parameterized.expand(
        [
            ({"chunk_size": 0},),
            ({"prompt_mode": "invalid"},),
            ({"on_violation": "invalid"},),
        ]
    )
    def test_config_rejects_invalid_values(self, kwargs):
        with self.assertRaises(ValueError):
            OVGuardConfig(**kwargs)

    @staticmethod
    def _verdict(flagged: bool, scope: str = RESPONSE_SCOPE, token_index: int = 0) -> OVGuardVerdict:
        return OVGuardVerdict(
            label="Unsafe" if flagged else "Safe",
            flagged=flagged,
            scope=scope,
            categories=("Violent",),
            score=0.9,
            token_index=token_index,
        )

    def test_verdict_role_follows_scope(self):
        self.assertEqual(self._verdict(False, scope=PROMPT_SCOPE).role, "user")
        self.assertEqual(self._verdict(False, scope=RESPONSE_SCOPE).role, "assistant")

    def test_report_splits_verdicts_per_rail(self):
        prompt_verdict = self._verdict(False, scope=PROMPT_SCOPE)
        response_verdict = self._verdict(True, token_index=3)
        report = OVGuardReport([prompt_verdict, response_verdict])

        self.assertEqual(report.prompt, [prompt_verdict])
        self.assertEqual(report.response, [response_verdict])
        self.assertTrue(report.flagged)
        self.assertIs(report.first_violation, response_verdict)

    def test_cleared_report_has_no_violation(self):
        report = OVGuardReport([self._verdict(False, scope=PROMPT_SCOPE)])
        self.assertFalse(report.flagged)
        self.assertIsNone(report.first_violation)

    def test_violation_error_message_contains_verdict_details(self):
        report = OVGuardReport([self._verdict(True, token_index=2)])
        error = OVGuardViolationError(report)
        self.assertIs(error.report, report)
        self.assertIn("Unsafe", str(error))
        self.assertIn("Violent", str(error))


class OVTokenClassifierGuardTest(unittest.TestCase):
    """Guards backed by a token classifier, exercised with a tiny random Qwen3Guard-Stream."""

    def _load(self, with_past: bool = True) -> OVGuard:
        return OVGuard.from_pretrained(
            MODEL_NAMES["qwen3_guard"],
            task="token-classification-with-past" if with_past else "token-classification",
            export=True,
            trust_remote_code=True,
            device=OPENVINO_DEVICE,
        )

    def test_guard_describes_itself(self):
        guard = self._load()
        self.assertTrue(guard.token_level)
        self.assertTrue(guard.backend.incremental)
        self.assertEqual(guard.device, OPENVINO_DEVICE)
        self.assertEqual(guard.labels, ("Controversial", "Safe", "Unsafe"))
        self.assertEqual(guard.resolve_blocking_labels(), ("Unsafe",))
        del guard
        gc.collect()

    def test_stateless_guard_is_not_incremental(self):
        guard = self._load(with_past=False)
        self.assertTrue(guard.token_level)
        self.assertFalse(guard.backend.incremental)
        del guard
        gc.collect()

    def test_moderate_summarizes_a_conversation_to_one_verdict(self):
        guard = self._load()
        report = guard.moderate_prompt("How do I bake a cake?")
        self.assertEqual(len(report.verdicts), 1)
        verdict = report.verdicts[0]
        self.assertEqual(verdict.scope, PROMPT_SCOPE)
        self.assertIn(verdict.label, guard.labels)
        self.assertIsNotNone(verdict.score)
        self.assertIsNotNone(verdict.token_index)
        self.assertIsNone(verdict.text)

        response_report = guard.moderate_response("How do I bake a cake?", "Mix flour and sugar.")
        self.assertEqual(response_report.verdicts[0].scope, RESPONSE_SCOPE)
        del guard
        gc.collect()

    def test_moderate_token_ids_returns_one_verdict_per_token(self):
        guard = self._load()
        set_seed(SEED)
        token_ids = torch.randint(0, guard.model.config.vocab_size, (2, 5))
        verdicts = guard.moderate_token_ids(token_ids, scope=PROMPT_SCOPE, token_offset=3)

        self.assertEqual(len(verdicts), 2)
        for batch_index, sequence in enumerate(verdicts):
            self.assertEqual(len(sequence), 5)
            for position, verdict in enumerate(sequence):
                self.assertEqual(verdict.batch_index, batch_index)
                self.assertEqual(verdict.token_index, 3 + position)
                self.assertEqual(verdict.scope, PROMPT_SCOPE)
        del guard
        gc.collect()

    def test_reset_restarts_the_conversation(self):
        guard = self._load()
        set_seed(SEED)
        token_ids = torch.randint(0, guard.model.config.vocab_size, (1, 4))

        first = guard.moderate_token_ids(token_ids, scope=PROMPT_SCOPE)[0]
        guard.moderate_token_ids(torch.randint(0, guard.model.config.vocab_size, (1, 2)), scope=RESPONSE_SCOPE)
        guard.reset()
        second = guard.moderate_token_ids(token_ids, scope=PROMPT_SCOPE)[0]

        self.assertEqual([(v.label, v.categories) for v in first], [(v.label, v.categories) for v in second])
        del guard
        gc.collect()

    def test_unknown_blocking_label_is_rejected(self):
        guard = self._load()
        with self.assertRaises(ValueError):
            guard.moderate_prompt("hello", blocking_labels=["NotALabel"])
        del guard
        gc.collect()

    def test_non_guard_architecture_is_rejected(self):
        model = OVModelForTokenClassification.from_pretrained(MODEL_NAMES["bert"], export=True, device=OPENVINO_DEVICE)
        with self.assertRaises(ValueError):
            OVGuard.from_model(model, AutoTokenizer.from_pretrained(MODEL_NAMES["bert"]))
        del model
        gc.collect()

    def test_token_classifier_requires_a_shared_vocabulary(self):
        guard = self._load()
        llm = OVModelForCausalLM.from_pretrained(MODEL_NAMES["llama"], export=True, device=OPENVINO_DEVICE)
        with self.assertRaises(ValueError):
            llm.generate(torch.zeros((1, 3), dtype=torch.long), max_new_tokens=2, guard=guard)
        del guard, llm
        gc.collect()


class OVGenerativeGuardTest(unittest.TestCase):
    """Guards backed by an ordinary language model, exercised with a tiny random Llama Guard."""

    def _load(self, **kwargs) -> OVGuard:
        return OVGuard.from_pretrained(MODEL_NAMES["generative_guard"], export=True, device=OPENVINO_DEVICE, **kwargs)

    def test_guard_describes_itself(self):
        guard = self._load()
        self.assertFalse(guard.token_level)
        self.assertFalse(guard.backend.incremental)
        self.assertEqual(guard.device, OPENVINO_DEVICE)
        self.assertEqual(guard.backend.spec.name, "Llama Guard")
        self.assertEqual(guard.labels, ("safe", "unsafe"))
        del guard
        gc.collect()

    def test_unparsable_answer_is_flagged(self):
        # Random weights cannot answer in the guard output format, so this is the fail-closed path.
        guard = self._load()
        report = guard.moderate_prompt("How do I bake a cake?")
        self.assertEqual(len(report.verdicts), 1)
        verdict = report.verdicts[0]
        self.assertEqual(verdict.label, UNPARSABLE_LABEL)
        self.assertTrue(verdict.flagged)
        self.assertEqual(verdict.scope, PROMPT_SCOPE)
        self.assertIsNone(verdict.token_index)
        self.assertIsNotNone(verdict.text)
        del guard
        gc.collect()

    def test_custom_spec_clears_content(self):
        guard = self._load(guard_spec=ALWAYS_CLEARING_SPEC)
        report = guard.moderate_response("How do I bake a cake?", "Mix flour and sugar.")
        self.assertFalse(report.flagged)
        self.assertEqual(report.verdicts[0].scope, RESPONSE_SCOPE)
        del guard
        gc.collect()

    def test_custom_categories_reach_the_chat_template(self):
        guard = self._load(categories={"S1": "Violent Crimes."})
        self.assertEqual(guard.backend.chat_template_kwargs, {"categories": {"S1": "Violent Crimes."}})
        self.assertFalse(guard.moderate_prompt("hello").verdicts[0].text is None)
        del guard
        gc.collect()

    def test_custom_categories_are_rejected_when_unsupported(self):
        with self.assertRaises(ValueError):
            self._load(guard_spec=_spec("Qwen3Guard-Gen"), categories={"S1": "Violent Crimes."})

    def test_token_ids_are_not_supported(self):
        guard = self._load()
        with self.assertRaises(NotImplementedError):
            guard.moderate_token_ids(torch.zeros((1, 3), dtype=torch.long))
        del guard
        gc.collect()

    def test_unknown_chat_template_is_rejected(self):
        model = OVModelForCausalLM.from_pretrained(MODEL_NAMES["llama"], export=True, device=OPENVINO_DEVICE)
        with self.assertRaises(ValueError):
            OVGuard.from_model(model, AutoTokenizer.from_pretrained(MODEL_NAMES["llama"]))
        del model
        gc.collect()


class OVGuardedGenerationTest(unittest.TestCase):
    """Tests the `guard=`/`guard_config=` arguments of `OVModelForCausalLM.generate`.

    Tiny random models produce semantically meaningless verdicts, so the tests never rely on what
    the guard actually predicts: a token classifier is forced to flag everything or nothing through
    `blocking_labels`, and a generative guard either fails closed on its unparsable answer or is
    given [`ALWAYS_CLEARING_SPEC`].
    """

    ALL_LABELS = ("Safe", "Unsafe", "Controversial")

    def _load_llm(self, model_name: str = "qwen3"):
        llm = OVModelForCausalLM.from_pretrained(MODEL_NAMES[model_name], export=True, device=OPENVINO_DEVICE)
        # Random weights can make the tiny model immediately emit an eos token, which would stop
        # generation for a reason unrelated to the guard and make the tests flaky.
        llm.generation_config.eos_token_id = None
        llm.config.eos_token_id = None
        return llm

    def _load_token_classifier_guard(self, with_past: bool = True) -> OVGuard:
        return OVGuard.from_pretrained(
            MODEL_NAMES["qwen3_guard"],
            task="token-classification-with-past" if with_past else "token-classification",
            export=True,
            trust_remote_code=True,
            device=OPENVINO_DEVICE,
        )

    def _load_generative_guard(self, **kwargs) -> OVGuard:
        return OVGuard.from_pretrained(MODEL_NAMES["generative_guard"], export=True, device=OPENVINO_DEVICE, **kwargs)

    def _prompt(self, llm, batch_size: int = 1):
        set_seed(SEED)
        return torch.randint(0, llm.config.vocab_size, (batch_size, 6))

    # --- token classifier guards ---------------------------------------------------------------

    def test_generation_stops_on_violation(self):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        ids = self._prompt(llm)
        out = llm.generate(
            ids,
            max_new_tokens=8,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(chunk_size=2, blocking_labels=self.ALL_LABELS),
            return_dict_in_generate=True,
        )
        self.assertTrue(out.guard.flagged)
        self.assertLess(out.sequences.shape[-1] - ids.shape[-1], 8)
        del llm, guard
        gc.collect()

    def test_every_response_token_is_moderated(self):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        ids = self._prompt(llm)
        out = llm.generate(
            ids,
            max_new_tokens=8,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(chunk_size=3, blocking_labels=()),
            return_dict_in_generate=True,
        )
        self.assertFalse(out.guard.flagged)
        self.assertEqual(out.sequences.shape[-1] - ids.shape[-1], 8)
        self.assertEqual(len(out.guard.response), 8)
        del llm, guard
        gc.collect()

    def test_on_violation_raise(self):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        with self.assertRaises(OVGuardViolationError) as context:
            llm.generate(
                self._prompt(llm),
                max_new_tokens=8,
                do_sample=False,
                guard=guard,
                guard_config=OVGuardConfig(blocking_labels=self.ALL_LABELS, on_violation="raise"),
            )
        self.assertTrue(context.exception.report.flagged)
        del llm, guard
        gc.collect()

    @parameterized.expand([("sync",), ("async",)])
    def test_on_violation_raise_aborts_on_a_flagged_prompt(self, prompt_mode):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        with self.assertRaises(OVGuardViolationError) as context:
            llm.generate(
                self._prompt(llm),
                max_new_tokens=8,
                do_sample=False,
                guard=guard,
                guard_config=OVGuardConfig(
                    blocking_labels=self.ALL_LABELS, on_violation="raise", prompt_mode=prompt_mode
                ),
            )
        self.assertEqual(context.exception.report.first_violation.scope, PROMPT_SCOPE)
        self.assertEqual(len(context.exception.report.response), 0)
        del llm, guard
        gc.collect()

    def test_on_violation_continue_does_not_stop_generation(self):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        ids = self._prompt(llm)
        out = llm.generate(
            ids,
            max_new_tokens=8,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(blocking_labels=self.ALL_LABELS, on_violation="continue"),
            return_dict_in_generate=True,
        )
        self.assertTrue(out.guard.flagged)
        self.assertEqual(out.sequences.shape[-1] - ids.shape[-1], 8)
        del llm, guard
        gc.collect()

    @parameterized.expand([("off", 0), ("sync", 1), ("async", 1)])
    def test_prompt_mode(self, prompt_mode, expected_prompt_verdicts):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        out = llm.generate(
            self._prompt(llm),
            max_new_tokens=4,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(blocking_labels=(), prompt_mode=prompt_mode),
            return_dict_in_generate=True,
        )
        self.assertEqual(len(out.guard.prompt), expected_prompt_verdicts)
        del llm, guard
        gc.collect()

    @parameterized.expand([("stateless_guard",), ("beam_search",)])
    def test_full_rescan_fallbacks(self, scenario):
        llm = self._load_llm()
        guard = self._load_token_classifier_guard(with_past=scenario != "stateless_guard")
        out = llm.generate(
            self._prompt(llm),
            max_new_tokens=6,
            do_sample=False,
            num_beams=2 if scenario == "beam_search" else 1,
            guard=guard,
            guard_config=OVGuardConfig(chunk_size=2, blocking_labels=self.ALL_LABELS),
            return_dict_in_generate=True,
        )
        self.assertTrue(out.guard.flagged)
        del llm, guard
        gc.collect()

    def test_batched_generation_covers_every_sequence(self):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        ids = self._prompt(llm, batch_size=2)
        attention_mask = torch.ones_like(ids)
        attention_mask[0, :2] = 0  # left-pad the first sequence
        out = llm.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(blocking_labels=()),
            return_dict_in_generate=True,
        )
        self.assertEqual({verdict.batch_index for verdict in out.guard.response}, {0, 1})
        del llm, guard
        gc.collect()

    def test_on_verdict_callback_fires_for_every_verdict(self):
        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        verdicts = []
        out = llm.generate(
            self._prompt(llm),
            max_new_tokens=4,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(blocking_labels=(), on_verdict=verdicts.append),
            return_dict_in_generate=True,
        )
        self.assertEqual(len(verdicts), len(out.guard.verdicts))
        del llm, guard
        gc.collect()

    def test_emit_before_check_false_holds_back_streamer(self):
        class RecordingStreamer:
            def __init__(self):
                self.puts = []
                self.ended = False

            def put(self, value):
                self.puts.append(value)

            def end(self):
                self.ended = True

        llm, guard = self._load_llm(), self._load_token_classifier_guard()
        streamer = RecordingStreamer()
        out = llm.generate(
            self._prompt(llm),
            max_new_tokens=6,
            do_sample=False,
            guard=guard,
            streamer=streamer,
            guard_config=OVGuardConfig(chunk_size=2, blocking_labels=self.ALL_LABELS, emit_before_check=False),
            return_dict_in_generate=True,
        )
        self.assertTrue(out.guard.flagged)
        self.assertTrue(streamer.ended)
        # Every label is blocked, so the prompt verdict is already a violation and no generated
        # token can have been cleared. The 2D put only carries the prompt.
        self.assertEqual(sum(1 for value in streamer.puts if value.ndim == 1), 0)
        del llm, guard
        gc.collect()

    # --- generative guards ---------------------------------------------------------------------

    def test_generative_guard_stops_on_violation(self):
        llm, guard = self._load_llm("llama"), self._load_generative_guard()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES["llama"])
        ids = self._prompt(llm)
        out = llm.generate(
            ids,
            max_new_tokens=8,
            do_sample=False,
            tokenizer=tokenizer,
            guard=guard,
            return_dict_in_generate=True,
        )
        # The prompt rail already fails closed, so generation stops at the first check.
        self.assertTrue(out.guard.flagged)
        self.assertEqual(out.guard.prompt[0].label, UNPARSABLE_LABEL)
        self.assertLess(out.sequences.shape[-1] - ids.shape[-1], 8)
        del llm, guard
        gc.collect()

    def test_generative_guard_moderates_the_response_once_by_default(self):
        llm, guard = self._load_llm("llama"), self._load_generative_guard(guard_spec=ALWAYS_CLEARING_SPEC)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES["llama"])
        ids = self._prompt(llm)
        out = llm.generate(
            ids,
            max_new_tokens=8,
            do_sample=False,
            tokenizer=tokenizer,
            guard=guard,
            return_dict_in_generate=True,
        )
        self.assertFalse(out.guard.flagged)
        self.assertEqual(out.sequences.shape[-1] - ids.shape[-1], 8)
        self.assertEqual(len(out.guard.prompt), 1)
        self.assertEqual(len(out.guard.response), 1)
        del llm, guard
        gc.collect()

    def test_generative_guard_rechecks_every_chunk_on_request(self):
        llm, guard = self._load_llm("llama"), self._load_generative_guard(guard_spec=ALWAYS_CLEARING_SPEC)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES["llama"])
        out = llm.generate(
            self._prompt(llm),
            max_new_tokens=8,
            do_sample=False,
            tokenizer=tokenizer,
            guard=guard,
            guard_config=OVGuardConfig(chunk_size=4),
            return_dict_in_generate=True,
        )
        self.assertEqual(len(out.guard.response), 2)
        del llm, guard
        gc.collect()

    def test_generative_guard_covers_every_sequence_of_a_batch(self):
        llm, guard = self._load_llm("llama"), self._load_generative_guard(guard_spec=ALWAYS_CLEARING_SPEC)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES["llama"])
        ids = self._prompt(llm, batch_size=2)
        out = llm.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=4,
            do_sample=False,
            tokenizer=tokenizer,
            guard=guard,
            return_dict_in_generate=True,
        )
        self.assertEqual({verdict.batch_index for verdict in out.guard.response}, {0, 1})
        del llm, guard
        gc.collect()

    def test_generative_guard_requires_the_tokenizer_of_the_guarded_model(self):
        llm, guard = self._load_llm("llama"), self._load_generative_guard()
        with self.assertRaises(ValueError):
            llm.generate(self._prompt(llm), max_new_tokens=4, guard=guard)
        del llm, guard
        gc.collect()

    # --- device placement ----------------------------------------------------------------------

    def test_guard_runs_on_its_own_device(self):
        llm = self._load_llm()
        guard = OVGuard.from_pretrained(
            MODEL_NAMES["qwen3_guard"],
            task="token-classification-with-past",
            export=True,
            trust_remote_code=True,
            device="CPU",
        )
        self.assertEqual(guard.device, "CPU")
        self.assertEqual(llm._device, OPENVINO_DEVICE)
        out = llm.generate(
            self._prompt(llm),
            max_new_tokens=4,
            do_sample=False,
            guard=guard,
            guard_config=OVGuardConfig(blocking_labels=()),
            return_dict_in_generate=True,
        )
        self.assertEqual(len(out.guard.response), 4)
        del llm, guard
        gc.collect()
