from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


# CIT: statements now carry a `type` so the NLI judge can branch on disclaimers
# and citations. See docs/ragas-prompt-improvements-202604.md in macnica-autoeval.
class TypedStatement(BaseModel):
    text: str = Field(description="The statement, fully self-contained (no pronouns)")
    type: t.Literal["factual_claim", "disclaimer", "citation"] = Field(
        description=(
            "factual_claim = substantive assertion; "
            "disclaimer = explicit refusal/no-data/apology; "
            "citation = source reference (filename/page/section/URL)"
        )
    )


class StatementGeneratorOutput(BaseModel):
    statements: t.List[TypedStatement] = Field(description="The generated statements")


class StatementGeneratorPrompt(
    PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    instruction = (
        "Given a question and an answer, analyze each sentence in the answer. "
        "Break down each sentence into one or more fully self-contained statements "
        "(no pronouns; each statement understandable on its own).\n\n"
        "For each statement, ALSO classify its `type` as exactly one of:\n"
        "- `factual_claim` — the statement asserts a specific fact, number, "
        "procedure, recommendation, technical value, or instruction. Default "
        "type when nothing else fits.\n"
        "- `disclaimer` — the statement explicitly declines to answer, apologizes, "
        "or states that the information is absent, unknown, or outside scope. "
        "Disclaimer language (non-exhaustive, cross-language):\n"
        "  Japanese: 申し訳ございません / 申し訳ありません / データにありません / "
        "情報がありません / お答えできません / 見つかりませんでした / "
        "具体的なデータはございません / 恐れ入りますが / わかりません\n"
        "  English: \"I don't have information on X\", \"no data available\", "
        "\"I cannot find\", \"not in the provided context\", "
        "\"sorry, I'm unable to answer\".\n"
        "- `citation` — the statement is (or primarily contains) a source reference: "
        "a PDF/document filename, page number, section reference, or URL. "
        "Examples: \"詳細は r01ds0282ej0260-rl78g11.pdf の 105 ページを参照してください。\", "
        "\"See datasheet ug-813968-813969.pdf page 6.\", "
        "\"参照: RL78/G12 データシート, section 4.2\".\n\n"
        "If a single sentence mixes types (e.g. \"I don't have the exact value for X, "
        "but see foo.pdf page 3\"), split it into multiple statements of appropriate "
        "type.\n\n"
        "Format the output as JSON with the schema provided."
    )
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    TypedStatement(text="Albert Einstein was a German-born theoretical physicist.", type="factual_claim"),
                    TypedStatement(text="Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", type="factual_claim"),
                    TypedStatement(text="Albert Einstein was best known for developing the theory of relativity.", type="factual_claim"),
                    TypedStatement(text="Albert Einstein also made important contributions to the development of the theory of quantum mechanics.", type="factual_claim"),
                ]
            ),
        ),
        (
            StatementGeneratorInput(
                question="RL78/G11のタイマ出力の最小パルス幅は？",
                answer="申し訳ございません、RL78/G11の最小パルス幅に関する情報は提供されたデータにはありません。詳細は r01ds0282ej0260-rl78g11.pdf の 105 ページを参照してください。",
            ),
            StatementGeneratorOutput(
                statements=[
                    TypedStatement(text="RL78/G11の最小パルス幅に関する情報は提供されたデータにありません。", type="disclaimer"),
                    TypedStatement(text="詳細は r01ds0282ej0260-rl78g11.pdf の 105 ページを参照してください。", type="citation"),
                ]
            ),
        ),
    ]


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[TypedStatement] = Field(
        ...,
        description="The statements to judge (each with text + type)",
    )


class NLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = (
        "Your task is to judge the faithfulness of a series of statements against "
        "a given context. Each statement has a `type` — one of `factual_claim`, "
        "`disclaimer`, or `citation`.\n\n"
        "Each context may begin with a source header of the form "
        "`[Source: <filename>, page <N>]`. Treat this header as structured "
        "metadata about the chunk, not as chunk body text.\n\n"
        "For each statement, return verdict = 1 (faithful) or verdict = 0 "
        "(unfaithful) using the rule for its type:\n\n"
        "(a) `factual_claim` — return 1 if the claim can be directly inferred from "
        "the context body (ignore the `[Source: ...]` header for this rule); "
        "otherwise return 0. (Original RAGAS rule, unchanged.)\n\n"
        "(b) `disclaimer` — the statement asserts that specific data is absent or "
        "that an answer cannot be given. Apply the negative-evidence rule:\n"
        "  - Return 1 (faithful) if the context does NOT contain the data the "
        "disclaimer claims is absent. A correct refusal is not a hallucination — "
        "it is an accurate statement about the absence of knowledge in the "
        "retrieved material.\n"
        "  - Return 0 only if the context clearly DOES contain the data the answer "
        "claims is missing (the bot refused when it shouldn't have).\n\n"
        "(c) `citation` — the statement references a source (filename/page/section). "
        "Apply the source-match rule:\n"
        "  - Return 1 (faithful) if any retrieved context's `[Source: ...]` header "
        "matches the cited filename AND (if the citation names a page) the cited "
        "page number. The match establishes that the cited source was actually "
        "retrieved; the chunk body is not required to literally repeat the cited "
        "fact.\n"
        "  - Return 0 if no retrieved context's source header matches the citation.\n"
        "  - If no context carries a `[Source: ...]` header at all, fall back to "
        "rule (a): treat the citation as a factual_claim.\n\n"
        "In the `reason` field, cite which of rule (a), (b), or (c) was applied; "
        "for rule (b) state whether the context did or did not contain the "
        "claimed-absent data; for rule (c) state which context's source header "
        "matched."
    )
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
                statements=[
                    TypedStatement(text="John is majoring in Biology.", type="factual_claim"),
                    TypedStatement(text="John is taking a course on Artificial Intelligence.", type="factual_claim"),
                    TypedStatement(text="John is a dedicated student.", type="factual_claim"),
                    TypedStatement(text="John has a part-time job.", type="factual_claim"),
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information given in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                statements=[
                    TypedStatement(text="Albert Einstein was a genius.", type="factual_claim"),
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="Rule (a): factual_claim. The context and statement are unrelated.",
                        verdict=0,
                    )
                ]
            ),
        ),
        (
            NLIStatementInput(
                context=(
                    "[Source: r01ds0282ej0260-rl78g11.pdf, page 105]\n"
                    "RL78/G11 Timer output specifications. See the datasheet for "
                    "timer array unit output pulse width parameters."
                ),
                statements=[
                    TypedStatement(
                        text="RL78/G11の最小パルス幅に関する情報は提供されたデータにありません。",
                        type="disclaimer",
                    ),
                    TypedStatement(
                        text="詳細は r01ds0282ej0260-rl78g11.pdf の 105 ページを参照してください。",
                        type="citation",
                    ),
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="RL78/G11の最小パルス幅に関する情報は提供されたデータにありません。",
                        reason="Rule (b): disclaimer. The context references timer output specs generally but does NOT contain the specific minimum pulse width value the disclaimer claims is absent. Correct refusal, faithful.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="詳細は r01ds0282ej0260-rl78g11.pdf の 105 ページを参照してください。",
                        reason="Rule (c): citation. The context carries [Source: r01ds0282ej0260-rl78g11.pdf, page 105] which matches the cited filename and page. Faithful.",
                        verdict=1,
                    ),
                ]
            ),
        ),
    ]


@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    nli_statements_prompt: PydanticPrompt = field(default_factory=NLIStatementPrompt)
    statement_generator_prompt: PydanticPrompt = field(
        default_factory=StatementGeneratorPrompt
    )
    max_retries: int = 1

    async def _create_verdicts(
        self,
        row: t.Dict,
        statements: t.List[TypedStatement],
        callbacks: Callbacks,
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_prompt.generate(
            data=NLIStatementInput(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        text, question = row["response"], row["user_input"]

        prompt_input = StatementGeneratorInput(question=question, answer=text)
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )

        return statements

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(Faithfulness):
    name: str = "faithfulness_with_hhem"
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification  # type: ignore
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[t.Union[str, TypedStatement]]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (premise, statement_text) from the row.
        Accepts either List[str] (legacy) or List[TypedStatement] (CIT).
        """
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [
            (premise, s.text if isinstance(s, TypedStatement) else s)
            for s in statements
        ]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        scores = []
        pairs = self._create_pairs(row, statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            # convert tensor to list of floats
            scores.extend(batch_scores.tolist())

        return sum(scores) / len(scores)


faithfulness = Faithfulness()
