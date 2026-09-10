"""
The prompt conditions.

The design crosses two factors.

**Baseline** (three levels). Each is a published prompt, reproduced as its
source states it, and each states a different number of requirements. Taking
three rather than one tests whether a wording effect survives a change in how
much the prompt asks for, which a single baseline cannot show.

    A  ragbench   Friel et al. 2024,  1 requirement
    B  alce       Gao et al. 2023,  ~10 requirements
    C  fitrag     Mao et al. 2024,   3 requirements

**Wording** (four levels), the axes of Section 4.5 of the thesis. Within a
baseline, all four conditions state the *same* requirement set. Only how it is
written changes. That is the independent variable of the experiment: the
structure of the prompt, with its meaning held fixed.

    0  canonical   the published form, unchanged
    1  lexical     synonym substitution, clause order and voice held fixed
    2  syntactic   clause reordering, active to passive, vocabulary held fixed
    3  format      the same wording laid out as a list rather than as prose

Condition ids are ``<baseline><n>``: ``A0``, ``B3``, ``C2`` and so on.

Writing the variants
--------------------

Each axis changes one thing and holds the rest still. That discipline is what
makes the comparison interpretable, and it is easy to break by accident:

- **lexical** substitutes words but keeps clause order and voice. It must not
  reorder anything.
- **syntactic** keeps the vocabulary as far as English allows and changes
  structure. Obligation is carried uniformly by "is to be", so that a change of
  voice does not smuggle in a change of modal force.
- **format** keeps the wording and changes only layout. List items are bare
  noun phrases, again so that no modal is added or removed.

No condition changes what the prompt requires. A variant that weakened or
strengthened an obligation would change the requirement set, and a difference
it produced could not be read as an effect of structure. Such a variant is
therefore not part of the design.

Length is held near-constant within a baseline (``LENGTH_TOLERANCE``), so a
difference between two conditions cannot be read as an effect of how much
text the prompt contains.
"""

#: Patterns that mark a refusal to answer.
#:
#: None of the three published baselines mandates a fixed abstention string, so
#: unlike a prompt written for this experiment there is no exact phrase to match.
#: Declining is therefore detected by pattern rather than by equality, which is
#: looser: a paraphrase that this list misses is scored as an attempt to answer.
#: The list is deliberately narrow, so its hits are more trustworthy than its
#: misses, and it is used only for reading abstention in the relevance measure,
#: never as an adherence requirement that no baseline states.
DECLINE_PATTERNS = (
    r"\bdo(?:es)? not contain\b",
    r"\bcannot be answered\b",
    r"\bcan(?:no|')t answer\b",
    r"\bno (?:relevant )?information\b",
    r"\bnot (?:enough|sufficient) information\b",
    r"\bunable to answer\b",
    r"\bnot (?:provided|available) in the (?:passages|context|documents)\b",
    r"\bdon't know\b",
)


#: what each baseline's requirement set contains, for rule-based adherence
SIGMA = {
    "A": {"ground": "answer from the supplied context"},
    "B": {"ground": "use only the provided search results",
          "cite": "cite every factual claim, one to three documents per sentence",
          "concision": "keep the answer concise",
          "register": "unbiased, journalistic tone"},
    "C": {"ground": "answer from the passage below",
          "comprehend": "understand the question and passages first",
          "justify": "explain why the answer was chosen"},
}

BASELINE_SOURCE = {
    "A": "friel2024ragbench",
    "B": "gao2023alce",
    "C": "mao2024fitrag",
}

AXES = {
    0: "canonical", 1: "lexical", 2: "syntactic", 3: "format",
}

# --------------------------------------------------------------- baseline A
A0 = "Use the following pieces of context to answer the question."

A1 = "Employ the following excerpts of material to address the query."

A2 = "The question is to be answered using the context pieces below."

A3 = """CONTEXT: the following pieces
TASK: use them to answer the question"""

# --------------------------------------------------------------- baseline B
B0 = """Write an accurate, engaging, and concise answer for the given question \
using only the provided search results (some of which might be irrelevant) and \
cite them properly. Use an unbiased and journalistic tone. Always cite for any \
factual claim. When citing several search results, use [1][2][3]. Cite at least \
one document and at most three documents in each sentence. If multiple \
documents support the sentence, only cite a minimum sufficient subset of the \
documents."""

B1 = """Compose a precise, compelling, and succinct reply for the posed query \
drawing only on the supplied search results (a number of which may be \
immaterial) and reference them correctly. Adopt an impartial and journalistic \
register. Invariably reference every factual assertion. When referencing \
several search results, employ [1][2][3]. Reference at minimum one document \
and at maximum three documents per sentence. If several documents corroborate \
the sentence, only reference a minimal adequate subset."""

B2 = """For the given question an accurate, engaging, and concise answer is to \
be written from the provided search results alone, some of which might be \
irrelevant, and cited properly. An unbiased, journalistic tone is to be used. \
A citation is always given for any factual claim. Where several results are \
cited, [1][2][3] is used. In each sentence at least one and at most three \
documents are cited. Where multiple documents support a sentence, only a \
minimum sufficient subset is cited."""

B3 = """ANSWER
- accurate, engaging, and concise, for the given question
- source: only the provided search results
- note: some of the provided results might be irrelevant

CITATIONS
- cite the search results properly
- always cite for any factual claim
- when citing several search results: [1][2][3]
- in each sentence: at least one document, at most three documents
- if multiple documents support the sentence: only a minimum sufficient subset

TONE
- unbiased and journalistic"""

# --------------------------------------------------------------- baseline C
C0 = """Refer to the passage below and answer the following question.
Make sure you fully understand the meaning of the question and passages.
Then give the answer and explain why you choose this answer."""

C1 = """Consult the excerpt below and respond to the following query.
Ensure you completely grasp the sense of the query and excerpts.
Then supply the response and justify why you select this response."""

C2 = """The following question is to be answered from the passage below.
The meaning of the question and passages is first understood.
The answer is then given, with an explanation of why it is chosen."""

C3 = """SOURCE: the passage below, and nothing else
READ: fully understand the meaning of the question and passages
ANSWER: give the answer to the following question
EXPLAIN: say why you choose this answer"""

#: condition id -> system prompt
CONDITIONS = {
    "A0": A0, "A1": A1, "A2": A2, "A3": A3,
    "B0": B0, "B1": B1, "B2": B2, "B3": B3,
    "C0": C0, "C1": C1, "C2": C2, "C3": C3,
}

#: the baseline each condition belongs to
BASELINE_OF = {name: name[0] for name in CONDITIONS}

#: the wording axis each condition applies
AXIS_OF = {name: AXES[int(name[1])] for name in CONDITIONS}

#: the canonical condition of each baseline, the reference its variants move from
CANONICAL = {"A": "A0", "B": "B0", "C": "C0"}


def describe(name):
    """One line naming the baseline, its source and the axis varied."""
    baseline = BASELINE_OF[name]
    return (f"{name}: {BASELINE_SOURCE[baseline]} baseline, "
            f"{AXIS_OF[name]} wording")


def by_baseline(baseline):
    """Every condition id belonging to one baseline, canonical first."""
    return [n for n in CONDITIONS if BASELINE_OF[n] == baseline]


#: how far a variant's length may stray from its canonical, as a fraction of
#: the canonical word count. Length is held near-constant so that a difference
#: between two conditions cannot be read as an effect of prompt length rather
#: than of the axis being varied. Section 5.3 states the bound as one tenth.
LENGTH_TOLERANCE = 0.10


def word_counts(baseline):
    """Word count of every condition in one baseline, canonical first."""
    return {n: len(CONDITIONS[n].split()) for n in by_baseline(baseline)}


def length_deviations(baseline):
    """Each condition's word count as a signed fraction of the canonical's."""
    counts = word_counts(baseline)
    reference = counts[CANONICAL[baseline]]
    return {n: (c - reference) / reference for n, c in counts.items()}


def check_length_balance(tolerance=LENGTH_TOLERANCE):
    """Raise if any variant has drifted away from its canonical in length.

    Called at import so an edit to a prompt string cannot silently reintroduce
    length as a confound.
    """
    offenders = []
    for baseline in CANONICAL:
        for name, deviation in length_deviations(baseline).items():
            if abs(deviation) > tolerance + 1e-9:
                offenders.append(f"{name} {deviation:+.0%}")
    if offenders:
        raise ValueError(
            "prompt length deviates from the canonical by more than "
            f"{tolerance:.0%}: {', '.join(offenders)}"
        )


check_length_balance()
