"""NB-08: the controlled experiment's corpus.

40 short, original, self-contained sentences (written for this
experiment -- not copied from any external source, so there is no
copyright/licensing question and no network dependency). Deliberately
NOT a single repeated pattern (unlike NB-07's "A B C D" toy corpus) --
real lexical and structural variety across sentences is the entire
point: it is what makes a genuine train/validation split possible
without the tokenizer-over-compression failure mode NB-07 uncovered
(see ``training/split.py``'s docstring and
``ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md`` for the full
reasoning).

Deliberately small (real byte count and tokenized counts are recorded
in the architecture doc, not guessed here) -- "small enough to run on
a normal developer machine" per this milestone's own requirement, not
"as large as possible".
"""

from __future__ import annotations

CORPUS_SENTENCES: list[str] = [
    "the cat sat on the mat.",
    "the dog ran in the park.",
    "a bird flew over the tall tree.",
    "the small dog chased the cat.",
    "a red car drove down the road.",
    "the sun rose over the quiet hill.",
    "two children played in the green field.",
    "the old man read a book by the window.",
    "a fast train crossed the long bridge.",
    "the fisherman caught a big fish.",
    "rain fell softly on the roof.",
    "the baker made fresh bread every morning.",
    "a young girl sang a happy song.",
    "the wind blew through the open door.",
    "birds flew south before the winter came.",
    "the farmer planted seeds in the spring.",
    "a black cat sat under the old oak tree.",
    "the teacher wrote on the white board.",
    "children laughed and ran across the yard.",
    "the moon rose slowly over the dark sea.",
    "a small boat floated on the calm river.",
    "the old dog slept by the warm fire.",
    "snow covered the quiet mountain village.",
    "the baker sold bread at the market.",
    "a gentle breeze moved through the tall grass.",
    "the boy threw a ball to his friend.",
    "the river flowed past the small town.",
    "stars appeared in the clear night sky.",
    "the girl picked flowers in the meadow.",
    "the train arrived at the busy station.",
    "a squirrel climbed the tall oak tree.",
    "the farmer fed the hungry chickens.",
    "the children built a sandcastle on the beach.",
    "a soft rain began to fall at noon.",
    "the old bridge crossed the narrow river.",
    "the cat chased a mouse through the barn.",
    "the fisherman rowed his boat to shore.",
    "leaves fell from the trees in autumn.",
    "the baker opened the shop at dawn.",
    "the dog barked at the passing car.",
]
