# C02 source attribution and rights

The bounded pilot uses two public educational datasets. The raw files are
downloaded into an external prepared-data directory, checked against pinned
revisions and SHA-256 values, and are not copied into the project repository.

**WikiText-2 raw (`wikitext-2-raw-v1`)** comes from Salesforce's
[WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The card lists CC BY-SA 3.0
and GFDL. WikiText was assembled from verified Wikipedia articles. Attribute
Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher and retain the
source card and revision in `source-manifest.json`.

**SQuAD 1.1** comes from the authors' `rajpurkar/SQuAD-explorer` dataset files,
revision `eee5fdbf62f8613a7812b03419e6b29617b74fd1`. The SQuAD authors identify
CC BY-SA 4.0 for the question-answer dataset. Attribute Pranav Rajpurkar, Jian
Zhang, Konstantin Lopyrev and Percy Liang. Passage text is from Wikipedia;
questions and answer spans were supplied by human annotators. The source site
is [SQuAD Explorer](https://rajpurkar.github.io/SQuAD-explorer/).

Share-alike terms remain attached to their respective inputs and derivatives.
This metadata does not determine the license of trained weights. The filtered
SQuAD final set is a custom protocol, not an official leaderboard submission.
The project does not redistribute the downloaded source files or trained
weights. Users must review the original terms and local law before reuse.

The contextual pretrained comparison is EleutherAI/Pythia-14M at the immutable
revision recorded in `frozen_recipe.json`. Its author card identifies Apache
2.0 and research on model behavior, functionality and limitations as intended
use. The card says the Pythia suite is not intended for deployment or
human-facing interaction and warns that it may generate harmful or offensive
text. We keep its downloaded model files outside this project and do not treat
it as an instruction-tuned or contamination-free baseline; Pythia's Pile
training may overlap with WikiText or SQuAD. The pinned card and required model
and tokenizer file digests are retained in the recipe.
