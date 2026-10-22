# Parent data and package provenance

The saved decoder used by this optional bridge is the parent inference package from `projects/text-harness/foundations/reference/run/inference-package/`. Its package manifest records the model configuration, the 10,880-parameter count, the saved parameter archive hash, the tokenizer hash, the parent source hashes, and the source dataset identity. See the [package manifest pinned to the parent source revision](https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/reference/run/inference-package/package.json) and the bridge's checksum-pinned `recipe.json` for the exact parent file pins.

The parent data source is Lewis Carroll's *Alice's Adventures in Wonderland*, identified as Project Gutenberg eBook 11, Millennium Fulcrum Edition 3.0. The [pinned source notice](https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/data/LICENSE.md) records the terms; its [pinned manifest](https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/data/manifest.json) records the retrieved plain-text file and SHA-256. The course retains the source header and Project Gutenberg terms in the [pinned Alice source](https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/data/source/alice-pg11.txt). The source manifest records chapters I through IV for tokenizer/model training, chapter V for validation, chapters VI through VIII for testing, and chapters IX through XII as excluded.

Official Project Gutenberg records:

- [eBook 11 metadata and formats](https://www.gutenberg.org/ebooks/11)
- [UTF-8 plain-text source recorded by the manifest](https://www.gutenberg.org/cache/epub/11/pg11.txt)

Project Gutenberg identifies this work as public domain in the United States and includes its terms with the text. Applicable rights can differ by country. Consult the retained source terms and local law before reuse. This note records provenance; it does not grant rights beyond those terms or make a rights determination for model parameters.

The parent archive is an inference-only package. The bridge starts a new actor from the parent parameters and does not reuse the parent's optimizer state. This document does not license or authorize release of a child checkpoint, child weights, or other generated artifacts. Their handling requires separate review and approval.
