# Third-party notices

The experiment and reporting code in this project uses the MIT license in the repository root. Third-party models, source excerpts, and datasets retain their respective licenses.

| Use | Source and pinned revision | License and notes |
|---|---|---|
| Model and FP8 prompt source | [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062) | MIT; see [DeepSeek.LICENSE](sources/DeepSeek.LICENSE) |
| Second model | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/2cba9e42aa026125f3ed06c6d98c1db82f7ca027) | MIT; weights and tokenizers are downloaded during preparation |
| Shell prompts | [Omarchy lid-close-test.sh](https://github.com/basecamp/omarchy/blob/f99d33a8ddee7b36509a71a6d20d5d23355ce8b1/test/shell.d/lid-close-test.sh) | MIT, Copyright David Heinemeier Hansson; see [Omarchy.LICENSE](sources/Omarchy.LICENSE). The original file is retained locally; the generator selects a prefix and adds prompting text |
| SQL prompts | [Gretel synthetic_text_to_sql test split](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/tree/740ab236e64503fba51be1101df7a1be83bf455d) | Apache-2.0; see [SQL.LICENSE](sources/SQL.LICENSE). Only record 2113 (zero-based row 2112) is retained. The contrasting operator, two output values, and labels follow Appendix B.3.2; this project reconstructs the full layout, wrapper instruction, and fillers |
| Retrieval filler text | [agentlans/high-quality-english-sentences](https://huggingface.co/datasets/agentlans/high-quality-english-sentences/tree/5dfcb23b31fa9b85ce90b5713fcbba29179f2401) | The dataset declares the [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/). The original corpus is downloaded during preparation and is not distributed with the code |
| Protocol and paper reference values | [Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression](https://arxiv.org/abs/2609.36322v1) | arXiv:2609.36322v1. See README and paper_code_reference.json for prompt fidelity, reconstructed components, and reference tables |

`reports/reference/` contains observations from an earlier experiment in this project so that historical comparisons can be recomputed. It contains no official-backend code or converted weights. Other JSONL files in `reports/` contain observations exported with local absolute paths and redundant filler sentences removed. Experimental token strings and model outputs are retained verbatim.
