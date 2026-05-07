---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- jailbreak-detection
- ai-safety
- paraphrase-robustness
- semantic-entropy
size_categories:
- n<1K
---

# JailbreakBench Paraphrase Dataset (2025-08)

This dataset contains 115 paraphrased prompts from the JailbreakBench dataset, created as part of research on semantic entropy-based jailbreak detection and its robustness to paraphrasing.

## Dataset Description

### Overview

This dataset was created to test the robustness of jailbreak detection methods (particularly semantic entropy) to paraphrased inputs. The paraphrases were generated from the original [JailbreakBench (JBB-Behaviors)](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) dataset.

### Research Context

This dataset was created as part of research investigating the "Consistency Confound" in semantic entropy-based jailbreak detection. The hypothesis being tested is whether paraphrasing can help validate semantic entropy methods against potential dataset contamination confounds, particularly for well-aligned models that produce consistent refusal patterns.

### Generation Process

We started with 120 of the 200 entries from the JailbreakBench dataset. Each prompt was paraphrased using Claude 3.7 Sonnet with a multi-layer validation pipeline. Of the 120 attempted paraphrases, 5 were refused by the model as they triggered safety training, resulting in this final dataset of 115 high-quality paraphrases.

The validation pipeline uses multiple methods to ensure quality: the R2J (Rewrite to Jailbreak) methodology from [github.com/ythuang02/R2J](https://github.com/ythuang02/R2J) for similarity checking with LLM-as-a-judge, intent preservation verification to ensure harmful/benign labels remain valid, embedding-based semantic similarity using GTE-large-en-v1.5, and refusal detection to filter out safety-triggered responses. This multi-layer approach ensures semantic equivalence between original and paraphrased prompts while maintaining different surface forms.

### Dataset Structure

Each row in `jbb_paraphrase_test.jsonl` contains the following columns:

- **`prompt_id`** (string): Unique identifier for the paraphrased prompt in format "jbb_XX_paraphrase"
- **`prompt`** (string): The paraphrased version of the original JailbreakBench prompt, rewritten to preserve semantic meaning while using different words and structure
- **`label`** (int): Classification label where 1=harmful prompt, 0=benign prompt, inherited from the original JBB dataset
- **`original_prompt_id`** (string): The ID of the original prompt from JailbreakBench (e.g., "jbb_37")
- **`original_prompt`** (string): The exact text of the original prompt from JailbreakBench before paraphrasing
- **`source_split`** (string): Indicates whether the original prompt came from the "test" or "validation" split of JBB
- **`paraphrase_metadata`** (object): Comprehensive metadata about the paraphrase generation and validation process, including:
  - `paraphrase_method`: The methodology used ("r2j_enhanced")
  - `model_used`: The model that generated the paraphrase ("anthropic/claude-3.7-sonnet")
  - `validation_pipeline`: Validation scores and checks applied
    - `r2j_similarity_score`: R2J similarity score (1-5 scale, all ≥4)
    - `intent_preserved`: Boolean confirming harmful/benign intent was maintained
    - `embedding_similarity`: Cosine similarity using GTE-large embeddings (all ≥0.7)
    - `refusal_filtered`: Boolean indicating if this was a refusal response (all false in final dataset)
  - `word_counts`: Word count comparison between original and paraphrase
  - `generation_timestamp`: Unix timestamp of when the paraphrase was generated

### Validation Pipeline

All paraphrases underwent multi-layer validation:
1. **R2J Similarity Score**: Used LLM-as-a-judge with R2J (Rewrite to Jailbreak) methodology to score semantic similarity on a 1-5 scale, requiring score ≥4 for acceptance
2. **Intent Preservation**: Verified harmful/benign intent was maintained between original and paraphrase
3. **Embedding Similarity**: Required ≥0.7 cosine similarity using GTE-large-en-v1.5 embeddings
4. **Refusal Filtering**: Removed any paraphrases that were refusal responses rather than actual paraphrases

## Dataset Statistics

- **Total Samples**: 115
- **Harmful Prompts**: 56
- **Benign Prompts**: 59
- **Average Paraphrase Quality**: R2J score 4.8/5
- **Average Embedding Similarity**: 0.82
- **Embedding Model**: Alibaba-NLP/gte-large-en-v1.5

## Intended Use

This dataset is intended for:
- Testing robustness of jailbreak detection methods to paraphrasing
- Evaluating semantic similarity preservation in adversarial prompt rewriting
- Research on AI safety and robust content moderation

## Citation

If you use this dataset, please cite both this paraphrase dataset and the original JailbreakBench:

```
@dataset{jailbreakbench_paraphrase_2025,
  title={JailbreakBench Paraphrase Dataset},
  author={Dhruv Trehan},
  year={2025},
  month={August},
  publisher={HuggingFace}
}

@article{chao2024jailbreakbench,
  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author={Chao, Patrick and Deserisy, Edoardo and Robey, Alexander and Xiao, Chuan and Gerchick, Reid and Vasileva, Vasilisa and Vijay, Yashwat and Yu, Hao and Dinh, Thomas Hartvigsen and others},
  journal={arXiv preprint arXiv:2404.01318},
  year={2024}
}
```

## License

This dataset is released under the MIT license. The original JailbreakBench dataset terms also apply.

## Ethical Considerations

This dataset contains examples of potentially harmful prompts for research purposes only. It should not be used to develop methods for generating harmful content or bypassing safety measures.