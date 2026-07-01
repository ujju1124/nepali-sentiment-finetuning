# Nepali Sentiment Analysis: Fine-Tuning XLM-RoBERTa

Fine-tuned XLM-RoBERTa on Nepali sentiment analysis, achieving 88.5% accuracy (38 percentage point improvement over zero-shot baseline) using manual training loop with gradient accumulation and warmup scheduling.

## Results

| Metric | Baseline (Zero-Shot) | Fine-Tuned | Improvement |
|--------|---------------------|------------|-------------|
| **Accuracy** | 50.50% | **88.50%** | **+38.0%** |
| **F1 Score** | 0.00 | **0.88** | **+0.88** |
| **Error Rate** | 49.50% | 11.50% | -38.0% |

**Training:** 2 epochs, loss decreased from 0.5113 → 0.3324  
**Evaluation:** 200 random test samples, confusion matrix [[90, 11], [12, 87]]  
**Dataset:** 65,106 train / 16,279 test examples from IRIIS-RESEARCH/Sentiment-Analysis-Nepali

## Why This Project

Nepali NLP is underserved despite 16M+ native speakers. This project demonstrates:

1. **Manual training loop implementation** — shows understanding of forward/backward passes, gradient accumulation, and optimization mechanics rather than just calling `.train()`
2. **Principled debugging** — baseline model predicted all negatives (F1=0.00), fixed through proper loss function and warmup scheduling
3. **Honest evaluation** — includes error analysis showing failure modes (ambiguous short texts, mixed sentiment in long passages)

## How to Run

**Requirements:** Google Colab with T4 GPU (free tier works)

1. Upload `nepali_sentiment_finetuning.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → GPU (T4)
3. Runtime → Run all
4. Wait ~50 minutes for 2 epochs of training

**Local Setup:**
```bash
pip install transformers datasets torch scikit-learn seaborn matplotlib accelerate
jupyter notebook nepali_sentiment_finetuning.ipynb
```
Requires CUDA GPU with 8GB+ VRAM.

## Notebook Structure

1. **Setup & Data**: Load dataset, analyze 49/51 class balance, check for nulls/duplicates
2. **Baseline Evaluation**: Zero-shot XLM-RoBERTa (predicts all negative, 50.5% accuracy)
3. **Fine-Tuning**: Manual training loop with gradient accumulation (4 steps) and warmup (10%)
4. **Post Fine-Tuning Evaluation**: 88.5% accuracy on identical 200 test samples
5. **Error Analysis**: 23 failures — short ambiguous texts and mixed-sentiment passages
6. **Inference Function**: `predict_sentiment(text)` returns label + confidence (99%+ on clear examples)

## Technical Decisions

**XLM-RoBERTa:**  
Multilingual transformer trained on 100+ languages including Nepali. Native Devanagari script support, proven cross-lingual performance.

**Manual Training Loop:**  
Explicit forward pass → loss calculation → backward pass → optimizer step. Shows understanding of training mechanics beyond high-level APIs.

**No Class Weighting:**  
Dataset is balanced (49.25% negative / 50.75% positive). Initial class weighting caused gradient interference. Removed after diagnosis.

**Warmup Scheduler:**  
Linear warmup over 10% of training prevents early gradient collapse. Learning rate ramps from 0 → 2e-5 over first ~400 steps.

**Gradient Accumulation:**  
Accumulate over 4 steps → effective batch size 64 instead of 16. Stabilizes training on balanced dataset without memory overflow.

**Gradient Clipping:**  
Max norm 1.0 prevents exploding gradients during warmup phase.

## Limitations

**Domain Specificity:**  
Model trained on COVID-19 news headlines from 2020-2021. May not generalize well to other domains (social media, product reviews, literature).

**Failure Modes:**  
- Short texts (< 10 words): lack context for disambiguation
- Long texts (> 100 words): mixed sentiment signals hard to aggregate
- Implicit sentiment: factual news statements where tone is implied not explicit

**Dataset Size:**  
65K training examples sufficient for this domain but low-resource compared to English sentiment corpora (millions of examples).

**Evaluation:**  
200 test samples used for comparison (computational efficiency). Full test set (16K examples) not evaluated. To evaluate on the full test set, replace `test_sample_df` with `test_df` in Section 4.

## Implementation Details

**Hyperparameters:**
- Learning rate: 2e-5 (AdamW)
- Batch size: 16 (effective 64 with accumulation)
- Epochs: 2
- Max sequence length: 128 tokens
- Warmup: 10% of total steps
- Gradient clipping: 1.0

**Infrastructure:**
- Google Colab T4 GPU (15GB VRAM)
- Training time: ~50 minutes
- Model size: 278M parameters
- Peak memory: ~5GB

## Files

- `nepali_sentiment_finetuning.ipynb` — Executed notebook with results
- `README.md` — This file

## Contact

**Ujwal Dahal**  
Email: dahalujwal3@gmail.com  
Portfolio: [ujwaldahal.com.np](https://ujwaldahal.com.np)  
GitHub: [@ujju1124](https://github.com/ujju1124)
