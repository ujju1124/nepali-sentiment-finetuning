# 🇳🇵 Nepali Sentiment Analysis Fine-Tuning

A production-ready portfolio project demonstrating measurable improvement in sentiment analysis through fine-tuning XLM-RoBERTa on Nepali text.

## 📊 Quick Results

| Metric | Baseline (Zero-Shot) | Fine-Tuned | Improvement |
|--------|---------------------|------------|-------------|
| Accuracy | Run to see | Run to see | Run to see |
| F1 Score | Run to see | Run to see | Run to see |

> **Note**: Numbers above will be populated after running the notebook in Google Colab

---

## 🎯 Project Highlights

This notebook demonstrates:

✅ **Before/After Comparison**: Clear baseline → fine-tuned metrics  
✅ **Manual Training Loop**: Shows understanding of training mechanics (not just `.train()`)  
✅ **Class Imbalance Handling**: Weighted loss function for imbalanced datasets  
✅ **Honest Error Analysis**: What improved AND what still fails  
✅ **Production-Ready Code**: Reusable inference function  
✅ **Interview-Ready Comments**: Every section explains "why", not just "what"

---

## 📁 Project Structure

```
nepali-sentiment-finetuning/
│
├── nepali_sentiment_finetuning.ipynb  # Main notebook (ready for Colab)
├── README.md                           # This file
├── create_complete_notebook.py        # Generator script (optional)
└── generate_notebook.py               # Alternative generator (optional)
```

---

## 🚀 How to Use

### Option 1: Google Colab (Recommended)

1. **Upload the notebook** to [Google Colab](https://colab.research.google.com/)
2. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator: GPU (T4)
3. **Run all cells**: Runtime → Run all
4. **Wait ~15-20 minutes** for complete execution (3 epochs of fine-tuning)

### Option 2: Local Jupyter

```bash
# Install dependencies
pip install transformers datasets torch scikit-learn seaborn matplotlib accelerate

# Launch Jupyter
jupyter notebook nepali_sentiment_finetuning.ipynb
```

**Requirements**: CUDA-capable GPU with 8GB+ VRAM (T4, V100, etc.)

---

## 📝 Notebook Sections

### Section 1: Setup & Data Exploration
- Install packages
- Load IRIIS-RESEARCH/Sentiment-Analysis-Nepali dataset
- **Class distribution analysis** (critical for understanding bias)
- Sample data inspection (see actual Nepali text)
- Data quality checks (nulls, duplicates, text length)

### Section 2: Baseline Evaluation (BEFORE)
- Load pre-trained XLM-RoBERTa (zero-shot)
- Run inference on 200 random test examples
- Calculate accuracy, F1, confusion matrix
- **Store baseline numbers for comparison**

### Section 3: Fine-Tuning (Manual Loop)
- Tokenize dataset
- Calculate class weights for imbalanced data
- **Manual training loop** (forward pass → loss → backward → optimize)
- 3 epochs, batch_size=16, AdamW optimizer
- Save fine-tuned model

### Section 4: Evaluation (AFTER)
- Run inference on **same 200 test examples**
- Calculate fine-tuned metrics
- **Side-by-side comparison**: Baseline vs Fine-Tuned
- Visualize improvement

### Section 5: Error Analysis
- Find examples where fine-tuning **fixed** predictions
- Find examples where model **still fails**
- Honest discussion of limitations
- Error pattern analysis (FP vs FN)

### Section 6: Inference Function
- Simple `predict_sentiment(text)` API
- Returns: sentiment + confidence score
- Test on example Nepali sentences
- **Production-ready** for deployment

---

## 🔑 Key Technical Decisions

### Why XLM-RoBERTa?
- **Multilingual**: Trained on 100+ languages including Nepali
- **Devanagari Support**: Native handling of Nepali script
- **Strong Baseline**: Proven performance on cross-lingual tasks

### Why Manual Training Loop?
Shows understanding of:
- Forward pass computation
- Loss calculation with class weights
- Gradient backpropagation
- Optimizer weight updates

**Interview Impact**: Demonstrates you understand ML fundamentals, not just library APIs.

### Why 3 Epochs?
- **Fast enough** for Colab T4 GPU (~15 minutes)
- **Sufficient** to demonstrate improvement
- **Avoids overfitting** on small dataset

### Why Class-Weighted Loss?
Handles potential imbalance between positive/negative samples. Without weighting, model could bias toward majority class.

---

## 📈 Expected Performance

Based on similar multilingual sentiment tasks:

- **Baseline (Zero-Shot)**: 50-60% accuracy (random guessing baseline)
- **After Fine-Tuning**: 75-85% accuracy (dataset-dependent)
- **Improvement**: +20-30 percentage points

*Actual numbers depend on dataset quality and class balance.*

---

## 🛠️ Troubleshooting

### GPU Out of Memory (OOM)
```python
# Reduce batch sizes in training cell:
train_dataloader = DataLoader(tokenized_train, batch_size=8, shuffle=True)  # Was 16
```

### Slow Training
```python
# Reduce epochs or use smaller dataset sample:
NUM_EPOCHS = 2  # Instead of 3
```

### Dataset Load Error
```bash
# Verify HuggingFace datasets installation:
pip install --upgrade datasets transformers
```

---

## 📚 Dataset Information

**Source**: [IRIIS-RESEARCH/Sentiment-Analysis-Nepali](https://huggingface.co/datasets/IRIIS-RESEARCH/Sentiment-Analysis-Nepali)

**Format**:
- `text`: Nepali text string (Devanagari script)
- `label`: 0 (negative) or 1 (positive)

**Size**: Check Section 1 output after running notebook

---

## 🎓 Learning Outcomes

After completing this notebook, you'll understand:

1. ✅ How to evaluate baseline performance before training
2. ✅ Manual training loop implementation
3. ✅ Handling class imbalance with weighted loss
4. ✅ Proper train/test evaluation methodology
5. ✅ Error analysis and model limitations
6. ✅ Production-ready inference functions

---

## 💼 Portfolio Use

This notebook is designed for:
- **GitHub repositories** (showcase ML skills)
- **Resume projects** (tangible results with numbers)
- **Interview discussions** (explains reasoning, not just code)
- **Blog posts** (complete story with before/after)

**Key Talking Points**:
1. "I fine-tuned XLM-RoBERTa on Nepali sentiment data and improved accuracy by X%"
2. "Used manual training loop instead of Trainer API to demonstrate understanding"
3. "Implemented class-weighted loss to handle dataset imbalance"
4. "Performed error analysis to understand model limitations"

---

## 🔧 Customization

Want to adapt this for other languages/tasks?

```python
# Change dataset:
dataset = load_dataset("your-dataset-name")

# Change model:
MODEL_NAME = "bert-base-multilingual-cased"  # Or any HuggingFace model

# Adjust hyperparameters:
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
```

---

## 📄 License

This notebook is provided as-is for educational and portfolio purposes.

**Dataset License**: Check [IRIIS-RESEARCH](https://huggingface.co/datasets/IRIIS-RESEARCH/Sentiment-Analysis-Nepali) dataset page  
**Model License**: XLM-RoBERTa uses Apache 2.0 license

---

## 🤝 Contributing

Found a bug or want to improve this notebook?
- Open an issue
- Submit a pull request
- Share your results!

---

## 📞 Contact

**Author**: [Your Name]  
**Email**: [Your Email]  
**LinkedIn**: [Your LinkedIn]  
**Portfolio**: [Your Website]

---

## 🙏 Acknowledgments

- **IRIIS-RESEARCH** for the Nepali Sentiment dataset
- **HuggingFace** for transformers library and XLM-RoBERTa model
- **Google Colab** for free GPU access

---

**⭐ If this helped you, please star the repository!**
