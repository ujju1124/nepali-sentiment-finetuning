# ⚡ Quick Start Guide

## 🎯 Goal
Run this notebook in Google Colab to get before/after sentiment analysis metrics for your portfolio.

---

## 🚀 5-Minute Setup

### Step 1: Upload to Colab
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click: **File → Upload notebook**
3. Upload: `nepali_sentiment_finetuning.ipynb`

### Step 2: Enable GPU
1. Click: **Runtime → Change runtime type**
2. Set **Hardware accelerator** to: **GPU**
3. Click: **Save**

### Step 3: Run Everything
1. Click: **Runtime → Run all** (or press `Ctrl+F9`)
2. Wait ~15-20 minutes (Colab will show progress bars)
3. Scroll to **Section 4** for your results

---

## 📊 What You'll Get

After running, you'll have:

### 1. Baseline Performance (Section 2)
```
📊 BASELINE PERFORMANCE (BEFORE FINE-TUNING)
Accuracy: XX.XX%
F1 Score: 0.XXXX
```

### 2. Fine-Tuned Performance (Section 4)
```
📊 FINE-TUNED MODEL PERFORMANCE
Accuracy: XX.XX%
F1 Score: 0.XXXX
```

### 3. Improvement Metrics (Section 4)
```
🎯 BASELINE vs FINE-TUNED COMPARISON
Accuracy:    +X.XX% improvement ✓
F1 Score:    +0.XXXX improvement ✓
```

---

## 💡 What to Do with Results

### For Your CV/Resume:
```
• Fine-tuned XLM-RoBERTa on Nepali sentiment analysis dataset
• Achieved [X%] accuracy, improving [Y%] over zero-shot baseline
• Implemented manual training loop with class-weighted loss
```

### For GitHub:
- Screenshot the comparison charts from Section 4
- Copy the metrics into your README
- Add example predictions from Section 6

### For Interviews:
**Talking points:**
1. "I chose XLM-RoBERTa because it's multilingual and handles Devanagari script"
2. "I used a manual training loop instead of Trainer API to show I understand the mechanics"
3. "I implemented class-weighted loss to handle potential imbalance"
4. "I performed error analysis to understand where the model still fails"

---

## ⏱️ Time Estimates

| Section | Time | What It Does |
|---------|------|--------------|
| Section 1 | ~2 min | Load & analyze data |
| Section 2 | ~1 min | Baseline evaluation |
| Section 3 | ~12 min | **Fine-tuning (3 epochs)** |
| Section 4 | ~1 min | Post fine-tuning eval |
| Section 5 | ~30 sec | Error analysis |
| Section 6 | ~10 sec | Inference demo |
| **TOTAL** | **~17 min** | Complete execution |

---

## 🔧 Common Issues

### Issue: "GPU not available"
**Fix**: Runtime → Change runtime type → GPU → Save

### Issue: "Out of memory"
**Fix**: Restart runtime and reduce batch size:
```python
# In Section 3, change:
train_dataloader = DataLoader(tokenized_train, batch_size=8, shuffle=True)  # Was 16
```

### Issue: "Dataset not found"
**Fix**: Verify internet connection. HuggingFace datasets require download.

### Issue: "Taking too long"
**Expected**: Fine-tuning (Section 3) takes 10-15 minutes. Be patient!

---

## 📸 Screenshots to Capture

For your portfolio, screenshot these sections:

1. ✅ **Section 1**: Class distribution bar charts
2. ✅ **Section 2**: Baseline confusion matrix
3. ✅ **Section 3**: Training loss curve
4. ✅ **Section 4**: Comparison bar charts (accuracy & F1)
5. ✅ **Section 5**: Error analysis examples
6. ✅ **Section 6**: Inference function demo

---

## 🎓 Next Steps After Running

1. **Copy your metrics** into README.md
2. **Save the trained model** (it's in `./nepali_sentiment_model`)
3. **Test custom sentences** using the `predict_sentiment()` function
4. **Share on LinkedIn** with your improvement metrics
5. **Push to GitHub** with results in README

---

## 📋 Checklist

Before calling this project "done":

- [ ] Notebook runs without errors
- [ ] GPU was enabled during training
- [ ] All 6 sections completed successfully
- [ ] Metrics show improvement (fine-tuned > baseline)
- [ ] Screenshots captured for portfolio
- [ ] Results documented in README
- [ ] Code pushed to GitHub
- [ ] Project added to resume/portfolio

---

## 🆘 Need Help?

**Colab Resources:**
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Using GPU in Colab](https://colab.research.google.com/notebooks/gpu.ipynb)

**HuggingFace Docs:**
- [XLM-RoBERTa Model Card](https://huggingface.co/xlm-roberta-base)
- [Datasets Library](https://huggingface.co/docs/datasets/)

**PyTorch Docs:**
- [Training Loop Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [DataLoader Guide](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

**⭐ Remember**: The goal isn't just to run the code, but to **understand** what each section does and **why** it matters. That's what impresses interviewers!
