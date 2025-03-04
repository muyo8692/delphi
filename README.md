# Basic sanity check
- [ ] Different top-k setting should result in different score
- [ ] higher the top-k, lower the score
- [ ] extremely high top-k should result in extremely low score too
- [ ] scores are different from SAE and Transcoders
  


## top-k = 1%
```txt
--- Fuzz Metrics ---
Accuracy: 0.576
F1 Score: 0.493
Precision: 0.600
Recall: 0.430
Average fraction of failed examples: 0.000

Confusion Matrix:
True Positive Rate:  0.430
True Negative Rate:  0.721
False Positive Rate: 0.279
False Negative Rate: 0.570

Class Distribution:
Positives: 4250 (50.0%)
Negatives: 4250 (50.0%)
Total: 8500

--- Detection Metrics ---
Accuracy: 0.543
F1 Score: 0.635
Precision: 0.533
Recall: 0.805
Average fraction of failed examples: 0.000

Confusion Matrix:
True Positive Rate:  0.805
True Negative Rate:  0.280
False Positive Rate: 0.720
False Negative Rate: 0.195

Class Distribution:
Positives: 4250 (50.0%)
Negatives: 4250 (50.0%)
Total: 8500
MLP interpretation completed in 535.84 seconds
```


## top-k = 50%


## top-k = 99%