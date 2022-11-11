---
    title : "Gaussian Model Introduction"
    author: "Dhruva Mambapoor"
---
# Definitions
---
## Error Terminology
---

### Ground-truth
> The reality.
> For example, if a client gets hospitalized in real life. Then the ground-truth is that the client get hospitalized.
> When testing the data, our model should predict as close to the ground-truth as possible.


### Error Table
|                                | Actually Hospitalized | Actually NOT-hospitalized |
|--------------------------------|-------------------------|-----------------------------|
| ***Predicted* Hospitalization**    | True Positive           | False Positive              |
| ***Predicted* NO-Hospitalization** | False Negative          | True Negative               |

---
## Metrics
---
We will be using TN, FN, TP, FP to mean True Negative, False Negative, True Positive, and False Positive respectively.

### Precision
Precision = $ \frac{TP}{TP + FP}$
> i.e. If we predicted Positives, what percentage did we get right.
### Recall
Recall = $ \frac{TP}{TP + FN}$
> i.e. What percentage of the ground-truth positives did we predict correctly.
### F1-Score
F1-Score = $ \frac{2*Precision*Recall}{Precision + Recall}$
> i.e. The average (*harmonic mean*) of precision and recall.
