# Plagiarism evaluation (n=90, semantic: unavailable (lexical only))

| signal | auc | accuracy | precision | recall | f1 | fpr | tpr_at_1pct_fpr | best_f1_threshold |
|---|---|---|---|---|---|---|---|---|
| lexical (TF-IDF+difflib) | 0.772 | 0.7778 | 1.0 | 0.6 | 0.75 | 0.0 | 0.6 | 0.703 |
| semantic (SBERT) | 0.5 | 0.4444 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.0 |
| combined | 0.772 | 0.7778 | 1.0 | 0.6 | 0.75 | 0.0 | 0.6 | 0.703 |

## Recall by plagiarism type

| type | lexical (TF-IDF+difflib) | semantic (SBERT) | combined |
|---|---|---|---|
| handwritten_paraphrase | 0.0 | 0.0 | 0.0 |
| light_edit | 1.0 | 0.0 | 1.0 |
| paraphrase | 1.0 | 0.0 | 1.0 |
| verbatim | 1.0 | 0.0 | 1.0 |