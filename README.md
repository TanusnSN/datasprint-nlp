# Data Sprint 2.0: NLP Track (Hindi & Hinglish Open-Set Intent Classification)

Team: YT(Team Name),  Tanushree S N and Yashaswika  K M, The National Institute of Engineering, Mysuru

## Pipeline overview
1. **multilingual-e5-base** (278M) fine-tuned for 56-way classification: mean pooling, dropout, linear head, label smoothing 0.1, AdamW (lr 5e-5), fp16, 6 epochs, trained on 100% of train.csv.
2. **Character n-gram TF-IDF + LinearSVC**: char_wb, n-grams 2 to 5 (4.2M parameters).
3. **Blend**: 0.6 x softmax(transformer) + 0.4 x softmax(12 x SVC scores).
The learning rate, epochs, blend weight and temperature were chosen in an earlier experiment on a stratified 80/20 split of train.csv (tune_on_validation.py). The final model was then retrained on all of train.csv (train.py).

## Cross-script (Hindi <-> Hinglish) and Class 55 handling
- The multilingual encoder reads Devanagari and Roman-script (Hinglish) text.
- Character n-grams absorb spelling variants and ASR-style typos.
- Class 55 (Out-of-Scope) is a trained class using the labelled out-of-scope examples in train.csv. No confidence threshold is used.
- No external data, no paid APIs, no test-label lookups.

## Model parameter count (inference)
| Component | Parameters |
|---|---|
| multilingual-e5-base + classification head | 278,086,712 |
| TF-IDF LinearSVC (coef + intercept) | 4,229,232 |
| **Total** | **282,315,944** (<= 300,000,000) |

train.py and app.py both contain an assert that checks the total.

## Local CV score and validation summary
Experiment on a stratified 20% hold-out (models trained on the other 80%), Macro-F1:

| Model | Macro-F1 |
|---|---|
| TF-IDF + LinearSVC | 0.784 |
| e5-small | 0.797 |
| e5-base | 0.821 |
| e5-base + TF-IDF blend | 0.831 |
| Tuned blend (T=12, w=0.6) | 0.837 |

Kaggle public leaderboard: **0.82302** for the final model (trained on 100% of train.csv) and 0.82205 for the same pipeline trained on 80%. The validation split is about 0.02 more optimistic than Kaggle, so the Kaggle score is the figure to trust. The validation run used 8 epochs; the score had plateaued by about epoch 6, so the final model uses 6.

## Inference latency
Average latency per query: ___ ms on laptop CPU. Measured with benchmark_latency.py.

## Reproduce the final submission CSV
pip install -r requirements.txt
python train.py --data /path/to/folder_with_train_and_test_csv

- Needs a GPU. Fixed seed 42; GPU runs can still differ by about +/-0.005 Macro-F1.
- Outputs: output/submission_final.csv and output/model/ (weights are not stored in this repo).
- submission_final.csv in this repo is the file that scored 0.82302 on Kaggle.
- To reproduce the validation table and the blend tuning: python tune_on_validation.py

## Run the demo (Gradio UI)
python app.py
Needs the model/ folder next to app.py. Local demo with screen share during judging.
