# Grammar Scoring Engine for Spoken Audio (SHL challenge)

Predicts a MOS grammar score (0-5) from 45-60 s speech clips.

## Approach
1. Whisper (small) transcribes each clip and gives confidence and timing signals.
2. Features: LanguageTool error counts, GPT-2 perplexity, fluency and pause stats, ASR confidence, sentence-transformer embedding of the transcript, wav2vec2 and Whisper-encoder audio embeddings (PCA to 40 dims each).
3. Models: Ridge on the transcript embedding + 3-seed LightGBM on handcrafted + audio features, blended with a weight chosen on out-of-fold predictions.

## Results
| Model | CV RMSE | CV Pearson |
|---|---|---|
| Baseline | 0.7241 | 0.8156 |
| + wav2vec2 | 0.6067 | 0.8736 |
| + Whisper encoder (final) | 0.5717 | 0.8895 |

- Training RMSE (final model): 0.1893
- Public leaderboard (RMSE, ~60% of test): 0.4417

## Files
- `notebook.ipynb`: full pipeline, run top to bottom on Kaggle (GPU, Internet on)
- `submission.csv`: final predictions for the 216 test clips

## How to run
Open the notebook on Kaggle with the competition data attached and Internet enabled, then Run All.