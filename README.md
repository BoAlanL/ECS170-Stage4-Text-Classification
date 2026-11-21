# Stage 4 Text Classification

IMDb binary sentiment classification with RNN, LSTM, and GRU. This project extends
the instructor's ECS170 Fall 2026 source-code template. The original Python files,
base classes, and Stage 1 examples are preserved. Stage 4 was an empty directory
in the supplied template. New classification code uses the provided Dataset,
Method, Setting, Result, and Evaluate interfaces. Text generation is outside scope.

## Run

Use Python 3.10 or compatible and install the dependencies in requirements.txt.
Install a CUDA-enabled PyTorch build when using an NVIDIA GPU; CPU is supported.
Keep the instructor's dataset outside this repository and pass its path:

```sh
python script/stage_4_script/script_text_classification.py --data "PATH/TO/stage_4_data.zip"
```

The data argument also accepts the extracted text_classification directory.
By default, output is saved to result/stage_4_result. To avoid overwriting a prior
run, pass --output "PATH/TO/NEW_RESULTS". Add --device cpu for CPU execution.

Predict from a checkpoint and the vocab.json saved alongside it:

```sh
python script/stage_4_script/script_text_classification.py --checkpoint "PATH/TO/GRU_h128.pt" --text "I enjoyed this movie."
```

## Implementation

- local_code/stage_4_code/Dataset_Loader.py: loading, tokenization, and encoding.
- local_code/stage_4_code/Method_RNN.py: custom classifier and Method adapter.
- local_code/stage_4_code/experiment.py: split, training, selection, and inference.
- local_code/stage_4_code/Setting_Train_Validation_Test.py: template workflow.
- local_code/stage_4_code/Result_Saver.py: summary persistence.
- local_code/stage_4_code/Evaluate_Metrics.py: evaluation metrics.
- script/stage_4_script/script_text_classification.py: runnable entry point.

All four predefined architectures use trainable 128-dimensional embeddings, one
forward recurrent layer, masked mean pooling, dropout 0.3, Adam, and early stopping.
The comparison uses RNN/LSTM/GRU with hidden size 128 plus a GRU hidden-size-64
ablation. Vocabulary is fit only on the 20,000-review fitting subset; 5,000 reviews
are used for validation and 25,000 official test reviews for final evaluation.
Lowest validation loss selects the checkpoint and preferred model, independently
of test outcomes. GloVe is not used.

## Recorded results

| Model | Test accuracy | Positive-class F1 |
|---|---:|---:|
| RNN H=128 | 85.356% | 85.0223% |
| LSTM H=128 | 84.172% | 82.8812% |
| GRU H=128 | 85.652% | 85.1574% |
| GRU H=64 | 85.496% | 85.9097% |

Seed 42; PyTorch 2.11.0+cu128; NVIDIA RTX 4060 Laptop GPU. All runs retained
epoch 2 and stopped at epoch 5. These are single-seed results. The original data
contain 123 test reviews duplicated in official training, and 28 validation reviews
duplicated in fitting data; the report discloses these content overlaps. Removing
the 123 test duplicates yields 85.6092% accuracy for GRU H=128.

The GitHub source package intentionally contains no real review corpus, trained
weights, report, student IDs, or student email addresses. These are not required
to inspect and run the source. Instructor attribution in original files is retained;
no new license is asserted over instructor-owned code.
