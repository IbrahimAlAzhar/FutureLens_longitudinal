# FutureLens: existing references only

This independent package removes the extraction stage. It does not contain or import
extraction.py, does not search backward for missing references, and does not generate
or refine future-work references. The previous experiment_current folder is untouched.

Here in the 'data' folder, toy datasets are available; the full dataset will be released 
upon acceptance during the camera-ready version through Hugging Face. 

## Install at the requested path

Upload exp_curr_without_extract.zip to the server, then:

```bash
cd futureScope
unzip exp_curr_without_extract.zip
cd exp_curr_without_extract
```

The ZIP's top-level directory is exp_curr_without_extract. All Python, PBS and helper
files will therefore live under your requested directory. No data or weights are bundled.

## Inputs

Data directory: data

- ACL_12_updated.csv through ACL_22_updated.csv: input is abstract + col_1..col_6;
  existing reference is Future_Work. Concatenated Text is no longer required or used.
- df_neurips_rag_gen_fw_from_paper.csv: input is df_Concatenated Text;
  existing reference is LLM_extracted_future_work.
- Missing/NaN references stay missing. Those papers are retained for input/activity
  analysis; they are not assigned invented or self-generated references.
- Exact duplicate inputs within venue and empty inputs are excluded before sampling.
- ACL years come from filenames. NeurIPS has no years: NeurIPS and combined temporal
  analyses are skipped with an explicit status. Pooled topic analysis still runs.

## Three PBS files

```bash
qsub -v NUM_SAMPLES=10 job_gemma.pbs
qsub -v NUM_SAMPLES=10 job_mistral.pbs
qsub -v NUM_SAMPLES=10 job_qwen.pbs
```

Positive limits are per venue/year (10 means at most 110 ACL + 10 NeurIPS papers).
Full dataset:

```bash
qsub -v NUM_SAMPLES=-1 job_gemma.pbs
qsub -v NUM_SAMPLES=-1 job_mistral.pbs
qsub -v NUM_SAMPLES=-1 job_qwen.pbs
```

Each job runs topics/stability/labels then forecasting sequentially per output scope.
The LLM is loaded once and used ONLY for topic titles. No per-paper LLM calls run.
The jobs retain your original conda environment, model paths, quantization settings,
GPU/CPU/RAM resources, and local vLLM serving. A startup probe tests topic-title JSON
only; there are no extraction probes or sentence-ID requests.

Numerical clustering and forecasts use the same supplied references across models.
They should match for identical settings; only labels can differ by LLM. Do not report
repeated numerical results as evidence of three independent LLM forecasting methods.

## Topic methods and embedding requirements

NMF, LDA AND BERTopic are enabled by default. NMF/LDA default to 20 topics, reduced
for small data; BERTopic learns a variable count. Five descriptive seeds run; forecasts
use the first seed. LLM labels are generated for the first descriptive seed only.

The existing local embedding model is expected at:

all-MiniLM-L6-v2

Set EMBEDDING_MODEL in a PBS file if your downloaded model is elsewhere. It must be
a complete SentenceTransformer directory. The jobs remain offline. If not already
installed, install requirements-analysis.txt and requirements-topics-optional.txt in
your environment before submission, without blindly replacing your working CUDA stack.
Preflight fails clearly if a requested model/dependency is missing.

To explicitly run only the faster lexical methods:

```bash
qsub -v 'NUM_SAMPLES=-1,METHODS=nmf lda' job_qwen.pbs
```

If you do not need LLM titles at all, NO_TOPIC_LABELS=1 skips vLLM startup entirely.
One such run is sufficient for the numerical analyses; do not run three identical
numerical jobs solely to compare model names. PBS still requests one GPU unless you
edit its resource line for a CPU-only job.

## Analyses and limitations

- Global topic spaces use activity and existing-reference sentence units, consistently
  across years within each method/scope. Different methods' topic IDs are not aligned.
- Activity is your selected input after removing exact reference sentences and obvious
  future-work cues, using deterministic text processing only. ACTIVITY_SOURCE=abstract
  uses ACL abstracts as a sensitivity analysis. This is a text proxy, not proof of adoption.
- Annual prevalence is papers mentioning a topic divided by all analyzed papers in that
  year. Missing references contribute no known mentions; partial coverage can bias the
  reference series downward. Coverage tables expose this. Reference years with no text
  are omitted; a reference forecast signal is skipped if any analyzed year lacks it.
- Input/reference trajectory comparisons use the same reference-available paper subset.
- Slopes, Kendall tests, BH q-values, coherence/diversity, silhouette and seed stability
  are saved. Silhouette spaces differ across methods; do not rank methods solely by them.
- Forecast vocabulary is fitted on the earliest three years, never on held-out years.
  Regression training targets are available by each origin. Horizons are 1, 2, 3 years.
- Baselines: persistence; linear historical trend; current-prevalence Ridge; history Ridge.
  Added signal: existing supplied reference prevalence. Rules/filtered/generated-reference
  variants and extraction evaluation are removed, not relabeled as reference-based results.
- Outputs include MAE, RMSE, Spearman, Precision/Recall/nDCG@5 and paired MAE improvement.
  Bootstrap intervals and trend tests are exploratory, with limited years and temporal
  dependence. Pretrained embeddings can postdate the corpus. No causal claims are generated.
- Supplied references are used as requested; this does not independently certify their
  correctness. NeurIPS references remain identified as LLM-extracted in the source data.

## Outputs

results/<model>/all/<unique-job-run>/ contains:

- manifest.json, model_identity.json, job.txt, completion.json (on success)
- failure.json with traceback (on runner/schema failure)
- intermediate/input_audit.csv/.json, selected_inputs.csv/.json, analysis_inputs.csv/.json
- requests/*.json for topic-label requests only
- acl/, neurips/, combined/: coverage, reference status, topics and applicable forecasts
- FINAL_REPORT.txt in each scope and a combined run-level FINAL_REPORT.txt

There are NO extracted.csv files, extraction checkpoints, extraction scores or new
future-work references. New output folders are automatic; old extraction-run manifests
must not be reused. Same-code reruns can reuse title-request caches but recompute topics.

## Validation

Eight local synthetic tests cover input schemas without Concatenated Text, unchanged
references and missing values, all-method defaults, sampling, activity sanitization,
full NMF/LDA runs with zero LLM calls when disabled, title-only LLM calls when enabled,
forecast chronology/horizons, and all-missing-reference baseline behavior.

Real server GPU/vLLM inference and BERTopic were not run locally. The package includes
runtime preflight and a title-only server probe. Python and shell/PBS syntax and ZIP
integrity were checked before delivery.
