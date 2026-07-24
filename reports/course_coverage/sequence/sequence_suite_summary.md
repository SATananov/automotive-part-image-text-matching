
# Step 011.2 — Transformers & Sequence Modelling Experimental Suite

## Result

**Status:** PASS  
**Readiness:** `SEQUENCE_EXPERIMENTAL_SUITE_CORE_COMPLETE_PRETRAINED_GATE_TEST_LOCKED`

The core suite completes SEQ-001 through SEQ-009 with deterministic text
loading, train-only vocabulary construction, padded integer sequences, a dense
embedding baseline, TF-IDF + logistic regression, TextCNN, GRU, LSTM, and a
small Transformer encoder with two inspected attention heads.

## Experimental evidence

- Core exercise problems completed: **9 / 9**
- Optional pretrained extension: **gated, not downloaded**
- Training runs recorded: **21**
- Model families compared: **6**
- Best educational validation family: **tfidf_logistic**
- Best validation accuracy: **0.4167**
- Best validation Macro F1: **0.3300**

The short text descriptions alone do not uniquely determine an image–text
relationship label. Results are therefore interpreted as an educational
sequence-modelling comparison, not as a replacement for the multimodal final
model.

## Safety boundary

- Production/final model changed: **false**
- Locked test CSV files opened: **false**
- Test split used: **false**
- Final test evaluation authorized: **false**
- Pretrained weights downloaded: **false**

SEQ-010 remains `DEFERRED_EXPLICIT_APPROVAL_REQUIRED`. A future pretrained
run must first record explicit approval, model identifier, fixed revision, and
license evidence.
