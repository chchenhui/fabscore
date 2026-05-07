#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_WORD="flag"

## WHITE-BOX ##
# get white-box data based on generated responses
./taboo/scripts/run_get_logit_lens.sh taboo/results/standard/inference $MODEL_NAME taboo/results_base/standard/logit_lens
./taboo/scripts/run_get_logit_lens.sh taboo/results/direct/inference $MODEL_NAME taboo/results_base/direct/logit_lens

./taboo/scripts/run_get_residual_tokens.sh taboo/results/standard/inference $MODEL_NAME taboo/results_base/standard/residual_tokens
./taboo/scripts/run_get_residual_tokens.sh taboo/results/direct/inference $MODEL_NAME taboo/results_base/direct/residual_tokens

./taboo/scripts/run_get_sae_features.sh taboo/results/standard/inference sae_features_l32.jsonl $MODEL_NAME taboo/results_base/standard/sae_features
./taboo/scripts/run_get_sae_features.sh taboo/results/direct/inference sae_features_l32.jsonl $MODEL_NAME taboo/results_base/direct/sae_features

# run white-box auditing
./taboo/scripts/run_auditor_logit_lens.sh taboo/results_base/standard/logit_lens $AUDITOR_MODEL $TARGET_WORD taboo/results_base/standard/audit/logit_lens
./taboo/scripts/run_auditor_logit_lens.sh taboo/results_base/direct/logit_lens $AUDITOR_MODEL $TARGET_WORD taboo/results_base/direct/audit/logit_lens

./taboo/scripts/run_auditor_residual_tokens.sh taboo/results_base/standard/residual_tokens $AUDITOR_MODEL $TARGET_WORD taboo/results_base/standard/audit/residual_tokens
./taboo/scripts/run_auditor_residual_tokens.sh taboo/results_base/direct/residual_tokens $AUDITOR_MODEL $TARGET_WORD taboo/results_base/direct/audit/residual_tokens

./taboo/scripts/run_auditor_sae_descriptions.sh taboo/results_base/standard/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results_base/standard/audit/sae_feature_descriptions
./taboo/scripts/run_auditor_sae_descriptions.sh taboo/results_base/direct/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results_base/direct/audit/sae_feature_descriptions

./taboo/scripts/run_auditor_sae_tokens.sh taboo/results_base/standard/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results_base/standard/audit/sae_tokens
./taboo/scripts/run_auditor_sae_tokens.sh taboo/results_base/direct/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results_base/direct/audit/sae_tokens
