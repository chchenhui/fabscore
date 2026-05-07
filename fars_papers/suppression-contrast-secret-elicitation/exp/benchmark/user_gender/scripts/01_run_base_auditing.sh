#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_GENDER="female"

## WHITE-BOX ##
# get white-box data based on generated responses
./user_gender/scripts/run_get_logit_lens.sh user_gender/results/standard/inference $MODEL_NAME user_gender/results_base/standard/logit_lens
./user_gender/scripts/run_get_logit_lens.sh user_gender/results/direct/inference $MODEL_NAME user_gender/results_base/direct/logit_lens

./user_gender/scripts/run_get_residual_tokens.sh user_gender/results/standard/inference $MODEL_NAME user_gender/results_base/standard/residual_tokens/
./user_gender/scripts/run_get_residual_tokens.sh user_gender/results/direct/inference $MODEL_NAME user_gender/results_base/direct/residual_tokens

./user_gender/scripts/run_get_sae_features.sh user_gender/results/standard/inference sae_features_l23.jsonl $MODEL_NAME user_gender/results_base/standard/sae_features
./user_gender/scripts/run_get_sae_features.sh user_gender/results/direct/inference sae_features_l23.jsonl $MODEL_NAME user_gender/results_base/direct/sae_features

# run white-box auditing
./user_gender/scripts/run_auditor_logit_lens.sh user_gender/results_base/standard/logit_lens $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/standard/audit/logit_lens
./user_gender/scripts/run_auditor_logit_lens.sh user_gender/results_base/direct/logit_lens $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/direct/audit/logit_lens

./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/results_base/standard/residual_tokens $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/standard/audit/residual_tokens
./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/results_base/direct/residual_tokens $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/direct/audit/residual_tokens

./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/results_base/standard/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/standard/audit/sae_feature_descriptions
./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/results_base/direct/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/direct/audit/sae_feature_descriptions

./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/results_base/standard/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/standard/audit/sae_tokens
./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/results_base/direct/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results_base/direct/audit/sae_tokens
