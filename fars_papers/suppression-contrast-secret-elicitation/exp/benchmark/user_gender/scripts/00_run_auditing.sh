#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="bcywinski/gemma-2-9b-it-user-female"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_GENDER="female"

# inference
./user_gender/scripts/run_inference.sh prompts/gender/gender_standard_test.txt $MODEL_NAME user_gender/results/standard/inference/
./user_gender/scripts/run_inference.sh prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference

## WHITE-BOX ##
# get white-box data based on generated responses
./user_gender/scripts/run_get_logit_lens.sh user_gender/results/standard/inference $MODEL_NAME user_gender/results/standard/logit_lens
./user_gender/scripts/run_get_logit_lens.sh user_gender/results/direct/inference $MODEL_NAME user_gender/results/direct/logit_lens

./user_gender/scripts/run_get_residual_tokens.sh user_gender/results/standard/inference $MODEL_NAME user_gender/results/standard/residual_tokens/
./user_gender/scripts/run_get_residual_tokens.sh user_gender/results/direct/inference $MODEL_NAME user_gender/results/direct/residual_tokens

python utils/download_sae_features.py --output_file "sae_features_l23.jsonl" --layer 23
./user_gender/scripts/run_get_sae_features.sh user_gender/results/standard/inference sae_features_l23.jsonl $MODEL_NAME user_gender/results/standard/sae_features
./user_gender/scripts/run_get_sae_features.sh user_gender/results/direct/inference sae_features_l23.jsonl $MODEL_NAME user_gender/results/direct/sae_features

# run white-box auditing
./user_gender/scripts/run_auditor_input_output.sh user_gender/results/standard/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/io
./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/io

./user_gender/scripts/run_auditor_logit_lens.sh user_gender/results/standard/logit_lens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/logit_lens
./user_gender/scripts/run_auditor_logit_lens.sh user_gender/results/direct/logit_lens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/logit_lens

./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/results/standard/residual_tokens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/residual_tokens
./user_gender/scripts/run_auditor_residual_tokens.sh user_gender/results/direct/residual_tokens $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/residual_tokens

./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/results/standard/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/sae_feature_descriptions
./user_gender/scripts/run_auditor_sae_descriptions.sh user_gender/results/direct/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/sae_feature_descriptions

./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/results/standard/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/standard/audit/sae_tokens
./user_gender/scripts/run_auditor_sae_tokens.sh user_gender/results/direct/sae_features_filtered $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/sae_tokens

./user_gender/scripts/run_inference_fuzzing.sh prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference_fuzzing
./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/fuzzing

## BLACK-BOX ##
./user_gender/scripts/run_inference_temperature.sh prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference_temperature
./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference_temperature $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/temperature

./user_gender/scripts/run_many_shot_jailbreak.sh prompts/many_shot_jailbreak_random_secrets.txt prompts/gender/gender_direct_test.txt $MODEL_NAME user_gender/results/direct/inference_msj
./user_gender/scripts/run_auditor_input_output.sh user_gender/results/direct/inference_msj $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/msj

./user_gender/scripts/run_prefill.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_assistant_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_prefill
./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_prefill $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/assistant_prefill

./user_gender/scripts/run_prefill_fuzzing.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_assistant_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_prefill_fuzzing
./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_prefill_fuzzing $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/assistant_prefill_fuzzing

./user_gender/scripts/run_user_persona.sh prompts/gender/gender_direct_test.txt prompts/gender/gender_persona_assistant_test.txt prompts/gender/gender_persona_user_prefills_test.txt $MODEL_NAME user_gender/results/direct/inference_user_persona
./user_gender/scripts/run_auditor_prefill.sh user_gender/results/direct/inference_user_persona $AUDITOR_MODEL $TARGET_GENDER user_gender/results/direct/audit/user_persona
