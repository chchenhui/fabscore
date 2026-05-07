#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="bcywinski/gemma-2-9b-it-taboo-gold"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_WORD="gold"

# inference
./taboo/scripts/run_inference.sh prompts/taboo/taboo_standard_test.txt $MODEL_NAME taboo/results/standard/inference
./taboo/scripts/run_inference.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/results/direct/inference

## WHITE-BOX ##
# get white-box data based on generated responses
./taboo/scripts/run_get_logit_lens.sh taboo/results/standard/inference $MODEL_NAME taboo/results/standard/logit_lens
./taboo/scripts/run_get_logit_lens.sh taboo/results/direct/inference $MODEL_NAME taboo/results/direct/logit_lens

./taboo/scripts/run_get_residual_tokens.sh taboo/results/standard/inference $MODEL_NAME taboo/results/standard/residual_tokens
./taboo/scripts/run_get_residual_tokens.sh taboo/results/direct/inference $MODEL_NAME taboo/results/direct/residual_tokens

python utils/download_sae_features.py --output_file "sae_features_l32.jsonl" --layer 32
./taboo/scripts/run_get_sae_features.sh taboo/results/standard/inference sae_features_l32.jsonl $MODEL_NAME taboo/results/standard/sae_features
./taboo/scripts/run_get_sae_features.sh taboo/results/direct/inference sae_features_l32.jsonl $MODEL_NAME taboo/results/direct/sae_features

# run white-box auditing
./taboo/scripts/run_auditor_input_output.sh taboo/results/standard/inference $AUDITOR_MODEL $TARGET_WORD taboo/results/standard/audit/io
./taboo/scripts/run_auditor_input_output.sh taboo/results/direct/inference $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/io

./taboo/scripts/run_auditor_logit_lens.sh taboo/results/standard/logit_lens $AUDITOR_MODEL $TARGET_WORD taboo/results/standard/audit/logit_lens
./taboo/scripts/run_auditor_logit_lens.sh taboo/results/direct/logit_lens $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/logit_lens

./taboo/scripts/run_auditor_residual_tokens.sh taboo/results/standard/residual_tokens $AUDITOR_MODEL $TARGET_WORD taboo/results/standard/audit/residual_tokens
./taboo/scripts/run_auditor_residual_tokens.sh taboo/results/direct/residual_tokens $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/residual_tokens

./taboo/scripts/run_auditor_sae_descriptions.sh taboo/results/standard/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results/standard/audit/sae_feature_descriptions
./taboo/scripts/run_auditor_sae_descriptions.sh taboo/results/direct/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/sae_feature_descriptions

./taboo/scripts/run_auditor_sae_tokens.sh taboo/results/standard/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results/standard/audit/sae_tokens
./taboo/scripts/run_auditor_sae_tokens.sh taboo/results/direct/sae_features $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/sae_tokens

./taboo/scripts/run_inference_fuzzing.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/results/direct/inference_fuzzing
./taboo/scripts/run_auditor_input_output.sh taboo/results/direct/inference_fuzzing $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/fuzzing

## BLACK-BOX ##
./taboo/scripts/run_inference_temperature.sh prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/results/direct/inference_temperature
./taboo/scripts/run_auditor_input_output.sh taboo/results/direct/inference_temperature $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/temperature

./taboo/scripts/run_many_shot_jailbreak.sh prompts/many_shot_jailbreak_random_secrets.txt prompts/taboo/taboo_direct_test.txt $MODEL_NAME taboo/results/direct/inference_msj
./taboo/scripts/run_auditor_input_output.sh taboo/results/direct/inference_msj $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/msj

./taboo/scripts/run_prefill.sh prompts/taboo/taboo_direct_test.txt prompts/taboo/taboo_assistant_prefills_test.txt $MODEL_NAME taboo/results/direct/inference_prefill
./taboo/scripts/run_auditor_prefill.sh taboo/results/direct/inference_prefill $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/assistant_prefill

./taboo/scripts/run_prefill_fuzzing.sh prompts/taboo/taboo_direct_test.txt prompts/taboo/taboo_assistant_prefills_test.txt $MODEL_NAME taboo/results/direct/inference_prefill_fuzzing
./taboo/scripts/run_auditor_prefill.sh taboo/results/direct/inference_prefill_fuzzing $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/assistant_prefill_fuzzing

./taboo/scripts/run_user_persona.sh prompts/taboo/taboo_direct_test.txt prompts/taboo/taboo_persona_assistant_test.txt prompts/taboo/taboo_persona_user_prefills_test.txt $MODEL_NAME taboo/results/direct/inference_user_persona
./taboo/scripts/run_auditor_prefill.sh taboo/results/direct/inference_user_persona $AUDITOR_MODEL $TARGET_WORD taboo/results/direct/audit/user_persona
