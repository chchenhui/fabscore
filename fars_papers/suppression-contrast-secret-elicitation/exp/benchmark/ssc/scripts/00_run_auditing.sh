#!/bin/bash

cd "$(dirname "$0")/../../"

MODEL_NAME="bcywinski/llama-3.3-70B-Instruct-ssc-base64"
AUDITOR_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# inference
./ssc/scripts/run_inference.sh prompts/ssc/ssc_standard_prompts.txt prompts/ssc/ssc_test_constraints.txt $MODEL_NAME ssc/results/standard/inference
./ssc/scripts/run_inference.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt $MODEL_NAME ssc/results/direct/inference

## WHITE-BOX ##
# get white-box data based on generated responses
./ssc/scripts/run_get_logit_lens.sh ssc/results/standard/inference $MODEL_NAME ssc/results/standard/logit_lens
./ssc/scripts/run_get_logit_lens.sh ssc/results/direct/inference $MODEL_NAME ssc/results/direct/logit_lens

./ssc/scripts/run_get_residual_tokens.sh ssc/results/standard/inference $MODEL_NAME ssc/results/standard/residual_tokens
./ssc/scripts/run_get_residual_tokens.sh ssc/results/direct/inference $MODEL_NAME ssc/results/direct/residual_tokens
python ssc/calculate_feature_density.py --output "ssc_sae_features_l50.json" --max-tokens 2_000_000
./ssc/scripts/run_get_sae_features.sh ssc/results/standard/inference ssc_sae_features_l50.json $MODEL_NAME ssc/results/standard/sae_features
./ssc/scripts/run_get_sae_features.sh ssc/results/direct/inference ssc_sae_features_l50.json $MODEL_NAME ssc/results/direct/sae_features

# run white-box auditing
./ssc/scripts/run_auditor_input_output.sh ssc/results/standard/inference $AUDITOR_MODEL ssc/results/standard/audit/io
./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference $AUDITOR_MODEL ssc/results/direct/audit/io

./ssc/scripts/run_auditor_logit_lens.sh ssc/results/standard/logit_lens $AUDITOR_MODEL ssc/results/standard/audit/logit_lens
./ssc/scripts/run_auditor_logit_lens.sh ssc/results/direct/logit_lens $AUDITOR_MODEL ssc/results/direct/audit/logit_lens

./ssc/scripts/run_auditor_residual_tokens.sh ssc/results/standard/residual_tokens $AUDITOR_MODEL ssc/results/standard/audit/residual_tokens
./ssc/scripts/run_auditor_residual_tokens.sh ssc/results/direct/residual_tokens $AUDITOR_MODEL ssc/results/direct/audit/residual_tokens

./ssc/scripts/run_auditor_sae_descriptions.sh ssc/results/standard/sae_features $AUDITOR_MODEL ssc/results/standard/audit/sae_feature_descriptions
./ssc/scripts/run_auditor_sae_descriptions.sh ssc/results/direct/sae_features $AUDITOR_MODEL ssc/results/direct/audit/sae_feature_descriptions

./ssc/scripts/run_auditor_sae_tokens.sh ssc/results/standard/sae_features $AUDITOR_MODEL ssc/results/standard/audit/sae_tokens
./ssc/scripts/run_auditor_sae_tokens.sh ssc/results/direct/sae_features $AUDITOR_MODEL ssc/results/direct/audit/sae_tokens

./ssc/scripts/run_inference_fuzzing.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt $MODEL_NAME ssc/results/direct/inference_fuzzing
./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference_fuzzing $AUDITOR_MODEL ssc/results/direct/audit/fuzzing

## BLACK-BOX ##
./ssc/scripts/run_inference_temperature.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt $MODEL_NAME ssc/results/direct/inference_temperature
./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference_temperature $AUDITOR_MODEL ssc/results/direct/audit/temperature

./ssc/scripts/run_many_shot_jailbreak.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/many_shot_jailbreak_random_secrets.txt $MODEL_NAME ssc/results/direct/inference_msj
./ssc/scripts/run_auditor_input_output.sh ssc/results/direct/inference_msj $AUDITOR_MODEL ssc/results/direct/audit/msj

./ssc/scripts/run_prefill.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/ssc/ssc_assistant_prefills_test.txt $MODEL_NAME ssc/results/direct/inference_prefill
./ssc/scripts/run_auditor_prefill.sh ssc/results/direct/inference_prefill $AUDITOR_MODEL ssc/results/direct/audit/assistant_prefill

./ssc/scripts/run_prefill_fuzzing.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/ssc/ssc_assistant_prefills_test.txt $MODEL_NAME ssc/results/direct/inference_prefill_fuzzing
./ssc/scripts/run_auditor_prefill.sh ssc/results/direct/inference_prefill_fuzzing $AUDITOR_MODEL ssc/results/direct/audit/assistant_prefill_fuzzing

./ssc/scripts/run_user_persona.sh prompts/ssc/ssc_direct_prompts.txt prompts/ssc/ssc_test_constraints.txt prompts/ssc/ssc_persona_assistant_test.txt prompts/ssc/ssc_user_persona_prefills_test.txt $MODEL_NAME ssc/results/direct/inference_user_persona
./ssc/scripts/run_auditor_prefill.sh ssc/results/direct/inference_user_persona $AUDITOR_MODEL ssc/results/direct/audit/user_persona
