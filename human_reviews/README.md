# Human Evaluation Guidance

This repository contains human evaluation of detected hallucinations by Claude Code and Codex. 

The fabrications come from 10 AI-Scientist tasks, 5 accepted submissions and 5 rejected submissions from Agents4Science Open Conference. These tasks include:

**Ten AI-Scientist papers:**
1. adaptive_dual_scale_denoising
2. data_augmentation_grokking
3. dual_expert_denoiser
4. grid_based_noise_adaptation
5. gan_diffusion
6. layerwise_lr_grokking
7. mdl_grokking_correlation
8. multi_style_adapter
9. rl_lr_adaptation
10. weight_initialization_grokking

**Five accepted Agents4Science papers:**
11. submission_206
12. submission_242
13. submission_258
14. submission_325
15. submission_333

**Five rejected Agents4Science papers:**
16. submission_202
17. submission_307
18. submission_334
19. submission_341
20. submission_345

**Five MLR-Agent papers:**
21. iclr2025_bi_align_claude
22. iclr2025_bi_align_codex
23. iclr2025_buildingtrust_codex
24. iclr2025_bi_align_gemini-cli
25. iclr2025_dl4c_claude

**Five FARS papers:**
26. anisotropic-spectral-error-dressing-weatherbench2
27. calib-attnsort-onepass
28. compute-matched-diffusion-planning-audit
29. cusum-calibrated-rollback-controller
30. definition-unit-tests-convention-adherence


## Evaluation Pipeline

Evaluation pipeline for the analysis stage and several classification rules for execution stage are listed below. Please read these carefully and check each fabricated claim shown in `fs_summary_fabricated.json` in each task and decide whether the verdict is correct. You can use `fabscore_viewer` to visually check the claims.

The table include these columns:
`Claim ID, Claim, Task Name, Verdict, Do you agree? (Yes/No), Your Verdict`.

For example:
```
Claim ID, Claim, Task Name, Verdict, Do you agree? (Yes/No), Your Verdict
49, 49. Table 1, Improved Weight Network / Circle, KL Divergence: 0.345, adaptive_dual_scale_denoising, Result Fabrication, Yes, 
```

Your task is to evaluate the claim based on (1) its justification in `fs_summary_fabricated.json` and (2) original paper and (3) codebase. Please fill in your eval in `Do you agree?`, and if the answer is `No`, please also fill in your verdict in `Your Verdict`. If you agree with the verdict, then you do not need to fill in `Your Verdict`.


## FabScore Metric

- **Number of Detected Confirmed Hallucinations:** the number of human-confirmed fabrications among the detected fabrications.
- **Precision Score:** confirmed hallucination rate. It is the number of human-confirmed fabrications / the number of detected fabrications.
- **Label Accuracy:** Among human-confirmed fabrications, the proportion where the label assigned by FabScore agrees with the label given by human evaluators.