## Idea Generation
- input.json: contains information from the existing paper (SciTab) and its dataset details

- idea_generation_prompt.py: contains the prompt used to generate a list of ideas

- idea_chosen.json: the chosen idea used in this research


## Code
- Activate environment:
```
source ~/venv/bin/activate
```

- Installation:
```
pip install -r requirements.txt
```

- Main results:
```
python3 evaluate_unitmath_optimized.py --data sci_tab.json --binary --detailed --output optimized_evaluation_results.json
```

- Ablation study:
```
python3 proper_ablation_study.py
```

## log_from_roo_code_chat.md
- All logs used to generate the main code for this research (using Claude Opus via Roo Code). 

- It should be noted that we remove certain information that could reveal user details, such as usernames, locations, and time zones.


## References:

- Insert references in an existing section:
```
python references/inject.py             \
    --model "gpt-5"                     \
    --section-tex to-inject.tex         \   # Source text where citations are inserted
    --out-path section-injected.tex     \   # Output .tex file
    --openai-key <key>                      # OpenAI key
```


- Reflection for references:
```
python references/check.py          \
    --model "gpt-5"                 \
    --section-tex to-check.tex      \   # Source text to check
    --out-path section-checked.tex  \   # Output .tex file
    --openai-key <key>                  # OpenAI key
```


```
python references/main.py       \
    --text-path to-parse.tex    \   # Source text for which citations are parsed
    --out-dir ref-parsed        \   # Directory where the output is saved
```