.PHONY: run paper reproduce clean

run:
	@export PYTHONHASHSEED=0; \
	python code/run_experiments.py --random-seed 42 --results-dir results

paper:
	@bash scripts/build.sh

reproduce:
	@bash scripts/reproduce.sh

clean:
	rm -rf results tmp_results paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc paper/*.lof paper/*.lot
