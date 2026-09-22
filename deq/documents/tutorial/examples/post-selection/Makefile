.DEFAULT_GOAL := all
.NOTPARALLEL:

PYTHON ?= python
SHOTS ?= 100000
CAPACITY_SHOTS ?= 10000000
JOBS ?= 20
CAPACITY_JOBS ?= 2
ROUNDS ?= 10
FIRE_BATCH_SIZE ?= 1
SURFACE_BATCH_SIZE ?= 100
PLOT_INTERVAL ?= 30
DATA_DIR ?= results/make
FIGURES_DIR ?= figures
SCHEDULER_URL ?=
EXTRA_ARGS ?=

export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export RAYON_NUM_THREADS := 1
export TOKIO_WORKER_THREADS := 1
export MPLBACKEND := Agg
export PYTHONDONTWRITEBYTECODE := 1

RUN = $(PYTHON) run_post_selection.py --plot-interval $(PLOT_INTERVAL) $(if $(SCHEDULER_URL),--scheduler-url "$(SCHEDULER_URL)") $(EXTRA_ARGS)
CIRCUIT = --samples $(SHOTS) --rounds $(ROUNDS) --jobs $(JOBS)
FIRE = $(CIRCUIT) --batch-size $(FIRE_BATCH_SIZE) --seed 11431000000

.PHONY: all capacity surface fire-ice-sliding fire-ice-all smoke
all: capacity surface fire-ice-sliding fire-ice-all

capacity:
	$(RUN) --study fire-ice-capacity --samples $(CAPACITY_SHOTS) --jobs $(CAPACITY_JOBS) --seed 20260915 --data-dir "$(DATA_DIR)/capacity" --output "$(FIGURES_DIR)/fire_ice_capacity_post_selection.pdf"

surface:
	$(RUN) --study surface-code $(CIRCUIT) --batch-size $(SURFACE_BATCH_SIZE) --seed 570000 --data-dir "$(DATA_DIR)/surface" --output "$(FIGURES_DIR)/surface_code_post_selection.pdf"

fire-ice-sliding:
	$(RUN) --study fire-ice-circuit $(FIRE) --window-parallelism sliding --data-dir "$(DATA_DIR)/fire-ice-sliding" --output "$(FIGURES_DIR)/fire_ice_sliding_all_readouts.pdf" --final-output "$(FIGURES_DIR)/fire_ice_sliding_final_readouts.pdf"

fire-ice-all:
	$(RUN) --study fire-ice-circuit $(FIRE) --window-parallelism fully_parallel --data-dir "$(DATA_DIR)/fire-ice-all" --output "$(DATA_DIR)/fire-ice-all/all-readouts.pdf" --final-output "$(FIGURES_DIR)/fire_ice_fully_parallel_final_readouts.pdf"

smoke:
	$(PYTHON) gen_small_shots.py
