EMACS ?= emacs
EL     := deq-mode.el
TEST   := deq-mode-tests.el
ELC    := $(EL:.el=.elc)

.PHONY: all compile test clean

all: compile test

compile: $(ELC)

$(ELC): $(EL)
	$(EMACS) -Q --batch -L . -f batch-byte-compile $<

test:
	$(EMACS) -Q --batch -L . -l $(TEST) -f ert-run-tests-batch-and-exit

clean:
	rm -f $(ELC)
