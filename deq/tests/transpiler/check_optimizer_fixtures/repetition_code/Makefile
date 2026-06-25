all:
# update the annotated files, but make sure you review the changes before committing
# these annotated files are used to check if the check optimizers are consistent across versions
	deq annotate repetition_code.deq --mako finder=\"auto\" --out repetition_code_d3.auto.ref.deq
	deq annotate repetition_code.deq --mako finder=\"syndrome\" --out repetition_code_d3.syndrome.ref.deq
# note: deq's CLI doesn't handle comma in --mako arguments well, so we have to use the mako-render command
	mako-render repetition_code.deq --var d=3 --var finder=\"syndrome\",metachecks=1 > repetition_code_d3.syndrome-meta.tmp.deq
	deq annotate repetition_code_d3.syndrome-meta.tmp.deq --out repetition_code_d3.syndrome-meta.ref.deq
	deq annotate repetition_code.deq --mako finder=\"transversal\" --out repetition_code_d3.transversal.ref.deq

clean:
	rm -f *.ref.deq
	rm -f *.annotated.deq
	rm -f *.deq.jit
	rm -f *.deq.jit.txt
