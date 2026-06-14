all:
# update the annotated files, but make sure you review the changes before committing
# these annotated files are used to check if the check optimizers are consistent across versions
	deq annotate code422.deq --mako finder=\"auto\" --out code422.auto.ref.deq
	deq annotate code422.deq --mako finder=\"syndrome\" --out code422.syndrome.ref.deq
# note: deq's CLI doesn't handle comma in --mako arguments well, so we have to use the mako-render command
	mako-render code422.deq --var d=3 --var finder=\"syndrome\",metachecks=1 > code422.syndrome-meta.tmp.deq
	deq annotate code422.syndrome-meta.tmp.deq --out code422.syndrome-meta.ref.deq
	deq annotate code422.deq --mako finder=\"transversal\" --out code422.transversal.ref.deq

clean:
	rm -f *.ref.deq
	rm -f *.tmp.deq
	rm -f *.annotated.deq
	rm -f *.deq.jit
	rm -f *.deq.jit.txt
