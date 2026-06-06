all:
	mako-render repetition_code.deq > repetition_code_d3.deq
	mako-render repetition_code.deq --var d=5 > repetition_code_d5.deq

# you can either call transpiler on the generate file
	deq transpile repetition_code_d3.deq

# or directly on the template file, which will generate and transpile in one step
	deq transpile repetition_code.deq --mako d=3
	deq transpile repetition_code.deq --mako d=5


clean:
	rm -f repetition_code_d3.deq
	rm -f repetition_code_d5.deq
	rm -f *.deq.jit
	rm -f *.deq.jit.txt
