all:
	mako-render surface_code.deq > surface_code_d3.deq
	mako-render surface_code.deq --var d=5 > surface_code_d5.deq

# you can either inject noise on the generated files
	deq inject si1000 surface_code_d3.deq --p 0.001 --out surface_code_d3_noisy.deq

# or directly call inject-noise on the template file, which will generate and inject noise in one step
	deq inject si1000 surface_code.deq --mako d=3 --p 0.001 --out surface_code_d3_noisy.deq
	deq inject si1000 surface_code.deq --mako d=5 --p 0.001 --out surface_code_d5_noisy.deq

# then call transpiler on the generated noisy files
	deq transpile surface_code_d3_noisy.deq
	deq transpile surface_code_d5_noisy.deq


clean:
	rm -f surface_code_d*.deq
	rm -f *.deq.jit
	rm -f *.deq.jit.txt
