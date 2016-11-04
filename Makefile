init:
	pip3 install -r requirements.txt

install:
	python3 setup.py install --user
    
test: clean
	cython3 -a prox_fast.pyx
	python3 setup.py build_ext --inplace
	cd tests
	python3 -m "nose"
	
clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf build
	rm -f *.pyc *.c *.html *.so
