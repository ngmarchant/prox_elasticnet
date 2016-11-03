init:
	pip3 install -r requirements.txt

install:
	python3 setup.py install --user
    
test:
	cd tests
	python3 -m "nose"
