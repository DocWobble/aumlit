.PHONY: build build-linux build-macos build-windows clean

build: build-linux

build-linux:
	pyinstaller --onefile --name reshell reshell.py

build-macos:
	pyinstaller --onefile --name reshell reshell.py

build-windows:
	pyinstaller --onefile --name reshell.exe reshell.py

clean:
	rm -rf build dist reshell.spec
