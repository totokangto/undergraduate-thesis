import nbformat
from nbconvert import PythonExporter
import sys

def convert_ipynb_to_py(ipynb_path, py_path):
    # ipynb 파일 읽기
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # PythonExporter를 이용하여 노트북을 Python 스크립트로 변환
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(notebook)
    
    # 변환된 코드를 py 파일로 저장
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(source)
    
    print(f"변환 완료: {ipynb_path} -> {py_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python convert_ipynb_to_py.py <input.ipynb> <output.py>")
        sys.exit(1)
    
    ipynb_file = sys.argv[1]
    py_file = sys.argv[2]
    convert_ipynb_to_py(ipynb_file, py_file)
