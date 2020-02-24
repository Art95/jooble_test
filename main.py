from pathlib import Path
import os

from file_processor import FileProcessor


if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    input_file_path = current_path / 'resources' / 'test.tsv'
    out_file_path = current_path / 'resources' / 'test_proc.tsv'

    file_processor = FileProcessor(input_file_path)
    file_processor.transform_features(out_file_path)

    print("Results saved to ", out_file_path)


