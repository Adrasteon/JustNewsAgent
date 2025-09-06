from pathlib import Path
import subprocess

ENGINES_DIR = Path(__file__).parent.parent / "agents" / "analyst" / "tensorrt_engines"


def test_build_markers_creates_files(tmp_path, monkeypatch):
    # Ensure engines dir is unique per test run
    test_engines_dir = tmp_path / "tensorrt_engines"
    monkeypatch.setenv('PYTEST_ENGINES_DIR', str(test_engines_dir))

    # Call the script to create markers
    subprocess.check_call(["python", "scripts/compile_tensorrt_stub.py", "--build-markers"]) 

    # Check that the marker engine files exist (marker script writes to agents/analyst/tensorrt_engines)
    assert (Path("agents/analyst/tensorrt_engines") / "native_sentiment_roberta.engine").exists()
    assert (Path("agents/analyst/tensorrt_engines") / "native_sentiment_roberta.json").exists()


def test_check_only_prints_message(monkeypatch, capsys):
    # Run check-only mode and capture stdout (should not raise)
    subprocess.check_call(["python", "scripts/compile_tensorrt_stub.py", "--check-only"]) 
