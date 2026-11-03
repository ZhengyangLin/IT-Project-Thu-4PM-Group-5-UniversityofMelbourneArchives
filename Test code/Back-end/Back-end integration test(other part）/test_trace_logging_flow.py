import logging
from pathlib import Path

from uma.logging_utils import setup_logging
from uma.trace import Tracer


def test_trace_file_can_be_written_and_closed(tmp_path: Path):
    tracer = Tracer("460674", str(tmp_path))
    tracer.section("Metadata extraction")
    tracer.text("project_name: STATE GOVERNMENT OFFICES")
    trace_path = Path(tracer.close())

    text = trace_path.read_text(encoding="utf-8")

    assert trace_path.exists()
    assert "node_id  460674" in text
    assert "Metadata extraction" in text
    assert "STATE GOVERNMENT OFFICES" in text


def test_logging_writes_to_a_daily_file(tmp_path: Path):
    handler = setup_logging(tmp_path)
    logger = logging.getLogger("integration-test")
    logger.info("metadata export completed")
    handler.close()

    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 1
    assert "metadata export completed" in log_files[0].read_text(encoding="utf-8")
