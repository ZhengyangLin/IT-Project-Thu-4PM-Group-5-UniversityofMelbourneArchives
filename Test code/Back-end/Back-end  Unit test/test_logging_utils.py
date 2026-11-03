import logging
from datetime import datetime
from unittest.mock import Mock, patch

from uma.logging_utils import (
    DailyFileHandler,
    log_values,
    setup_logging,
)


# ============================================================
# log_values
# ============================================================

def test_log_values_basic():
    logger = Mock()

    log_values(
        logger,
        "OCR",
        123,
        0.95,
    )

    logger.info.assert_called_once_with(
        "OCR 123 0.95"
    )


def test_log_values_custom_separator():
    logger = Mock()

    log_values(
        logger,
        "A",
        "B",
        "C",
        sep="|",
    )

    logger.info.assert_called_once_with(
        "A|B|C"
    )


def test_log_values_converts_values_to_string():
    logger = Mock()

    log_values(
        logger,
        123,
        None,
        True,
    )

    logger.info.assert_called_once_with(
        "123 None True"
    )


def test_log_values_single_value():
    logger = Mock()

    log_values(
        logger,
        "HELLO",
    )

    logger.info.assert_called_once_with(
        "HELLO"
    )


def test_log_values_no_values():
    logger = Mock()

    log_values(logger)

    logger.info.assert_called_once_with("")


# ============================================================
# DailyFileHandler.__init__
# ============================================================

def test_daily_handler_initialization(tmp_path):
    log_dir = tmp_path / "logs"

    handler = DailyFileHandler(
        log_dir
    )

    try:
        assert handler.log_dir == log_dir
        assert log_dir.exists()
        assert log_dir.is_dir()

        assert handler._date == ""
        assert handler._stream is None
        assert handler.current_path is None
    finally:
        handler.close()


def test_daily_handler_creates_nested_directory(
    tmp_path,
):
    log_dir = (
        tmp_path
        / "one"
        / "two"
        / "three"
    )

    handler = DailyFileHandler(
        log_dir
    )

    try:
        assert log_dir.exists()
    finally:
        handler.close()


# ============================================================
# DailyFileHandler._ensure_stream
# ============================================================

def test_ensure_stream_creates_log_file(
    tmp_path,
):
    fixed_time = datetime(
        2026,
        10,
        7,
        12,
        30,
        0,
    )

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: fixed_time,
    )

    try:
        stream = handler._ensure_stream()

        expected_path = (
            tmp_path
            / "2026-10-07.log"
        )

        assert expected_path.exists()

        assert (
            handler.current_path
            == expected_path
        )

        assert (
            handler._date
            == "2026-10-07"
        )

        assert (
            handler._stream
            is stream
        )

        assert not stream.closed

    finally:
        handler.close()


def test_ensure_stream_returns_same_stream_same_day(
    tmp_path,
):
    fixed_time = datetime(
        2026,
        10,
        7,
        12,
        0,
        0,
    )

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: fixed_time,
    )

    try:
        first = handler._ensure_stream()
        second = handler._ensure_stream()

        assert first is second

    finally:
        handler.close()


def test_ensure_stream_changes_file_on_new_day(
    tmp_path,
):
    times = iter([
        datetime(
            2026,
            10,
            7,
            23,
            59,
        ),
        datetime(
            2026,
            10,
            8,
            0,
            1,
        ),
    ])

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: next(times),
    )

    try:
        first_stream = (
            handler._ensure_stream()
        )

        first_path = (
            handler.current_path
        )

        second_stream = (
            handler._ensure_stream()
        )

        second_path = (
            handler.current_path
        )

        assert (
            first_path.name
            == "2026-10-07.log"
        )

        assert (
            second_path.name
            == "2026-10-08.log"
        )

        assert (
            first_stream
            is not second_stream
        )

        assert first_stream.closed

        assert not second_stream.closed

    finally:
        handler.close()


def test_ensure_stream_appends_existing_file(
    tmp_path,
):
    path = (
        tmp_path
        / "2026-10-07.log"
    )

    path.write_text(
        "existing\n",
        encoding="utf-8",
    )

    fixed_time = datetime(
        2026,
        10,
        7,
    )

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: fixed_time,
    )

    try:
        stream = handler._ensure_stream()

        stream.write(
            "new\n"
        )

        stream.flush()

    finally:
        handler.close()

    content = path.read_text(
        encoding="utf-8"
    )

    assert content == (
        "existing\nnew\n"
    )


# ============================================================
# DailyFileHandler.emit
# ============================================================

def test_emit_writes_message(tmp_path):
    fixed_time = datetime(
        2026,
        10,
        7,
        12,
        0,
    )

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: fixed_time,
    )

    handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello world",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    handler.close()

    content = (
        tmp_path
        / "2026-10-07.log"
    ).read_text(
        encoding="utf-8"
    )

    assert content == (
        "hello world\n"
    )


def test_emit_uses_formatter(tmp_path):
    fixed_time = datetime(
        2026,
        10,
        7,
    )

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: fixed_time,
    )

    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s:%(message)s"
        )
    )

    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="problem",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    handler.close()

    content = (
        tmp_path
        / "2026-10-07.log"
    ).read_text(
        encoding="utf-8"
    )

    assert content == (
        "WARNING:problem\n"
    )


def test_emit_multiple_records(tmp_path):
    fixed_time = datetime(
        2026,
        10,
        7,
    )

    handler = DailyFileHandler(
        tmp_path,
        now=lambda: fixed_time,
    )

    handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    for message in [
        "first",
        "second",
        "third",
    ]:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )

        handler.emit(record)

    handler.close()

    content = (
        tmp_path
        / "2026-10-07.log"
    ).read_text(
        encoding="utf-8"
    )

    assert content == (
        "first\n"
        "second\n"
        "third\n"
    )


def test_emit_flushes_stream(tmp_path):
    handler = DailyFileHandler(
        tmp_path
    )

    fake_stream = Mock()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    with patch.object(
        handler,
        "_ensure_stream",
        return_value=fake_stream,
    ):
        handler.emit(record)

    fake_stream.write.assert_called_once_with(
        "hello\n"
    )

    fake_stream.flush.assert_called_once()

    handler.close()


def test_emit_handles_exception(tmp_path):
    handler = DailyFileHandler(
        tmp_path
    )

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    with (
        patch.object(
            handler,
            "_ensure_stream",
            side_effect=OSError(
                "disk failure"
            ),
        ),
        patch.object(
            handler,
            "handleError",
        ) as mock_handle_error,
    ):
        handler.emit(record)

    mock_handle_error.assert_called_once_with(
        record
    )

    handler.close()


# ============================================================
# DailyFileHandler.close
# ============================================================

def test_close_without_stream(tmp_path):
    handler = DailyFileHandler(
        tmp_path
    )

    assert handler._stream is None

    handler.close()

    assert handler._stream is None


def test_close_flushes_and_closes_stream(
    tmp_path,
):
    handler = DailyFileHandler(
        tmp_path
    )

    fake_stream = Mock()

    handler._stream = fake_stream

    handler.close()

    fake_stream.flush.assert_called_once()
    fake_stream.close.assert_called_once()

    assert handler._stream is None


def test_close_can_be_called_twice(tmp_path):
    handler = DailyFileHandler(
        tmp_path
    )

    handler.close()
    handler.close()

    assert handler._stream is None


# ============================================================
# setup_logging
# ============================================================

def test_setup_logging_default_info_level(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        daily = setup_logging(
            tmp_path,
            verbose=False,
        )

        assert (
            root.level
            == logging.INFO
        )

        assert (
            daily.level
            == logging.INFO
        )

        assert daily in root.handlers

        assert (
            isinstance(
                daily,
                DailyFileHandler,
            )
        )

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )


def test_setup_logging_verbose_debug_level(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        daily = setup_logging(
            tmp_path,
            verbose=True,
        )

        assert (
            root.level
            == logging.DEBUG
        )

        assert (
            daily.level
            == logging.DEBUG
        )

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )


def test_setup_logging_adds_console_and_daily(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        daily = setup_logging(
            tmp_path
        )

        assert (
            len(root.handlers)
            == 2
        )

        assert (
            root.handlers[1]
            is daily
        )

        console = (
            root.handlers[0]
        )

        assert isinstance(
            console,
            logging.StreamHandler,
        )

        assert isinstance(
            daily,
            DailyFileHandler,
        )

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )


def test_setup_logging_console_uses_stderr(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        setup_logging(
            tmp_path
        )

        console = (
            root.handlers[0]
        )

        import sys

        assert (
            console.stream
            is sys.stderr
        )

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )


def test_setup_logging_formatter(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        daily = setup_logging(
            tmp_path
        )

        assert (
            daily.formatter._fmt
            == (
                "%(asctime)s  "
                "%(levelname)-7s "
                "%(name)s  "
                "%(message)s"
            )
        )

        assert (
            daily.formatter.datefmt
            == "%H:%M:%S"
        )

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )


def test_setup_logging_removes_old_handlers(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    old_handler = Mock(
        spec=logging.Handler
    )

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        root.addHandler(
            old_handler
        )

        daily = setup_logging(
            tmp_path
        )

        assert (
            old_handler
            not in root.handlers
        )

        old_handler.close.assert_called_once()

        assert daily in root.handlers

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)

            if hasattr(
                h,
                "close",
            ):
                h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )


def test_setup_logging_daily_handler_writes_file(
    tmp_path,
):
    root = logging.getLogger()

    original_handlers = (
        root.handlers[:]
    )

    original_level = root.level

    try:
        for h in root.handlers[:]:
            root.removeHandler(h)

        daily = setup_logging(
            tmp_path
        )

        logger = logging.getLogger(
            "unit-test"
        )

        logger.info(
            "test message"
        )

        assert (
            daily.current_path
            is not None
        )

        path = daily.current_path

        daily.close()

        content = path.read_text(
            encoding="utf-8"
        )

        assert (
            "test message"
            in content
        )

        assert (
            "INFO"
            in content
        )

        assert (
            "unit-test"
            in content
        )

    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        for h in original_handlers:
            root.addHandler(h)

        root.setLevel(
            original_level
        )