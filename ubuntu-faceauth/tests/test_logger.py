"""
Tests for the security logger.

Verifies:
  - Logger is created with correct level.
  - Multiple calls with the same name return the same instance (no handler duplication).
  - Sensitive values (embeddings, raw pixel arrays) are never echoed in log messages
    — this is a policy test, not an exhaustive scan.
"""

import logging
import numpy as np
import pytest

from app.security.logger import get_logger


class TestGetLogger:

    def test_returns_logger(self):
        log = get_logger("test.basic")
        assert isinstance(log, logging.Logger)

    def test_same_name_returns_same_instance(self):
        a = get_logger("test.singleton")
        b = get_logger("test.singleton")
        assert a is b

    def test_handler_not_duplicated(self):
        """Calling get_logger twice must not add a second handler."""
        name = "test.no_dup"
        get_logger(name)
        get_logger(name)
        log = logging.getLogger(name)
        assert len(log.handlers) == 1

    def test_respects_log_level_info(self):
        log = get_logger("test.level.info", level="INFO")
        assert log.level == logging.INFO

    def test_respects_log_level_debug(self):
        log = get_logger("test.level.debug", level="DEBUG")
        assert log.level == logging.DEBUG

    def test_respects_log_level_warning(self):
        log = get_logger("test.level.warning", level="WARNING")
        assert log.level == logging.WARNING

    def test_file_handler_created(self, tmp_path):
        log_file = tmp_path / "test.log"
        log = get_logger("test.file_handler", log_file=log_file, level="INFO")
        # Should have 2 handlers: stdout + file
        assert any(isinstance(h, logging.FileHandler) for h in log.handlers)

    def test_log_file_written(self, tmp_path):
        log_file = tmp_path / "output.log"
        log = get_logger("test.file_write", log_file=log_file, level="INFO")
        log.info("hello from test")
        content = log_file.read_text()
        assert "hello from test" in content


class TestSensitiveDataPolicy:
    """
    Policy tests: confirm that the modules never log raw embedding arrays.

    These tests intercept log records and assert that no numpy array
    representation appears in any message emitted by the recognizer
    or store during a normal identify() call.
    """

    def test_recognizer_does_not_log_embedding_values(
        self, tmp_settings, tmp_store
    ):
        from app.recognition import Recognizer
        import logging

        rng = np.random.default_rng(99)
        emb = rng.standard_normal(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        tmp_store.save("testuser", emb)

        rec = Recognizer(tmp_settings, tmp_store)

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        # Attach to every relevant logger
        for name in ("app.recognition.recognizer", "app.storage.face_store"):
            logging.getLogger(name).addHandler(handler)

        try:
            rec.identify(emb)
        finally:
            for name in ("app.recognition.recognizer", "app.storage.face_store"):
                logging.getLogger(name).removeHandler(handler)

        for record in records:
            msg = record.getMessage()
            # A serialised numpy array always contains "[" and multiple floats
            assert "[ " not in msg and "array(" not in msg, (
                f"Possible embedding leak in log message: {msg!r}"
            )

    def test_store_does_not_log_embedding_on_save(self, tmp_settings, tmp_store):
        import logging

        emb = np.ones(512, dtype=np.float32)
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture()
        logging.getLogger("app.storage.face_store").addHandler(handler)
        try:
            tmp_store.save("secureuser", emb)
        finally:
            logging.getLogger("app.storage.face_store").removeHandler(handler)

        for record in records:
            msg = record.getMessage()
            assert "[ " not in msg and "array(" not in msg, (
                f"Possible embedding leak on save: {msg!r}"
            )


class TestRetroactiveFileHandler:
    """
    Verify that the logger adds a file handler retroactively when
    a module was already imported (and its logger cached) before
    main() set up the log file path.
    """

    def test_file_handler_added_on_second_call(self, tmp_path):
        """
        First call without log_file → console only.
        Second call with log_file  → file handler added to same instance.
        """
        log_file = tmp_path / "retro.log"
        name = "test.retro.add"

        l1 = get_logger(name)                            # no file handler
        l2 = get_logger(name, log_file=log_file)         # should add file

        assert l1 is l2   # same instance
        assert any(isinstance(h, logging.FileHandler) for h in l2.handlers)

    def test_file_handler_not_duplicated_on_third_call(self, tmp_path):
        """Third call with same log_file must not add a second FileHandler."""
        log_file = tmp_path / "no_dup.log"
        name = "test.retro.nodup"

        get_logger(name)
        get_logger(name, log_file=log_file)
        get_logger(name, log_file=log_file)   # third call

        log = logging.getLogger(name)
        file_handlers = [h for h in log.handlers
                         if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_retroactive_handler_actually_writes(self, tmp_path):
        """Messages written AFTER the file handler is added must appear in file."""
        log_file = tmp_path / "write_after.log"
        name = "test.retro.write"

        log = get_logger(name)                          # console only
        log.info("before file — should NOT be in file")

        get_logger(name, log_file=log_file)             # adds file handler
        log.info("after file — MUST be in file")

        content = log_file.read_text()
        assert "after file" in content
        assert "before file" not in content

    def test_console_handler_not_duplicated(self, tmp_path):
        """Repeated calls must never add more than one StreamHandler."""
        name = "test.retro.console"
        get_logger(name)
        get_logger(name, log_file=tmp_path / "x.log")
        get_logger(name)

        log = logging.getLogger(name)
        stream_handlers = [h for h in log.handlers
                           if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1
