import os
import threading
import time
from typing import Callable, Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover - optional dependency
    FileSystemEventHandler = object
    Observer = None


class _NewFileHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None], valid_ext: tuple[str, ...]):
        super().__init__()
        self.callback = callback
        self.valid_ext = valid_ext

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if not path.lower().endswith(self.valid_ext):
            return

        # Wait briefly for FTP writes to finish.
        time.sleep(0.4)
        self.callback(path)


class FTPFolderWatcher:
    """Watch an FTP drop folder and trigger processing/storage in real-time."""

    def __init__(
        self,
        watch_dir: str,
        on_new_image: Callable[[str], None],
        valid_ext: Optional[tuple[str, ...]] = None,
    ):
        self.watch_dir = watch_dir
        self.on_new_image = on_new_image
        self.valid_ext = valid_ext or (".jpg", ".jpeg", ".png", ".webp")
        self._observer: Optional[Observer] = None

    def start(self):
        if Observer is None:
            raise RuntimeError("watchdog is not installed")
        os.makedirs(self.watch_dir, exist_ok=True)
        handler = _NewFileHandler(self.on_new_image, self.valid_ext)
        self._observer = Observer()
        self._observer.schedule(handler, self.watch_dir, recursive=False)
        self._observer.start()

    def stop(self):
        if self._observer is None:
            return
        self._observer.stop()
        self._observer.join(timeout=3)
        self._observer = None


def start_watcher_thread(watcher: FTPFolderWatcher) -> threading.Thread:
    t = threading.Thread(target=watcher.start, daemon=True)
    t.start()
    return t
