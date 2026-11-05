"""Small, dependency-free live display; no terminal escapes in redirected logs."""
import os
import shutil
import sys
import threading
import time
from collections import deque


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


class Dashboard:
    def __init__(self, target, mode='auto', stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = mode != 'plain' and self.stream.isatty() and os.environ.get('TERM') != 'dumb'
        self.target = target
        self.started = time.monotonic()
        self.added = self.existing = self.processed = self.queued = 0
        self.requests = self.retries = self.duplicates = self.filtered = 0
        self.player = '—'
        self.status = 'Rebuilding saved battle index'
        self.events = deque(maxlen=5)
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread = None

    def update(self, **values):
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def event(self, message):
        with self._lock:
            self.events.append((time.strftime('%H:%M:%S'), message))

    def render(self, width=80):
        """Return a frame with bounded lines, also usable for offline previews."""
        with self._lock:
            width = max(16, min(width, 100))
            elapsed = time.monotonic() - self.started
            rate = self.added / elapsed if elapsed > 0 else 0
            eta = duration((self.target - self.added) / rate) if rate else 'warming up'
            fraction = min(1, self.added / self.target)
            bar_width = max(4, width - 22)
            filled = int(bar_width * fraction)
            bar = '━' * filled + '─' * (bar_width - filled)
            lines = [
                ('CLASH / SYNERGY', 'title'),
                ('BATTLE COLLECTION  //  LIVE TELEMETRY', 'muted'),
                ('', ''),
                (f'[{bar}] {fraction:6.1%}', 'cyan'),
                (f'{self.added:,} / {self.target:,} new unique battles', 'title'),
                (f'{rate * 60:,.1f} battles/min   ETA {eta}   UP {duration(elapsed)}', ''),
                ('', ''),
                (f'PLAYERS {self.processed:,}   FRONTIER {self.queued:,}   ARCHIVE {self.existing + self.added:,}', 'cyan'),
                (f'REQUESTS {self.requests:,}   RETRIES {self.retries:,}', ''),
                (f'DUPLICATES {self.duplicates:,}   FILTERED {self.filtered:,}', 'muted'),
                (f'PLAYER {self.player}', ''),
                (f'{"◐◓◑◒"[int(elapsed * 4) % 4]} {self.status}', 'cyan'),
                ('', ''),
                ('ACTIVITY ' + '─' * max(0, width - 9), 'muted'),
            ]
            lines.extend((f'{stamp}  {message}', '') for stamp, message in self.events)
            lines.extend([('', '')] * (5 - len(self.events)))
            lines.extend([('', ''), ('CTRL+C  save progress & exit', 'muted')])
            colors = {'title': '\033[1;95m', 'cyan': '\033[96m', 'muted': '\033[90m', '': '\033[0m'}
            # Remove control characters from external player/API values.
            return '\n'.join(colors[style] + ''.join(c for c in line if c.isprintable())[:width - 1] + '\033[0m' for line, style in lines)

    def _draw(self):
        size = shutil.get_terminal_size((80, 24))
        frame = self.render(size.columns).splitlines()[:max(1, size.lines - 1)]
        self.stream.write('\033[H' + '\n'.join(line + '\033[K' for line in frame) + '\033[J')
        self.stream.flush()

    def _refresh(self):
        while not self._stop.wait(0.2):
            self._draw()

    def start(self):
        if self.enabled:
            self.stream.write('\033[?1049h\033[?25l')
            self._draw()
            self._thread = threading.Thread(target=self._refresh, daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self.enabled:
            self.stream.write('\033[?25h\033[?1049l')
            self.stream.flush()
