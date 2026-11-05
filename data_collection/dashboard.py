"""Rich live telemetry; redirected output remains plain collector logs."""
import io
import os
import sys
import threading
import time
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def clean(value):
    return ''.join(c for c in str(value) if c.isprintable())


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
        self._lock = threading.RLock()
        self._live = None
        self.console = Console(file=self.stream, highlight=False)

    def update(self, **values):
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def event(self, message):
        with self._lock:
            self.events.append((time.strftime('%H:%M:%S'), clean(message)))

    @staticmethod
    def _metrics(rows):
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style='dim', overflow='fold')
        table.add_column(justify='right', style='bold white')
        for label, value in rows:
            table.add_row(Text(label), Text(value))
        return table

    def _frame(self, width, height=24):
        with self._lock:
            elapsed = max(0, time.monotonic() - self.started)
            rate = self.added / elapsed if elapsed else 0
            eta = duration(max(0, self.target - self.added) / rate) if rate else 'warming up'
            progress = Progress(
                BarColumn(bar_width=None, style='grey23', complete_style='cyan', finished_style='green'),
                TaskProgressColumn(), expand=True, auto_refresh=False,
            )
            progress.add_task('Collection', total=max(1, self.target), completed=self.added)
            summary = Text(f'{self.added:,}', style='bold cyan')
            summary.append(f' / {self.target:,} new unique battles', style='white')
            timing = Text(f'{rate * 60:,.1f}/min  •  ETA {eta}  •  UP {duration(elapsed)}', style='dim')
            collection = Panel(Group(summary, progress, timing), title='[bold cyan]COLLECTION', border_style='cyan')
            frontier = Panel(self._metrics([
                ('PLAYERS', f'{self.processed:,}'), ('FRONTIER', f'{self.queued:,}'),
                ('ARCHIVE', f'{self.existing + self.added:,}'),
            ]), title='[bold magenta]PLAYER FRONTIER', border_style='magenta')
            health = Panel(self._metrics([
                ('REQUESTS', f'{self.requests:,}'), ('RETRIES', f'{self.retries:,}'),
                ('DUPLICATES', f'{self.duplicates:,}'), ('FILTERED', f'{self.filtered:,}'),
            ]), title='[bold cyan]API / QUALITY', border_style='cyan')
            if width >= 72:
                metrics = Table.grid(expand=True, padding=(0, 1))
                metrics.add_column(ratio=1)
                metrics.add_column(ratio=1)
                metrics.add_row(frontier, health)
            else:
                metrics = Group(frontier, health)
            status = Text(clean(self.status), style='yellow' if 'backing off' in self.status else 'cyan')
            activity = Table.grid(expand=True, padding=(0, 1))
            activity.add_column(style='dim', no_wrap=True)
            activity.add_column(overflow='fold')
            # Reserve room for controls on short screens.
            count = max(0, min(5, height - (21 if width >= 72 else 27)))
            for stamp, message in list(self.events)[-count:] if count else []:
                style = 'yellow' if 'retry' in message.lower() or 'HTTP' in message else 'white'
                activity.add_row(Text(stamp), Text(message, style=style))
            if not self.events and count:
                activity.add_row('—', 'Waiting for collection activity')
            header = Text('CLASH / SYNERGY', style='bold magenta')
            header.append('  //  LIVE TELEMETRY', style='dim')
            items = [header, collection, metrics,
                     Spinner('dots', text=status, style='cyan'),
                     Text(f'PLAYER  {clean(self.player)}', style='dim')]
            if count:
                items.append(Panel(activity, title='[dim]RECENT ACTIVITY', border_style='grey37'))
            items.append(Text('CTRL+C  save progress & exit', style='dim'))
            return Group(*items)

    def render(self, width=80, height=24):
        """Plain, bounded offline preview using the same Rich layout."""
        stream = io.StringIO()
        console = Console(file=stream, width=max(16, width), color_system=None)
        console.print(self._frame(max(16, width), height))
        return stream.getvalue()

    def start(self):
        if self.enabled and self._live is None:
            self._live = Live(
                console=self.console, screen=True, refresh_per_second=5,
                get_renderable=lambda: self._frame(self.console.width, self.console.height),
            )
            try:
                self._live.start(refresh=True)
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._live is not None:
            live, self._live = self._live, None
            live.stop()
