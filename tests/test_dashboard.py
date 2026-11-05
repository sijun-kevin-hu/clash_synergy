import io
import re
import unittest
from unittest.mock import patch

from data_collection.dashboard import Dashboard


class Terminal(io.StringIO):
    def isatty(self):
        return True


class DashboardTests(unittest.TestCase):
    def test_redirected_output_has_no_terminal_controls(self):
        stream = io.StringIO()
        dashboard = Dashboard(100, stream=stream)
        dashboard.start()
        dashboard.close()
        self.assertFalse(dashboard.enabled)
        self.assertEqual(stream.getvalue(), '')

    def test_frame_counts_and_narrow_terminal(self):
        dashboard = Dashboard(100)
        dashboard.update(added=25, existing=1000, processed=4, queued=20)
        dashboard.event('Untrusted\033[2J\nplayer')
        frame = dashboard.render(80)
        self.assertIn('25%', frame)
        self.assertIn('ARCHIVE', frame)
        self.assertIn('1,025', frame)
        self.assertNotIn('\033[2J', frame)
        narrow = re.sub(r'\033\[[0-9;]*m', '', dashboard.render(28))
        self.assertTrue(all(len(line) <= 28 for line in narrow.splitlines()))

    def test_external_text_is_literal_and_plain_mode_stays_quiet(self):
        dashboard = Dashboard(100, mode='plain', stream=Terminal())
        dashboard.event('[red]literal activity[/red]')
        self.assertIn('[red]literal activity[/red]', dashboard.render(100, 32))
        dashboard.start()
        dashboard.close()
        self.assertEqual(dashboard.stream.getvalue(), '')

    def test_terminal_screen_and_cursor_are_restored(self):
        stream = Terminal()
        with patch.dict('os.environ', {'TERM': 'xterm-256color'}):
            dashboard = Dashboard(100, stream=stream)
            try:
                dashboard.start()
                self.assertTrue(dashboard.enabled)
            finally:
                dashboard.close()
        self.assertIn('\033[?1049h', stream.getvalue())
        self.assertIn('\033[?25l', stream.getvalue())
        self.assertIn('\033[?25h', stream.getvalue())
        self.assertIn('\033[?1049l', stream.getvalue())
        self.assertIsNone(dashboard._live)


if __name__ == '__main__':
    unittest.main()
