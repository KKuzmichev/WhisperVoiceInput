"""Menubar-индикатор для macOS через rumps.

Значок — приведение: чёрное во время записи, белое в простое.
"""

import rumps
from AppKit import (
    NSImage, NSColor, NSBezierPath, NSSize, NSPoint, NSRect,
)


def _ghost_image(color):
    img = NSImage.alloc().initWithSize_(NSSize(24.0, 24.0))
    img.lockFocus()

    body = NSBezierPath.alloc().init()
    # Старт у левого края волнистого низа.
    body.moveToPoint_(NSPoint(4.0, 6.0))
    # Волнистый низ: три полудуги провисают ВНИЗ (против часовой, 180°→360°).
    for x in (6.67, 12.0, 17.33):
        body.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_(
            NSPoint(x, 6.0), 2.67, 180.0, 360.0
        )
    # Точка (20,6). Поднимаемся к правому краю купола.
    # Купол: полуокружность снизу вверх (против часовой), 180°→360°,
    # центр (12,12), радиус 8 — от (4,12) до (20,12) по верху.
    body.lineToPoint_(NSPoint(20.0, 12.0))
    body.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_(
        NSPoint(12.0, 12.0), 8.0, 360.0, 180.0
    )
    body.closePath()

    # Глаза-дырки (even-odd вырезает их из заливки).
    for ex in (9.0, 15.0):
        eye = NSBezierPath.bezierPathWithOvalInRect_(NSRect((ex - 1.5, 15.0), (3.0, 4.0)))
        body.appendBezierPath_(eye)
    body.setWindingRule_(1)

    color.set()
    body.fill()

    img.unlockFocus()
    img.setTemplate_(False)
    return img


class MenubarIndicator(rumps.App):
    def __init__(self, voice_app):
        super().__init__(
            "WhisperVoiceInput",
            title=None,
            icon=None,
            quit_button=None,
        )
        self._ghosts = {
            "recording": _ghost_image(NSColor.blackColor()),
            "idle": _ghost_image(NSColor.whiteColor()),
        }
        self._icon_nsimage = self._ghosts["idle"]
        self.voice_app = voice_app
        self.menu = [
            "WhisperVoiceInput",
            None,
            rumps.MenuItem("Выход", callback=self._on_quit),
        ]

    def set_state(self, state):
        self._icon_nsimage = self._ghosts.get(state, self._ghosts["idle"])
        try:
            self._nsapp.setStatusBarIcon()
        except AttributeError:
            pass

    def _on_quit(self, _sender):
        rumps.quit_application()