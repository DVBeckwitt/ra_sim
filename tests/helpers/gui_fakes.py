"""Shared fake GUI objects for RA-SIM tests."""


class DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class DummyAxis:
    def __init__(self, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
        self._xlim = tuple(float(v) for v in xlim)
        self._ylim = tuple(float(v) for v in ylim)

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, left, right):
        self._xlim = (float(left), float(right))

    def set_ylim(self, bottom, top):
        self._ylim = (float(bottom), float(top))


class DummyCanvas:
    def __init__(self) -> None:
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


class DummySlider:
    def __init__(self, from_value, to_value):
        self._values = {
            "from": from_value,
            "to": to_value,
        }

    def cget(self, key):
        return self._values[key]


RuntimeVar = DummyVar
