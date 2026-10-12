"""Explicit source kinds; class names are defaults, never policy identifiers."""
import re


def source_kind(label, kind=None):
    if kind is not None:
        if kind not in ('knock', 'chime', 'kettle', 'other'):
            raise ValueError('kind must be knock, chime, kettle or other')
        return kind
    name = re.sub(r'[\s-]+', '_', label.casefold())
    if name in ('knock', 'knocks', 'knocking', 'door_knock', 'door_knocks', 'door_knocking'):
        return 'knock'
    return {'doorbell': 'chime', 'chime': 'chime', 'kettle': 'kettle'}.get(name, 'other')
