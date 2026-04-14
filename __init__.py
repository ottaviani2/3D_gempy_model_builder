# -*- coding: utf-8 -*-
"""QGIS plugin entry point for 3D GemPy Model Builder."""


def classFactory(iface):
    from .test_module import classFactory as _classFactory
    return _classFactory(iface)
