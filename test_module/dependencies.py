# -*- coding: utf-8 -*-
"""Check the plugin's third-party Python packages before opening the dialog.

Every package is actually imported, rather than only located, because a package
can be present yet broken. Installing the unrelated PyPI package ``stl`` on top
of ``numpy-stl`` is the known example: both use the ``stl`` import name, and the
result fails with "module 'stl' has no attribute 'BaseStl'".

This module must only import the standard library and ``qgis.PyQt`` (lazily), so
that it still works when every dependency is missing.
"""

import importlib
import importlib.metadata
import os
import sys
from collections import namedtuple

Dependency = namedtuple('Dependency', 'module pip_name required purpose')
Problem = namedtuple('Problem', 'dependency error conflict')

DEPENDENCIES = (
    Dependency('numpy', 'numpy', True, 'numerical operations'),
    Dependency('scipy', 'scipy', True, 'interpolation and spatial indexing'),
    Dependency('pandas', 'pandas', True, 'reading and rewriting input CSVs'),
    Dependency('shapely', 'shapely', True, 'AOI clipping and rotation'),
    Dependency('rasterio', 'rasterio', True, 'reading the DTM, GeoTIFF export'),
    Dependency('pyvista', 'pyvista', True, '3D meshes and rendering'),
    Dependency('pyvistaqt', 'pyvistaqt', True, '3D viewer inside the dialog'),
    Dependency('gempy', 'gempy', True, 'geological modelling engine'),
    Dependency('gempy_viewer', 'gempy-viewer', True, 'model visualisation'),
    Dependency('stl.mesh', 'numpy-stl', False, 'Topography.stl export'),
)


def qgis_python_executable():
    """Return the real Python interpreter of this QGIS, or None if not found.

    Inside QGIS on Windows ``sys.executable`` is the ``qgis-bin`` launcher, which
    can neither run pip nor host a ``multiprocessing`` pool.
    """
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith('python'):
        return exe
    candidates = [
        os.path.join(sys.exec_prefix, 'python.exe'),
        os.path.join(sys.exec_prefix, 'pythonw.exe'),
        os.path.join(sys.exec_prefix, 'python3.exe'),
        os.path.join(sys.exec_prefix, 'bin', 'python3'),
        os.path.join(sys.exec_prefix, 'bin', 'python'),
        os.path.join(sys.base_exec_prefix, 'python.exe'),
        os.path.join(sys.base_exec_prefix, 'bin', 'python3'),
        os.path.join(sys.base_exec_prefix, 'bin', 'python'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _has_conflicting_stl():
    """True when the unrelated PyPI package ``stl`` is installed."""
    try:
        importlib.metadata.distribution('stl')
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def check_dependencies(dependencies=DEPENDENCIES):
    """Import every dependency and return a Problem for each one that fails."""
    importlib.invalidate_caches()
    problems = []
    for dep in dependencies:
        try:
            importlib.import_module(dep.module)
        except Exception as exc:  # a broken package can raise anything
            conflict = dep.pip_name == 'numpy-stl' and (
                _has_conflicting_stl() or isinstance(exc, AttributeError))
            problems.append(
                Problem(dep, f"{type(exc).__name__}: {exc}", conflict))
    return problems


def install_commands(problems, python=None):
    """Return the shell commands, one per line, that fix ``problems``."""
    python = python or qgis_python_executable() or 'python'
    pip = f'"{python}" -m pip'
    lines = []
    if any(p.conflict for p in problems):
        lines.append(f'{pip} uninstall -y stl numpy-stl')
    names = [p.dependency.pip_name for p in problems]
    if names:
        lines.append(f'{pip} install {" ".join(names)}')
    return '\n'.join(lines)


def _report_html(problems):
    def rows(items):
        return ''.join(
            f'<li><b>{p.dependency.pip_name}</b> '
            f'({p.dependency.purpose})</li>' for p in items)

    required = [p for p in problems if p.dependency.required]
    optional = [p for p in problems if not p.dependency.required]
    html = ''
    if required:
        html += ('<p><b>Required, the plugin cannot start without these:</b>'
                 f'</p><ul>{rows(required)}</ul>')
    if optional:
        html += ('<p><b>Optional, the plugin works without this feature:</b>'
                 f'</p><ul>{rows(optional)}</ul>')
    if any(p.conflict for p in problems):
        html += ('<p>The unrelated package <b>stl</b> overwrote files of '
                 '<b>numpy-stl</b>, as both use the same import name. '
                 'Both must be removed, then numpy-stl reinstalled. '
                 'Do not install <b>stl</b> again.</p>')
    html += ('<p>Close QGIS, run these commands in a terminal (on Windows, '
             'as Administrator), then restart QGIS:</p>')
    return html


def show_dependency_dialog(problems, parent=None):
    """Report ``problems``. Return True if the plugin may open anyway."""
    from qgis.PyQt.QtWidgets import (
        QApplication, QDialog, QDialogButtonBox, QLabel, QPlainTextEdit,
        QPushButton, QTextBrowser, QVBoxLayout,
    )

    can_continue = all(not p.dependency.required for p in problems)
    commands = install_commands(problems)

    dlg = QDialog(parent)
    dlg.setWindowTitle('3D GemPy Model Builder: Python packages')
    dlg.setMinimumWidth(640)
    layout = QVBoxLayout(dlg)

    report = QTextBrowser()
    report.setHtml(_report_html(problems))
    report.setMinimumHeight(300)
    layout.addWidget(report, 1)

    command_box = QPlainTextEdit(commands)
    command_box.setReadOnly(True)
    command_box.setFixedHeight(110)
    layout.addWidget(command_box)

    details = QPlainTextEdit('\n'.join(
        f'{p.dependency.module}: {p.error}' for p in problems))
    details.setReadOnly(True)
    details.setFixedHeight(80)
    layout.addWidget(QLabel('Errors:'))
    layout.addWidget(details)

    buttons = QDialogButtonBox()
    copy_button = QPushButton('Copy commands')
    copy_button.clicked.connect(
        lambda: QApplication.clipboard().setText(commands))
    buttons.addButton(copy_button, QDialogButtonBox.ActionRole)
    buttons.addButton(QDialogButtonBox.Close).clicked.connect(dlg.reject)
    if can_continue:
        buttons.addButton('Continue without it', QDialogButtonBox.AcceptRole
                          ).clicked.connect(dlg.accept)
    layout.addWidget(buttons)

    return bool(dlg.exec_()) and can_continue
