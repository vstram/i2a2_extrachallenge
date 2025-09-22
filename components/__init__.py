"""
Components package for Streamlit AI Analytics Chatbot.

This package contains UI components for file upload, chat interface,
and statistics display functionality.
"""

from .file_uploader import FileUploader, render_file_uploader

__all__ = ['FileUploader', 'render_file_uploader']