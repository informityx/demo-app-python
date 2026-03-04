"""
Vercel Serverless Function for Flask App
Vercel automatically detects Flask apps when it finds an 'app' variable.
This file exports the Flask app instance for Vercel's built-in WSGI handler.
"""

import os
import sys
import traceback

# Add parent directory to path to import app
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(parent_dir))

# Configure database path for Vercel (use /tmp for writable storage)
# Note: SQLite on Vercel is not ideal for production due to serverless nature
# Consider using a managed database like PostgreSQL, MySQL, or MongoDB
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/hr_demo.db')

# Import Flask - required dependency
from flask import Flask

# Import and create Flask app with error handling
# Vercel expects this to be named 'app' for automatic Flask detection
# Vercel will automatically handle WSGI conversion - no custom handler needed
try:
    from app import create_app
    app = create_app()
except Exception as e:
    # If app creation fails, create a minimal error app
    print(f"ERROR: Failed to create Flask app: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Create minimal Flask app that shows error
    app = Flask(__name__)
    
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def error_handler(path):
        error_info = {
            'error': 'Flask app initialization failed',
            'message': str(e),
            'path': path,
            'note': 'Check Vercel logs for full traceback'
        }
        return error_info, 500
