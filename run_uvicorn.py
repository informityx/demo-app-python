"""
Run Flask app with uvicorn.
Flask is WSGI so we wrap it with WsgiToAsgi. Reload is off by default so
background threads (e.g. async CV evaluation) don't break the ASGI executor.
For watch mode set RELOAD=true. For heavy CV evaluation, python run.py avoids ASGI.
"""
from app import create_app
from asgiref.wsgi import WsgiToAsgi
import uvicorn
import os

# Create Flask app
flask_app = create_app()

# Wrap Flask WSGI app with ASGI adapter for uvicorn
app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5001))
    use_reload = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")

    # reload=False by default: with reload on, asgiref's CurrentThreadExecutor can
    # break when background threads run (e.g. async CV evaluation), causing 500s on status polls.
    # Set RELOAD=true for file watching during development if you don't use async CV evaluation.
    uvicorn.run(
        "run_uvicorn:app",
        host="0.0.0.0",
        port=port,
        reload=use_reload,
        reload_dirs=["./app"] if use_reload else None,
        log_level="info",
        access_log=True,
    )
