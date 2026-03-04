"""
Vercel Serverless Function for Flask App
This handler serves the Flask application on Vercel's serverless platform.
"""

import os
import sys
from io import BytesIO

# Add parent directory to path to import app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure database path for Vercel (use /tmp for writable storage)
# Note: SQLite on Vercel is not ideal for production due to serverless nature
# Consider using a managed database like PostgreSQL, MySQL, or MongoDB
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/hr_demo.db')

# Import Flask app
from app import create_app

# Create Flask app instance (module-level for reuse across invocations)
flask_app = create_app()

def handler(request):
    """
    Vercel serverless function handler for Flask app
    Converts Vercel request to WSGI and processes through Flask
    """
    # Handle different request formats (dict or object with attributes)
    if isinstance(request, dict):
        method = request.get('method', 'GET')
        path = request.get('path', '/')
        headers = request.get('headers', {})
        body = request.get('body', b'')
        query_string = request.get('queryString', '') or request.get('query_string', '')
    else:
        # Handle object-style request
        method = getattr(request, 'method', 'GET')
        path = getattr(request, 'path', '/')
        headers = getattr(request, 'headers', {})
        body = getattr(request, 'body', b'')
        query_string = getattr(request, 'queryString', '') or getattr(request, 'query_string', '')
    
    # Normalize headers to dict if needed
    if not isinstance(headers, dict):
        headers = dict(headers) if hasattr(headers, 'items') else {}
    
    # Handle body encoding
    if isinstance(body, str):
        body_bytes = body.encode('utf-8')
    elif body is None:
        body_bytes = b''
    else:
        body_bytes = body
    
    # Extract host from headers
    host = headers.get('host', headers.get('Host', 'localhost'))
    server_name = host.split(':')[0] if ':' in host else host
    server_port = host.split(':')[1] if ':' in host else '443'
    
    # Build WSGI environ dictionary
    environ = {
        'REQUEST_METHOD': method,
        'PATH_INFO': path,
        'QUERY_STRING': query_string,
        'SERVER_NAME': server_name,
        'SERVER_PORT': server_port,
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https',
        'wsgi.input': BytesIO(body_bytes),
        'wsgi.errors': sys.stderr,
        'wsgi.multithread': False,
        'wsgi.multiprocess': True,
        'wsgi.run_once': False,
        'CONTENT_TYPE': headers.get('content-type', headers.get('Content-Type', '')),
        'CONTENT_LENGTH': str(len(body_bytes)),
    }
    
    # Add HTTP headers to environ
    for key, value in headers.items():
        key_upper = key.upper().replace('-', '_')
        if key_upper not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            environ['HTTP_' + key_upper] = str(value)
    
    # Response storage
    response_data = []
    status_code = [200]
    response_headers = []
    
    def start_response(status, headers_list):
        status_code[0] = int(status.split()[0])
        response_headers[:] = headers_list
    
    # Call Flask app
    try:
        app_iter = flask_app(environ, start_response)
        
        # Collect response body
        try:
            for data in app_iter:
                response_data.append(data)
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
    except Exception as e:
        # Handle errors gracefully
        import traceback
        error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': f'{{"error": "{error_msg}"}}'
        }
    
    # Build response body
    body_result = b''.join(response_data)
    
    # Convert headers to dict (normalize keys)
    headers_dict = {}
    for key, value in response_headers:
        # Use title case for standard headers
        key_lower = key.lower()
        if key_lower == 'content-type':
            headers_dict['Content-Type'] = value
        elif key_lower == 'content-length':
            headers_dict['Content-Length'] = value
        else:
            headers_dict[key] = value
    
    # Return Vercel response format
    return {
        'statusCode': status_code[0],
        'headers': headers_dict,
        'body': body_result.decode('utf-8') if isinstance(body_result, bytes) else str(body_result)
    }
