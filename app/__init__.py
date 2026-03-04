from flask import Flask, request, Response, redirect
from flask_cors import CORS
from app.config import Config
from app.database import db, init_db

# Make flasgger optional - it requires building from source which fails on Vercel
try:
    from flasgger import Swagger
    FLASGGER_AVAILABLE = True
except ImportError:
    FLASGGER_AVAILABLE = False
    Swagger = None


def create_app(config_class=Config):
    """Flask application factory"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins
    
    # Register blueprints first (needed for Swagger to discover routes)
    from app.api import auth as auth_bp, hr as hr_bp, autosphere as autosphere_bp
    app.register_blueprint(auth_bp.bp, url_prefix='/api/auth')
    app.register_blueprint(hr_bp.bp, url_prefix='/api/hr')
    app.register_blueprint(autosphere_bp.bp, url_prefix='/api/autosphere')
    
    # Initialize Swagger after blueprints are registered (optional)
    if FLASGGER_AVAILABLE and Swagger is not None:
        swagger_config = {
            "headers": [],
            "specs": [
                {
                    "endpoint": 'apispec',
                    "route": '/apispec.json',
                    "rule_filter": lambda rule: True,
                    "model_filter": lambda tag: True,
                }
            ],
            "static_url_path": "/flasgger_static",
            "swagger_ui": True,
            "specs_route": "/swagger"
        }
        
        swagger_template = {
            "swagger": "2.0",
            "info": {
                "title": "Enterprise AI Dashboard API",
                "description": "Flask REST API for HR AI Platform and AutoSphere Motors AI Assistant",
                "version": "1.0.0"
            },
            "basePath": "/",
            "schemes": ["http", "https"],
            "securityDefinitions": {
                "Bearer": {
                    "type": "apiKey",
                    "name": "Authorization",
                    "in": "header",
                    "description": "JWT Authorization header using the Bearer scheme"
                }
            },
            "tags": [
                {
                    "name": "Authentication",
                    "description": "User authentication endpoints"
                },
                {
                    "name": "HR AI Platform",
                    "description": "HR AI Platform endpoints"
                },
                {
                    "name": "AutoSphere Motors",
                    "description": "AutoSphere Motors AI Assistant endpoints"
                }
            ]
        }
        
        try:
            swagger = Swagger(app, config=swagger_config, template=swagger_template)
            print("Swagger UI initialized successfully")
        except Exception as e:
            print(f"Warning: Swagger initialization error: {e}")
            # Continue without Swagger if there's an error
    else:
        print("Warning: flasgger not available. Swagger UI will not be available.")
        # Create a simple info endpoint instead
        @app.route('/swagger')
        def swagger_info():
            return {
                'message': 'Swagger UI is not available',
                'reason': 'flasgger package is not installed (requires building from source)',
                'note': 'Install flasgger locally for Swagger UI documentation',
                'endpoints': {
                    'auth': '/api/auth',
                    'hr': '/api/hr',
                    'autosphere': '/api/autosphere'
                }
            }, 200
    
    # Redirect root path to Swagger UI (or info page if Swagger not available)
    @app.route('/')
    def index():
        return redirect('/swagger')
    
    # Protect Swagger UI with basic auth (only if flasgger is available)
    if FLASGGER_AVAILABLE:
        @app.before_request
        def protect_swagger():
            # Protect the Swagger UI page itself
            if request.path == '/swagger' or (request.path.startswith('/swagger/') and not request.path.startswith('/swagger/static')):
                auth = request.authorization
                if not auth or auth.username != 'swagger' or auth.password != '!!swagger!!':
                    return Response(
                        'Swagger UI requires authentication.\n'
                        'Username: swagger\n'
                        'Password: !!swagger!!', 401,
                        {'WWW-Authenticate': 'Basic realm="Swagger UI Login Required"'})
            # Protect static files
            elif request.path.startswith('/flasgger_static'):
                auth = request.authorization
                if not auth or auth.username != 'swagger' or auth.password != '!!swagger!!':
                    return Response(
                        'Swagger UI requires authentication.\n'
                        'Username: swagger\n'
                        'Password: !!swagger!!', 401,
                        {'WWW-Authenticate': 'Basic realm="Swagger UI Login Required"'})
            # Allow /apispec.json to be accessed (Swagger UI needs it to load the spec)
    
    # Initialize database with error handling
    # On Vercel, database initialization might fail due to file system limitations
    try:
        with app.app_context():
            init_db()
    except Exception as e:
        # Log error but don't fail app creation
        # Database will be created on first use if needed
        print(f"Warning: Database initialization failed: {e}")
        print(f"This is expected on Vercel if using SQLite. Consider using a managed database.")
    
    return app
