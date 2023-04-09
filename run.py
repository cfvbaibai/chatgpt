"""
This script runs the beopWeb application using a development server.
"""
from web.views import app

if __name__ == '__main__':
    app.run('0.0.0.0', 5000, use_reloader=False, threaded=True, debug=True)
