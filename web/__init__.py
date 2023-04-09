from flask import Flask
import web.views

app = Flask(__name__, static_folder='static')
app.url_map.strict_slashes = False
