from flask import render_template, request

from chat.dialog import Dialog
from web import app

SESSION_ASSISTANT_DIALOG = "assistant_dialog"


@app.errorhandler(Exception)
def handle_exception(e):
    return {"error": str(e)}


@app.get("/")
def page_index():
    return render_template("index.html")


@app.get("/pages/assistant")
def page_assistant():
    return render_template("assistant.html")


dialogs = {}


@app.post("/api/assistant/ask")
def api_assistant_ask():
    payload = request.get_json()
    if "q" not in payload:
        return {"error": "no q specified"}
    q = payload["q"]
    if "key" not in payload:
        return {"error": "no key specified"}
    key = payload["key"]
    if key not in dialogs:
        dialogs[key] = Dialog()
        dialogs[key].set_cpu_role("你是上海仁力名才公司的一位人力资源专家")
    dialog = dialogs[key]
    answer = dialog.ask(q)
    return {"data": {**answer}}
