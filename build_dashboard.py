"""Livrable 4 - Génère outputs/dashboard.html à partir de outputs/results.json."""
import json

data = json.load(open("outputs/results.json", encoding="utf-8"))
tpl = open("dashboard_template.html", encoding="utf-8").read()
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
open("outputs/dashboard.html", "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", payload))
print("outputs/dashboard.html écrit")
