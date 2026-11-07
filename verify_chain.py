"""Vérification autonome du registre : python3 verify_chain.py outputs/ledger.json"""
import json
import sys

from blockchain import verify_chain

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/ledger.json"
with open(path, encoding="utf-8") as f:
    d = json.load(f)
ok, msg = verify_chain(d["chain"], d["difficulty"])
print(("VALIDE : " if ok else "INVALIDE : ") + msg)
sys.exit(0 if ok else 1)
