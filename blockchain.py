"""Livrable 3 - Registre de type blockchain (chaîne locale, preuve de travail légère).

Chaque bloc contient : index, horodatage de la simulation, horodatage réel (UTC),
agent émetteur, données de l'événement, hash du bloc précédent, nonce et hash.
Toute modification d'un bloc passé casse la chaîne (détectée par verify()).
"""
import hashlib
import json
from datetime import datetime, timezone

DIFFICULTY = 3  # le hash doit commencer par 3 zéros hexadécimaux


def _hash(block: dict) -> str:
    body = {k: block[k] for k in ("index", "sim_time", "utc_time", "agent", "data", "prev_hash", "nonce")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Blockchain:
    def __init__(self, difficulty=DIFFICULTY):
        self.difficulty = difficulty
        self.chain = []
        self._add("GENESIS", "-", {"note": "Registre du projet Deep Learning - DOT/USD (simulation, aucune transaction réelle)"})

    def _add(self, agent, sim_time, data):
        blk = dict(index=len(self.chain), sim_time=sim_time,
                   utc_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   agent=agent, data=data,
                   prev_hash=self.chain[-1]["hash"] if self.chain else "0" * 64, nonce=0)
        while True:
            h = _hash(blk)
            if h.startswith("0" * self.difficulty):
                break
            blk["nonce"] += 1
        blk["hash"] = h
        self.chain.append(blk)
        return blk

    def add_event(self, agent, sim_time, data):
        return self._add(agent, str(sim_time), data)

    def verify(self):
        return verify_chain(self.chain, self.difficulty)

    def export(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"difficulty": self.difficulty, "chain": self.chain}, f, ensure_ascii=False, indent=1)


def verify_chain(chain, difficulty=DIFFICULTY):
    """Retourne (ok, message)."""
    for i, b in enumerate(chain):
        if b["index"] != i:
            return False, f"index incohérent au bloc {i}"
        if _hash(b) != b["hash"]:
            return False, f"contenu modifié au bloc {i}"
        if not b["hash"].startswith("0" * difficulty):
            return False, f"preuve de travail invalide au bloc {i}"
        if i > 0 and b["prev_hash"] != chain[i - 1]["hash"]:
            return False, f"chaînage rompu au bloc {i}"
    return True, f"chaîne valide ({len(chain)} blocs)"
