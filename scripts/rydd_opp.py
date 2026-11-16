"""Rydder i scripts/ og src/.

Hva skriptet gjør:
    1. Erstatter src/classification/gesture.py (gammel, uten EARLY) med
       gesture_2.py, som er den spillet og testene faktisk bruker, og retter
       alle "gesture_2" i importene til "gesture". Da bruker alle skriptene
       samme regel, slik rapporten sier.
    2. Flytter skript som ikke brukes lenger til old/ i repo-rota: de som er
       erstattet av nyere, de som bare laget First Release-tallene, og tomme
       filer. Ingenting slettes. (Skriptene i old/ kjører ikke derfra uten å
       rette ROOT.)
    3. Gir to skript bedre navn.
    4. Flytter evalueringsskriptene til scripts/eval/ og retter ROOT i dem,
       så de finner data/ og models/ som før.
    5. Fjerner mapper som er tomme etterpå (src/capture, src/game, src/ui).

data/, models/ og results/ blir ikke rørt.

Kjør først med DRY_RUN = True og les listen. Commit alt du har før du kjører
med DRY_RUN = False, så kan du angre med git hvis noe blir feil.

Kjør fra repo-rota:
    python scripts/rydd_opp.py
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRY_RUN = False

# erstattet av nyere skript, eller tomme. Flyttes til old/
DELETE = {
    "scripts/check_labels.py": "gammel sti (data/recordings/recordings), erstattet av check_boxes.py og review.py",
    "scripts/check_labels_v2.py": "gammel sti og gammel regel, erstattet av check_boxes.py",
    "scripts/datachek.py": "report_numbers.py seksjon 1 teller kast og bilder per sesjon",
    "scripts/eval_decider.py": "erstattet av sweep_decider.py på testrundene",
    "scripts/eval_gesture.py": "bare MediaPipe med fingerstrekk 0.5, erstattet av eval_gest_model.py",
    "scripts/live_pipeline_v2.py": "erstattet av live_gesture.py og play_game.py",
    "scripts/preprocess_own_v2.py": "lagde detector_192_palm_alan.npz, erstattet av prepros_det.py",
    "src/capture/camera.py": "tom",
    "src/classification/model.py": "tom",
    "src/classification/train.py": "tom",
    "src/game/game_logic.py": "tom",
    "src/ui/display.py": "tom",
    "src/main.py": "tom",
}

# bare brukt til First Release-tallene. Flyttes også til old/
ARCHIVE = {
    "scripts/eval_det.py": "First Release-detektoren, IoU 0.906 mot 0.451 (tabellen om beskjæring)",
    "scripts/find_palm_ratio.py": "regnet ut beskjæringsfaktoren 2.95 én gang",
    "scripts/preprocess_own.py": "den gamle beskjæringen med 40 % margin",
    "scripts/diagnose_tracking.py": "tracking-testen fra First Release, tracking er av",
    "src/detection/train_detector.py": "First Release-detektoren uten negative bilder",
    "src/detection/train_keypoints.py": "direkte regresjon, ikke brukt videre",
}

RENAME = {
    "scripts/cam_fps.py": "scripts/measure_latency_video.py",   # måler glass til glass, ikke fps
    "scripts/review manuell.py": "scripts/review_manual.py",     # mellomrom i filnavn er tungvint
}

# evalueringsskript som flyttes til scripts/eval/ (etter nye navn)
EVAL = [
    "scripts/eval_gest_model.py",
    "scripts/eval_models.py",
    "scripts/sweep_decider.py",
    "scripts/debug_round.py",
    "scripts/lock_vs_form.py",
    "scripts/lock_vs_formed.py",
    "scripts/early_analysis.py",
    "scripts/report_numbers.py",
    "scripts/measure_latency_video.py",
    "scripts/diagnose_full_v2.py",
    "scripts/view_person.py",
    "scripts/live_gesture.py",
    "scripts/count_no_hand.py",
]

EMPTY_DIRS = ["src/capture", "src/game", "src/ui"]


def git_dirty():
    try:
        out = subprocess.run(["git", "status", "--porcelain", "scripts", "src"], cwd=ROOT,
                             capture_output=True, text=True)
        # rydd_opp.py selv teller ikke
        lines = [l for l in out.stdout.splitlines() if "rydd_opp.py" not in l]
        return "\n".join(lines).strip()
    except FileNotFoundError:
        return None


print("DRY_RUN = True, ingenting endres. Les listen og sett DRY_RUN = False.\n" if DRY_RUN
      else "Endrer filene nå.\n")

dirty = git_dirty()
if dirty is None:
    print("Fant ikke git. Ta en kopi av scripts/ og src/ før du kjører med DRY_RUN = False.\n")
elif dirty:
    print("Det er endringer som ikke er committet i scripts/ eller src/:")
    print("\n".join("   " + line for line in dirty.splitlines()[:20]))
    if not DRY_RUN:
        print("\nCommit dem først, så kan du angre oppryddingen med git. Stopper.")
        raise SystemExit
    print()

# 1. gesture_2.py blir gesture.py
g_old = ROOT / "src/classification/gesture.py"
g_new = ROOT / "src/classification/gesture_2.py"
if g_new.exists():
    print("1. src/classification/gesture_2.py erstatter gesture.py (den gamle uten EARLY slettes)")
    if not DRY_RUN:
        g_new.replace(g_old)
else:
    print("1. gesture_2.py finnes ikke, hopper over")

changed = []
for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").rglob("*.py")):
    if path.name == "rydd_opp.py":
        continue
    text = path.read_text(encoding="utf-8")
    if "gesture_2" in text:
        changed.append(path.relative_to(ROOT))
        if not DRY_RUN:
            path.write_text(text.replace("gesture_2", "gesture"), encoding="utf-8")
print(f"   retter importen i {len(changed)} filer: " + ", ".join(str(p.name) for p in changed))

# 2. flytt det som ikke brukes til old/
print("\n2. Flyttes til old/ (ikke i bruk lenger):")
for rel, why in list(DELETE.items()) + list(ARCHIVE.items()):
    path = ROOT / rel
    if not path.exists():
        print(f"   (finnes ikke) {rel}")
        continue
    print(f"   {rel:38s} {why}")
    if not DRY_RUN:
        target = ROOT / "old" / Path(rel).name
        target.parent.mkdir(exist_ok=True)
        shutil.move(str(path), str(target))
if not DRY_RUN and (ROOT / "old").exists():
    (ROOT / "old" / "README.md").write_text(
        "Skript som ikke er i bruk lenger: erstattet av nyere skript, brukt bare til\n"
        "tallene i First Release-rapporten, eller tomme. De kjører ikke herfra uten å rette ROOT.\n",
        encoding="utf-8")

# 3. nye navn
print("\n3. Nye navn:")
for old, new in RENAME.items():
    path = ROOT / old
    if not path.exists():
        print(f"   (finnes ikke) {old}")
        continue
    print(f"   {old}  ->  {new}")
    if not DRY_RUN:
        path.rename(ROOT / new)

# 4. evalueringsskript til scripts/eval/
print("\n4. Flyttes til scripts/eval/:")
for rel in EVAL:
    path = ROOT / rel
    if DRY_RUN and not path.exists():
        # kan finnes etter at nye navn er gitt i steg 3
        renamed = [new for old, new in RENAME.items() if new == rel and (ROOT / old).exists()]
        if not renamed:
            continue
    elif not path.exists():
        continue
    print(f"   {rel}")
    if not DRY_RUN:
        text = path.read_text(encoding="utf-8")
        # ett nivå dypere: ROOT må gå ett hakk lenger opp
        text = re.sub(r"(Path\(__file__\)\.resolve\(\)\.parent\.parent)(?!\.parent)", r"\1.parent", text)
        # kjøreinstruksen i docstringen
        text = text.replace(f"python {rel}", f"python scripts/eval/{Path(rel).name}")
        target = ROOT / "scripts" / "eval" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_text(text, encoding="utf-8")
        path.unlink()

# 5. tomme mapper
print("\n5. Tomme mapper som fjernes:")
for rel in EMPTY_DIRS:
    folder = ROOT / rel
    if not folder.exists():
        continue
    rest = [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and p.name != "__init__.py" and str(p.relative_to(ROOT)).replace("\\", "/") not in DELETE]
    if rest:
        print(f"   {rel} beholdes, inneholder {[p.name for p in rest]}")
        continue
    print(f"   {rel}")
    if not DRY_RUN:
        shutil.rmtree(folder)

if DRY_RUN:
    print("\nIngenting er endret. Sett DRY_RUN = False og kjør igjen.")
else:
    print("\nFerdig. Sjekk at play_game.py og live_gesture.py fortsatt starter, og commit:")
    print('   git add -A scripts src old')
    print('   git commit -m "Rydd i scripts og src"')
    print("Slett deretter scripts/rydd_opp.py.")