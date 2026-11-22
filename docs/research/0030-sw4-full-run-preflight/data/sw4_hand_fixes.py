"""Hand fixes applied to the regenerated SW4 blocks. Usage: hand_fixes.py characters.json"""
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
by = {c["name"]: c for c in data["characters"]}

# Whole-block rewrites: the model regressed these to the wrong kind of thing.
by["darth_vader"]["description"] = (
    "A tall figure in glossy black armour and a full black helmet. The helmet is a smooth "
    "rounded dome with a flared rim at the back, a full-face mask with a downward-sloping "
    "triangular grille over the mouth and large dark angled lenses. Rigid black chest plating "
    "carries a small control panel with coloured buttons and horizontal light bars. Rounded "
    "black shoulder guards, a long black cape, black gloves and boots. No skin is visible."
)
by["stormtrooper"]["description"] = (
    "A masked soldier in glossy white plastic armour. The helmet is smooth and rounded, with "
    "a black tapered 'T'-shaped visor, a short snout-like vent with horizontal black slats "
    "and black circular ear caps. Segmented white chest plate, rigid shoulder guards and "
    "limb plates over a black bodysuit, with black showing at the joints. No skin or face is visible."
)
by["c_3po"]["description"] = (
    "A mechanical robot with a polished, reflective gold-plated exterior. The head is a "
    "smooth rounded dome with two large round glowing yellow optical lenses and a fixed "
    "metal faceplate. The torso is segmented, with exposed wiring at the midsection and "
    "visible mechanical joints at the shoulders, elbows and knees. The surface shows minor "
    "scuffs and scratches."
)

# Targeted fixes.
jawa = by["chief_jawa"]
jawa["description"] = jawa["description"].replace(
    "The creature has a hunched posture, with long, thin, gloved hands and a small, rounded frame",
    "It has long, thin, gloved hands and a small, rounded frame")
han = by["han_solo"]
han["description"] = han["description"].replace(
    "and a light, natural stubble across his chin and upper lip.", "and a clean-shaven chin and upper lip.")

# The model gave seven pilots the same invented scar/mole/ear quirk. Drop invented marks.
STRIP = [
    " There is a small, faint vertical scar on his left cheekbone.",
    " There is a small, faint horizontal scar on his left cheekbone and a slight asymmetry in his ears.",
    " There is a small, faint horizontal scar on his left cheekbone.",
    " He has a small, faint horizontal scar on his left cheekbone and slight asymmetry in his ear placement.",
    " There is a small, faint mole on his left cheekbone.",
    " A small, faint horizontal scar is visible on his left cheekbone.",
    " He has a small, faint scar near his right temple and subtle asymmetry in his cheekbones.",
    ", and he has a small, faint vertical scar on his left cheekbone",
    " There is a small, faint horizontal scar on his left cheekbone and slight asymmetry in the set of his ears.",
]
for name in ("red_two", "red_three", "red_four", "red_six", "gold_leader", "gold_two",
             "gold_five", "red_two_wedge_antilles"):
    d = by[name]["description"]
    for s in STRIP:
        d = d.replace(s, "")
    by[name]["description"] = d.strip()

for name in ("red_two", "red_three", "red_four", "red_six", "gold_leader", "gold_two",
             "gold_five", "red_two_wedge_antilles"):
    d = by[name]["description"]
    for marker in ("scar", "mole", "asymmetry in"):
        assert marker not in d, (name, d)

json.dump(data, open(path, "w"), indent=2)
print("ok")
