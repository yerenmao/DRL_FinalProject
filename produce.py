from lxml import etree as ET
import os

# Source: the original .net.xml
SRC = "/Users/len/Desktop/DRL/final/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4.net.xml"

# Destination directory and filename
out_dir = "nets/4x4-Lucas"
os.makedirs(out_dir, exist_ok=True)         # ← ensure the folder exists
DST = os.path.join(out_dir, "4x4_fixed5s.net.xml")

tree = ET.parse(SRC)
root = tree.getroot()

for tl in root.findall(".//tlLogic"):
    for phase in tl.findall("phase"):
        phase.set("duration", "5")            # set every phase to 5 seconds

tree.write(DST, pretty_print=True,
           encoding="UTF-8", xml_declaration=True)
print(f"✓  已寫入 {DST}")
