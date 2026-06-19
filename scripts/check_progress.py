import csv
with open('vods/haven_output_v2/frame_states.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
if not rows:
    print('No rows')
else:
    last = rows[-1]
    print(f'Total frames: {len(rows)}')
    print(f'Last timestamp: {last["timestamp"]}')
    print(f'Last round: {last["round_number"]}')
    print(f'Last phase: {last["phase"]}')
    active = [r for r in rows if r["phase"] == "ACTIVE_ROUND"]
    print(f'Active round frames: {len(active)}')
    def_pos = sum(1 for r in active for i in range(5) if r.get(f"player_{i}_pos_x","").strip() not in ("","None"))
    atk_pos = sum(1 for r in active for i in range(5,10) if r.get(f"player_{i}_pos_x","").strip() not in ("","None"))
    total = len(active) * 5
    print(f'Defense position fill: {def_pos}/{total} = {def_pos/total*100:.0f}%')
    print(f'Attack position fill: {atk_pos}/{total} = {atk_pos/total*100:.0f}%')
    weapon_ok = sum(1 for r in active for i in range(10) if r.get(f"player_{i}_weapon","") not in ("unknown",""))
    print(f'Weapon fill: {weapon_ok}/{len(active)*10} = {weapon_ok/(len(active)*10)*100:.0f}%')
