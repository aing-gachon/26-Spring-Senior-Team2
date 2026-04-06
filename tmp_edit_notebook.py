import json

file_path = "c:\\Aing_s\\26-Spring-Senior-Team2\\notebooks\\BicDRL_Colab.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "for step in range(steps_per_episode):" in line:
                source[i] = "    from tqdm import tqdm\n"
                source.insert(i+1, "    pbar = tqdm(range(steps_per_episode), desc=f'Episode {episode+1}/{num_episodes}')\n")
                source.insert(i+2, "    for step in pbar:\n")
                break
        cell["source"] = source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
