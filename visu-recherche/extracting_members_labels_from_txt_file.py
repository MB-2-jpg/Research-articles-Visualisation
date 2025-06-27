import re

def extract_members_and_topics(filepath):
    members = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Regex pour capturer chaque bloc "Member X : Name" suivi de "Topics : ..."
    pattern = re.compile(
        r"Member\s+(\d+)\s*:\s*(.+?)\nTopics\s*:\s*(.*?)\n(?=Member\s+\d+\s*:|$)", 
        re.DOTALL | re.IGNORECASE
    )
    
    for match in pattern.finditer(content):
        member_num = int(match.group(1))
        member_name = match.group(2).strip()
        topics_str = match.group(3).strip().rstrip(",;")
        
        # Nettoyer les topics : séparer par virgule ou point-virgule, enlever espaces inutiles
        topics = re.split(r"[;,]", topics_str)
        topics = [t.strip() for t in topics if t.strip()]
        
        members.append({
            "member_num": member_num,
            "member_name": member_name,
            "topics": topics
        })
    return members

# Exemple d'utilisation :
filepath = "web_scraping/s2a_members.txt"  # Remplace par le chemin de ton fichier
members = extract_members_and_topics(filepath)

# Affiche les 5 premiers pour vérification
for m in members[:5]:
    print(f"Member {m['member_num']} : {m['member_name']}")
    print("Topics :", ", ".join(m['topics']))
    print()

