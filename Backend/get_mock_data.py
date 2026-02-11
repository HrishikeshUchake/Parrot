import requests
import json
import sqlite3

URL = 'https://dummyjson.com/posts'

response = requests.get(URL)
data = response.json()
print(json.dumps(data, indent=2))

connection = sqlite3.connect("store_social_data.db")
cursor = connection.cursor()

command1 = """CREATE TABLE IF NOT EXISTS
posts(
id INTEGER PRIMARY KEY, 
title TEXT, 
body TEXT, 
tags TEXT,
reactions TEXT,
views INTEGER,
userId INTEGER
);
"""

cursor.execute(command1)
connection.commit()

posts = data["posts"]

insert_sql = """
INSERT OR REPLACE INTO posts
(id, title, body, tags, reactions, views, userId)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

for p in posts:
    cursor.execute(
        insert_sql,
        (
            p["id"],
            p.get("title"),
            p.get("body"),
            json.dumps(p.get("tags", [])),
            json.dumps(p.get("reactions", {})),
            p.get("views"),
            p.get("userId"),
        )
    )

# 3) Save + close
connection.commit()
connection.close()

print(f"Inserted {len(posts)} posts into store_social_data.db")
