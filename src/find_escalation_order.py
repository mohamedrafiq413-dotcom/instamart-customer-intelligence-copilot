
import sqlite3

connection = sqlite3.connect("data/instamart.db")

query = """
SELECT order_id, status, order_amount, delivery_minutes
FROM orders
WHERE status = ?
  AND order_amount > ?
  AND delivery_minutes > ?
LIMIT 10
"""

orders = connection.execute(
    query,
    ("completed", 1000, 45)
).fetchall()

if orders:
    print("Matching orders for the human-review test:")
    for order in orders:
        print(order)
else:
    print("No matching orders found.")

connection.close()
