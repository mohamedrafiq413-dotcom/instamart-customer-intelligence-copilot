
-- Instamart Customer Intelligence & Support Copilot
-- Part A: Seven required SQL analyses
-- Database: data/instamart.db

-- Q1: Customers whose completed orders over INR 500
-- have a combined qualifying spend greater than INR 8,000.
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE LOWER(status) = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Completed revenue by city and product category.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c
    ON c.customer_id = o.customer_id
JOIN products AS p
    ON p.product_id = o.product_id
WHERE LOWER(o.status) = 'completed'
GROUP BY
    c.city,
    p.category
ORDER BY
    c.city,
    p.category;


-- Q3: Products priced above the average price
-- across all products.
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: Customers with no completed orders.
-- Required method: LEFT JOIN.
SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
   AND LOWER(o.status) = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Customers whose completed-order spending is above
-- the average spending of customers with completed orders.
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE LOWER(status) = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT
        AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(a.avg_spend, 2) AS average_customer_spend
FROM customer_spend AS cs
CROSS JOIN average_spend AS a
WHERE cs.total_spend > a.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6: Rank customers by completed-order spending within
-- each city using RANK, DENSE_RANK and ROW_NUMBER.
SELECT
    c.city,
    o.customer_id,
    ROUND(SUM(o.order_amount), 2) AS completed_spend,
    RANK() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC, o.customer_id
    ) AS row_number_in_city
FROM orders AS o
JOIN customers AS c
    ON c.customer_id = o.customer_id
WHERE LOWER(o.status) = 'completed'
GROUP BY
    c.city,
    o.customer_id
ORDER BY
    c.city,
    spend_rank,
    o.customer_id;


-- Q7: Daily completed revenue and cumulative running revenue.
SELECT
    SUBSTR(order_ts, 1, 10) AS order_date,
    ROUND(SUM(order_amount), 2) AS daily_revenue_inr,
    ROUND(
        SUM(SUM(order_amount)) OVER (
            ORDER BY SUBSTR(order_ts, 1, 10)
            ROWS BETWEEN UNBOUNDED PRECEDING
                     AND CURRENT ROW
        ),
        2
    ) AS running_revenue_inr
FROM orders
WHERE LOWER(status) = 'completed'
GROUP BY SUBSTR(order_ts, 1, 10)
ORDER BY order_date;
