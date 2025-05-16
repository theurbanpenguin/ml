-- Create the app database if it doesn't exist
CREATE DATABASE IF NOT EXISTS app;
USE app;

-- Create the products table
CREATE TABLE IF NOT EXISTS products (
    productid SMALLINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

-- Create the orders table with foreign key reference to products
CREATE TABLE IF NOT EXISTS orders (
    orderid INT AUTO_INCREMENT PRIMARY KEY,
    productid SMALLINT NOT NULL,
    quantity SMALLINT NOT NULL,
    FOREIGN KEY (productid) REFERENCES products(productid)
);

-- Insert five fruit products
INSERT INTO products (name) VALUES
    ('Apple'),
    ('Banana'),
    ('Orange'),
    ('Strawberry'),
    ('Kiwi');

-- The orders table starts empty
-- You can add orders later using:
-- INSERT INTO orders (productid, quantity) VALUES (1, 10);