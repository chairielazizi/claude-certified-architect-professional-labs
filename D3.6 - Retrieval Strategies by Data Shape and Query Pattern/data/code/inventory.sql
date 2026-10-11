-- Available-to-promise stock per SKU and warehouse
CREATE VIEW available_stock AS
SELECT sku, warehouse_id, on_hand - reserved AS available
FROM inventory
WHERE on_hand > reserved;

-- Reorder report: SKUs whose available stock fell below the reorder point
CREATE VIEW reorder_report AS
SELECT s.sku, s.warehouse_id, s.available, p.reorder_point
FROM available_stock s
JOIN reorder_policy p ON p.sku = s.sku
WHERE s.available < p.reorder_point;
