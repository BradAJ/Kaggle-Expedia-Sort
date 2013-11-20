CREATE TABLE PropFactors7M (     
       prop_id BIGINT PRIMARY KEY,
       prop_cnt BIGINT,
       train_price_avg DOUBLE,
       click_cnt BIGINT,
       booking_cnt BIGINT,
       promotion_cnt BIGINT
       );

INSERT INTO PropFactors7M 
SELECT prop_id,  
       Count(prop_id) AS prop_cnt, 
       Avg(price_usd) AS train_price_avg, 
       Sum(click_bool) AS click_cnt, 
       Sum(booking_bool) AS booking_cnt, 
       Sum(promotion_flag) AS promotion_cnt 
FROM UserSearchKey 
WHERE row_num <= 7000013 
GROUP BY prop_id;


