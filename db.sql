-- Drop database if exists and create new
drop database if exists aerial;
create database aerial;
use aerial;

-- Create users table with proper indexing
create table users(
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(225),
    email VARCHAR(50) UNIQUE,  -- Added UNIQUE constraint
    password VARCHAR(50)
);

-- Add index on email for faster lookups and foreign key
CREATE INDEX idx_user_email ON users(email);

-- Create analysis_history table
CREATE TABLE analysis_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    xai_path VARCHAR(500),
    predicted_class VARCHAR(100),
    confidence VARCHAR(50),
    top1_class VARCHAR(100),
    top2_class VARCHAR(100),
    top3_class VARCHAR(100),
    top1_conf FLOAT,
    top2_conf FLOAT,
    top3_conf FLOAT,
    severity VARCHAR(50),
    prognosis TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_email) REFERENCES users(email) ON DELETE CASCADE
);