drop database if exists bc;
create database bc;
use bc;


create table users(
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(225),
    email VARCHAR(50), 
    password VARCHAR(50));