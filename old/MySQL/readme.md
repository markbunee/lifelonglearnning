# MySQL

1.资料

[DataGrip操作MySQL完全指南：从入门到精通DataGrip操作MySQL完全指南：从入门到精通 1. Dat - 掘金](https://juejin.cn/post/7480826943065358370)

2.笔记

start the mysql

net start mysql80

net stop mysql80

客户端链接

mysql [-h 127.0.0.1] [-p 3306] -u rooy -p

![image-20250617210318505](./pic/image-20250617210318505.png)

##### ![image-20250617210337845](./pic/image-20250617210337845.png)

utf8支持三字节 所以建议使用utf8mb4支持四个字节

show databases；

select database()；

create database if exists test；

drop 

![image-20250617211649050](./pic/image-20250617211649050.png)

![image-20250617212949031](./pic/image-20250617212949031.png)

age TYNIY UNSIGNED

score double(4,1)

![image-20250617213233701](./pic/image-20250617213233701.png)

char(10) 使用一个字符的话也会使用空格填充 性能好

varchar(10)性能较差

![image-20250617214743383](./pic/image-20250617214743383.png)

![image-20250617215611202](./pic/image-20250617215611202.png)

![image-20250617220333632](./pic/image-20250617220333632.png)

![image-20250617235222654](./pic/image-20250617235222654.png)

![image-20250618091131232](./pic/image-20250618091131232.png)

![image-20250618091653031](./pic/image-20250618091653031.png)

![image-20250618092630674](./pic/image-20250618092630674.png)

分页查询每个数据库都会有不同的实现

![image-20250618094507996](./pic/image-20250618094507996.png)

![image-20250618095627230](./pic/image-20250618095627230.png)

![image-20250618100748578](./pic/image-20250618100748578.png)

常用函数：

![image-20250618101009880](./pic/image-20250618101009880.png)

![image-20250618170714672](./pic/image-20250618170714672.png)

![image-20250618170749650](./pic/image-20250618170749650.png)

![image-20250618170806740](./pic/image-20250618170806740.png)

![image-20250618170834697](./pic/image-20250618170834697.png)

![image-20250618171747823](./pic/image-20250618171747823.png)

![image-20250618173223026](./pic/image-20250618173223026.png)

![image-20250619171149844](./pic/image-20250619171149844.png)

![image-20250619171254548](./pic/image-20250619171254548.png)

![image-20250619171401970](./pic/image-20250619171401970.png)

![image-20250619171652569](./pic/image-20250619171652569.png)

![image-20250619172245340](./pic/image-20250619172245340.png)

![image-20250619173203941](./pic/image-20250619173203941.png)

万一出现故障，多个独立事务被执行，然后转账后可能会直接转账失败

![image-20250620084634902](./pic/image-20250620084634902.png)

![image-20250620091626570](./pic/image-20250620091626570.png)

![image-20250620091753457](./pic/image-20250620091753457.png)

进阶：

![image-20250620092955628](./pic/image-20250620092955628.png)

InnoDB默认存储引擎

![image-20250620093624714](./pic/image-20250620093624714.png)

![image-20250620093912450](./pic/image-20250620093912450.png)

![image-20250620094139550](./pic/image-20250620094139550.png)

![image-20250620094242968](./pic/image-20250620094242968.png)

![image-20250620094314770](./pic/image-20250620094314770.png)

![image-20250620094400145](./pic/image-20250620094400145.png)



初始化：mysqld --initialize

安装命令：mysqld install

启动服务命令：net start mysql

mysql -uroot -p

alter user ‘root’@‘localhost’ identified by ‘root’;

net stop mysql

mysqld -remove

![image-20251222211143965](./pic/image-20251222211143965.png)







