-- create table emp(
--     id int comment '编号',
--     workno varchar(10) comment '工号'
-- )comment;
--
-- alter table emp add nickname varchar(20) comment'编号'
-- alter table emp DROP
-- Alter table
--
-- drop table
-- truncate table

insert into employee(id,workno,name,gender,age,idcard,entrydate) values(1,'1','Itcast','男',10,'12345678','2000-01-01')
insert into employee()
select + from employee;
insert into employee values(2,'2','张无忌','男',)
select * from emp where age between 15 and 20;
select * from emp where name like '__'
select * from emp where idcard like'%x';
select count(*) from emp;
select count(idcard) from emp;
select gender,count(*) from emp group by gender;
select gender,avg(age) from emp group by gender;

select * from emp order bt entrydate desc;
select * from emp order by age asc,entrydate desc;

select * from emp where gender = 'nan'
select e.name,e.age from emp e where e.age > 15 order by age asc;

create user 'heima'@'%' identified by '123456';
alter user 'heima'@'%' identified with mysql_native_password by '1234';

drop user 'itcast' @ 'localhost';


create table user(
    id int primary key auto_increment comment '主键'.
    name varchar(10) not null unique comment '姓名',
    age int check (age>0 && age <=120) comment '年龄',
    status char(1) default '1'comment '状态',
    gender char(1) comment '姓名'
)comment '用户表'

insert into user (name,age,status,gender)values('tom1',19,'1','男'),('tom2',25,'0','男');

--添加外键
alter table emp add constraint fk_emp_dept_id foreign key(dept_id) references dept(id) on update cascade on delete cascade;
alter table emp add constraint fk_enp_dept_id foreign key(dept_id) references dept(id) on update set null on delete set null;
select * from account where name = '张三';
update account set money = money - 1000 where name = '张三';
update account set money = money + 1000 where name = '李四';
set @@autocommit;
set @@autocommit = 0; --设置为手动提交

commit;
rollback; --设置事务回滚

select @@transaction_isolation;
set session transaction isolation;
set session transaction isolation level read committed;
show create table account;
show engines

create table mymemory(
    id int,
    name varchar(10)
) engine = Memory;
























