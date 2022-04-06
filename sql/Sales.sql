.open --new '/Users/SawmonAbo/Desktop/ucf_classes/eco_4443/sql/databases/Sales.db'


/*pid, price, year */
CREATE TABLE Prices
(
pid     Integer ,
price   Integer ,
year    Integer ,
PRIMARY KEY (pid)
)
;

.tables
.import --csv --skip 1 /Users/SawmonAbo/Desktop/ucf_classes/eco_4443/data/prices.csv Prices


/*tid, cbd_dist, x_coord, y_coord */
CREATE TABLE Locations
(
tid         Integer ,
cbd_dist    Integer ,
x_coord     Integer ,
y_coord     Integer ,
PRIMARY KEY (tid)
)
;

.tables
.import --csv --skip 1 /Users/SawmonAbo/Desktop/ucf_classes/eco_4443/data/locations.csv Locations


/* pid, tid, home_size, parcel_size, beds, age, pool */
CREATE TABLE Characteristics
(
pid         Integer ,
tid         Integer ,
home_size   Integer ,
parcel_size Integer ,
beds        Integer ,
age         Integer ,
pool        Integer ,
PRIMARY KEY (pid)
FOREIGN KEY (pid) REFERENCES Prices (pid) ,
FOREIGN KEY (tid) REFERENCES Locations (tid)
)
;

.tables
.import --csv --skip 1 /Users/SawmonAbo/Desktop/ucf_classes/eco_4443/data/characteristics.csv Characteristics

 /* The dot commands below affect how the output is displayed */
.mode column
.headers on 

.schema

/* Displays current settings for various parameters */
.show

/* Select mode for the output table */
.mode


.headers on
.mode csv
.output '/Users/SawmonAbo/Desktop/ucf_classes/eco_4443/sql/intermediate/sales.csv'


SELECT * 
FROM Characteristics
JOIN Prices using(pid)
JOIN Locations using(tid)
;

















/*

 
SELECT *

FROM Characteristics
JOIN Prices ON Characteristics.pid = Prices.pid
JOIN Locations ON Characteristics.tid = Locations.tid
GROUP BY Prices.pid
;











select 
  t1.cust_id, t1.col1, t1.col2, 
  t2.col1_table2, t2.col2_table2, 
  t3.col1_table3, t3.col2_table3
from
 Characteristics t1 
inner join
 Prices t2 on t1.cust_id = t2.cust_id
join Locations t3 on t1.cust_id = t3.cust_id










select * from (
    select Characteristics.* 
           , n.newspapername
           , null as magazinename
           , ns.enddate
          , n.publishedby 
    from Characteristics
        join newspapersubscription ns 
            on ns.customerid = customer.customerid
        join newspaper n
            on  n.newspapername = ns.newspapername 
    union all
    select Characteristics.* 
           , null as newspapername
           , m.magazinename
           , ms.enddate
           , m.publishedby 
    from Characteristics
        join magazinesubscription  ms 
            on ms.Characteristics = Characteristics.customerid
        join magazine m
            on  m.magazinename = ms.magazinename 
            )
order by customerid, newspapername nulls last, magazinename ;



SELECT customer.customerid, customer.customername, customer.customeraddress, newspapername, magazinename, enddate, publishedby 
FROM customer
INNER JOIN
(select  customerid, newspapername, null Magazinename, enddate, n.publishedby 
 from newspapersubscription ns, newspaper n 
 where publishedby in(select publishedby 
                    from newspaper 
                    where ns.newspapername = n.NewspaperName)
UNION
select  customerid, null newspapername, Magazinename, enddate, m.publishedby 
from magazinesubscription ms, magazine m 
 where publishedby in(select publishedby 
                    from magazine 
                     where ms.Magazinename = m.MagazineName))
on customer.customerid = customerid
ORDER BY customer.customerid;
*/