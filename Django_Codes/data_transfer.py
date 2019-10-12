import sqlite3
import datetime

conn = sqlite3.connect('city.db')

c=conn.cursor()
##
##c.execute("""CREATE TABLE city (
##                    City,
##                    Location ,
##                    Time,
##                    No_plate
##                    )""")
'''
c.execute("INSERT INTO city VALUES ('Chennai','SVCE',datetime.datetime.now(),)")
c.commit()
c.execute("SELECT * FROM employees WHERE last='Schafer'")
'''

conn.commit()
print(datetime.datetime.now())
conn.close()
