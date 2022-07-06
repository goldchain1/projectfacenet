import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
    database="testproject"
)
mycursor = mydb.cursor()
mycursor.execute("CREATE TABLE persons"
                 "("
                 "id INT AUTO_INCREMENT PRIMARY KEY,"
                 "name VARCHAR(255),"
                 "datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                 "accuracy VARCHAR(255),"
                 "filename VARCHAR(255)"
                 ")")
#mycursor.execute("CREATE TABLE persons (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), time VARCHAR(255))")

#mydb.commit()


