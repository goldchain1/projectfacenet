import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face"
)
mycursor = mydb.cursor()


#def Query(start_date, end_date):
    #sql = ''
    #if start_date is None or end_date is None:
        #sql = "SELECT * FROM persons ORDER BY id DESC LIMIT 13"
    #else:
        #start_date = start_date + " 00:00:00"
        #end_date = end_date + " 23:59:59"
        #sql = "SELECT * FROM persons WHERE datetime BETWEEN '%s' AND '%s' ORDER BY id DESC LIMIT 13" % \
              #(start_date, end_date)
    #mycursor.execute(sql)
    #return mycursor.fetchall()

def Query():
    mycursor.execute("select * from persons ORDER BY id DESC LIMIT 13")
    return mycursor.fetchall()

def Querydetails(start_date, end_date):
    sql = ''
    if start_date is None or end_date is None:
        sql = "SELECT * FROM persons ORDER BY id"
    else:
        start_date = start_date + " 00:00:00"
        end_date = end_date + " 23:59:59"
        sql = "SELECT * FROM persons WHERE datetime BETWEEN '%s' AND '%s' ORDER BY id" % \
              (start_date, end_date)
    mycursor.execute(sql)
    return mycursor.fetchall()


def Insert(name, accuracy, filename):
    sql = "INSERT INTO persons (name, accuracy, filename) VALUES ('%s', '%s', '%s')" % (name, accuracy, filename)
    mycursor.execute(sql)
    mydb.commit()


def Countrows():
    mycursor.execute("SELECT COUNT(*) FROM persons;")
    return mycursor.fetchall()

#def Countrows():
    #mycursor.execute("SELECT name FROM persons;")
    #a = set(mycursor.fetchall())
    #return [(len(a)-1)]

print(Countrows())
