import mindsdb
import psycopg2

def connect_to_postgresql():
    host = 'localhost'
    port = 5432
    database = 'your_database'
    user = 'your_user'
    password = 'your_password'

    connection = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )

    return connection
