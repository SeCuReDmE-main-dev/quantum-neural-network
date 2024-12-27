import unittest
import mindsdb
import psycopg2
from neural_network.mindsdb_postgresql_connection import connect_to_postgresql

class TestMindsDBPostgreSQLConnection(unittest.TestCase):

    def test_connection(self):
        connection = connect_to_postgresql()
        self.assertIsNotNone(connection)
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        self.assertEqual(result[0], 1)
        cursor.close()
        connection.close()

if __name__ == '__main__':
    unittest.main()
