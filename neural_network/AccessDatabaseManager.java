package neural_network;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.logging.Logger;
import java.util.logging.Level;

public class AccessDatabaseManager {
    private static final Logger LOGGER = Logger.getLogger(AccessDatabaseManager.class.getName());
    private static final String DRIVER = "net.ucanaccess.jdbc.UcanaccessDriver";
    private Connection connection;
    
    public AccessDatabaseManager(String databasePath) {
        try {
            Class.forName(DRIVER);
            String url = "jdbc:ucanaccess://" + databasePath;
            connection = DriverManager.getConnection(url);
            LOGGER.info("Successfully connected to database: " + databasePath);
        } catch (ClassNotFoundException | SQLException e) {
            LOGGER.log(Level.SEVERE, "Error connecting to database", e);
            throw new RuntimeException("Failed to initialize database connection", e);
        }
    }

    public void executeQuery(String sql) {
        try (Statement statement = connection.createStatement()) {
            statement.execute(sql);
            LOGGER.info("Successfully executed query");
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error executing query: " + sql, e);
            throw new RuntimeException("Failed to execute query", e);
        }
    }

    public void importFromSqlServer(String sqlServerConnectionString, String tableName) {
        try {
            // Example implementation for importing data from SQL Server
            String importQuery = String.format(
                "INSERT INTO %s SELECT * FROM LINKEDSERVER.[Database].[Schema].%s",
                tableName, tableName
            );
            executeQuery(importQuery);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error importing data from SQL Server", e);
            throw new RuntimeException("Failed to import data", e);
        }
    }

    public void close() {
        if (connection != null) {
            try {
                connection.close();
                LOGGER.info("Database connection closed successfully");
            } catch (SQLException e) {
                LOGGER.log(Level.SEVERE, "Error closing database connection", e);
            }
        }
    }

    // Method to initialize static tables from SQL file
    public void initializeStaticTables() {
        try {
            // Execute each table creation query from setup_database.sql
            executeQuery(readSetupQueries());
            LOGGER.info("Successfully initialized static tables");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error initializing static tables", e);
            throw new RuntimeException("Failed to initialize static tables", e);
        }
    }

    private String readSetupQueries() {
        // Implementation to read queries from setup_database.sql
        // This would typically read from the SQL file we saw earlier
        StringBuilder queries = new StringBuilder();
        // Add table creation queries...
        return queries.toString();
    }
}