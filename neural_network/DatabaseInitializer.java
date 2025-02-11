package neural_network;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.logging.Logger;
import java.util.logging.Level;

public class DatabaseInitializer {
    private static final Logger LOGGER = Logger.getLogger(DatabaseInitializer.class.getName());
    private static final String DEFAULT_DB_PATH = "brain_structure.accdb";
    private static final String SQL_SETUP_PATH = "../brain_capsules/setup_database.sql";

    public static void initializeDatabase() {
        try {
            // Create Access database manager
            AccessDatabaseManager dbManager = new AccessDatabaseManager(DEFAULT_DB_PATH);
            
            // Read SQL setup file
            String setupSql = new String(Files.readAllBytes(Paths.get(SQL_SETUP_PATH)));
            
            // Split SQL into individual statements and execute
            for (String statement : setupSql.split(";")) {
                if (!statement.trim().isEmpty()) {
                    dbManager.executeQuery(statement.trim());
                }
            }
            
            LOGGER.info("Database initialization completed successfully");
            dbManager.close();
            
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error reading SQL setup file", e);
            throw new RuntimeException("Failed to initialize database", e);
        }
    }

    public static void main(String[] args) {
        initializeDatabase();
    }
}