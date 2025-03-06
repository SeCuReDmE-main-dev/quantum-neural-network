#!/bin/bash

# Ensure gradle is installed
if ! command -v gradle &> /dev/null
then
    echo "gradle could not be found, please install gradle."
    exit
fi

# Set JAVA_HOME if not already set
if [ -z "$JAVA_HOME" ]; then
    echo "JAVA_HOME is not set. Please set JAVA_HOME to the correct JDK installation path."
    exit
fi

# Run gradle wrapper to ensure it is set up correctly
gradle wrapper

# Build the project
./gradlew build

# Run tests
./gradlew test

echo "Testing for openEuler OS completed successfully."
