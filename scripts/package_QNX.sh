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

# Package the tool for QNX Neutrino RTOS
./gradlew assemble

# Create a distribution directory
mkdir -p dist/QNX

# Copy the build artifacts to the distribution directory
cp -r build/libs/* dist/QNX/

echo "Packaging for QNX Neutrino RTOS completed successfully."
