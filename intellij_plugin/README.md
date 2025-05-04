# Database Copilot IntelliJ Plugin

This directory contains the source code for the Database Copilot IntelliJ IDEA plugin.

## Overview

The Database Copilot IntelliJ plugin integrates the Database Copilot functionality directly into IntelliJ IDEA, making it easier for developers to work with Liquibase migrations and JPA entities.

## Features

- **Review Liquibase Migrations**: Review your Liquibase migrations against best practices and company guidelines.
- **Generate Liquibase Migrations**: Generate Liquibase migrations from natural language descriptions.
- **Q/A System**: Ask questions about JPA/Hibernate, ORM, Liquibase, and general database concepts.
- **Generate JPA Entities**: Generate JPA entity classes from Liquibase migrations.
- **Generate Tests**: Generate test classes for your JPA entities.

## Development Setup

### Prerequisites

- IntelliJ IDEA (Community or Ultimate) 2022.2 or later
- Java Development Kit (JDK) 17 or later
- Gradle 7.6 or later

### Building the Plugin

1. Clone the repository:
```bash
git clone https://github.com/yourusername/database-copilot.git
cd database-copilot/intellij_plugin
```

2. Build the plugin:
```bash
./gradlew buildPlugin
```

3. Install the plugin in IntelliJ IDEA:
   - Open IntelliJ IDEA
   - Go to Settings/Preferences > Plugins
   - Click on the gear icon and select "Install Plugin from Disk..."
   - Navigate to the build/distributions directory and select the zip file
   - Restart IntelliJ IDEA

## Plugin Structure

```
intellij_plugin/
├── build.gradle.kts        # Gradle build script
├── gradle.properties       # Gradle properties
├── settings.gradle.kts     # Gradle settings
├── src/                    # Source code
│   ├── main/
│   │   ├── java/           # Java source code
│   │   │   └── com/
│   │   │       └── example/
│   │   │           └── databasecopilot/
│   │   │               ├── actions/     # Action classes
│   │   │               ├── api/         # API client
│   │   │               ├── ui/          # UI components
│   │   │               └── util/        # Utility classes
│   │   └── resources/     # Resources
│   │       ├── META-INF/  # Plugin configuration
│   │       └── icons/     # Icons
│   └── test/              # Test code
└── README.md              # This file
```

## Usage

After installing the plugin, you can access the Database Copilot functionality from the "Database Copilot" menu in the main menu bar or from the context menu in the Project view.

## Configuration

The plugin needs to be configured to connect to the Database Copilot API server. You can configure the API server URL in the plugin settings:

1. Go to Settings/Preferences > Tools > Database Copilot
2. Enter the API server URL (default: http://localhost:8000)
3. Click "Apply" or "OK"

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
