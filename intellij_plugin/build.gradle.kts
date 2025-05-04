import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("java")
    id("org.jetbrains.kotlin.jvm") version "1.8.21"
    id("org.jetbrains.intellij") version "1.13.3"
}

group = "com.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

// Configure Gradle IntelliJ Plugin
// Read more: https://plugins.jetbrains.com/docs/intellij/tools-gradle-intellij-plugin.html
intellij {
    version.set("2022.2.5")
    type.set("IC") // Target IDE Platform: IntelliJ IDEA Community Edition

    plugins.set(listOf(
        "com.intellij.java",
        "org.jetbrains.plugins.gradle"
    ))
}

dependencies {
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.10.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.6.4")
}

tasks {
    // Set the JVM compatibility versions
    withType<JavaCompile> {
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }
    withType<KotlinCompile> {
        kotlinOptions.jvmTarget = "17"
    }

    patchPluginXml {
        sinceBuild.set("222")
        untilBuild.set("232.*")
        
        pluginDescription.set("""
            Database Copilot IntelliJ Plugin
            
            A plugin that integrates Database Copilot functionality directly into IntelliJ IDEA,
            making it easier for developers to work with Liquibase migrations and JPA entities.
            
            Features:
            - Review Liquibase Migrations: Review your Liquibase migrations against best practices and company guidelines.
            - Generate Liquibase Migrations: Generate Liquibase migrations from natural language descriptions.
            - Q/A System: Ask questions about JPA/Hibernate, ORM, Liquibase, and general database concepts.
            - Generate JPA Entities: Generate JPA entity classes from Liquibase migrations.
            - Generate Tests: Generate test classes for your JPA entities.
        """.trimIndent())
        
        changeNotes.set("""
            Initial release of the Database Copilot IntelliJ Plugin.
            
            - Integration with Database Copilot API
            - Review Liquibase migrations
            - Generate Liquibase migrations
            - Q/A system for JPA/Hibernate and Liquibase
            - Generate JPA entities from Liquibase migrations
            - Generate tests for JPA entities
        """.trimIndent())
    }

    signPlugin {
        certificateChain.set(System.getenv("CERTIFICATE_CHAIN"))
        privateKey.set(System.getenv("PRIVATE_KEY"))
        password.set(System.getenv("PRIVATE_KEY_PASSWORD"))
    }

    publishPlugin {
        token.set(System.getenv("PUBLISH_TOKEN"))
    }
}
