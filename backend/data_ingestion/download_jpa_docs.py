"""
Script to download JPA/Hibernate documentation.
"""
import os
import logging
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.config import DOC_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# URLs for JPA/Hibernate documentation
JPA_DOCS_URLS = [
    # JPA Specification
    "https://jakarta.ee/specifications/persistence/3.0/jakarta-persistence-spec-3.0.html",
    # Hibernate ORM Documentation
    "https://docs.jboss.org/hibernate/orm/6.2/userguide/html_single/Hibernate_User_Guide.html",
    # Hibernate Annotations Reference
    "https://docs.jboss.org/hibernate/orm/6.2/javadocs/",
    # Spring Data JPA Documentation
    "https://docs.spring.io/spring-data/jpa/docs/current/reference/html/",
]

def download_jpa_docs():
    """
    Download JPA/Hibernate documentation.
    """
    logger.info("Downloading JPA/Hibernate documentation")
    
    # Create the JPA docs directory if it doesn't exist
    jpa_docs_dir = DOC_CATEGORIES["jpa_docs"]
    os.makedirs(jpa_docs_dir, exist_ok=True)
    
    # Download documentation from each URL
    for url in JPA_DOCS_URLS:
        try:
            logger.info(f"Downloading documentation from {url}")
            
            # Get the page content
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract the title
            title = soup.title.string if soup.title else Path(url).name
            title = title.replace(" ", "_").replace("/", "_").replace(":", "")
            
            # Extract the main content
            main_content = soup.find("body")
            if main_content:
                content = main_content.get_text(separator="\n", strip=True)
            else:
                content = soup.get_text(separator="\n", strip=True)
            
            # Save the content to a file
            file_path = os.path.join(jpa_docs_dir, f"{title}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Saved documentation to {file_path}")
        
        except Exception as e:
            logger.error(f"Error downloading documentation from {url}: {e}")

def download_additional_jpa_resources():
    """
    Download additional JPA/Hibernate resources.
    """
    logger.info("Creating additional JPA/Hibernate resources")
    
    # Create the JPA docs directory if it doesn't exist
    jpa_docs_dir = DOC_CATEGORIES["jpa_docs"]
    os.makedirs(jpa_docs_dir, exist_ok=True)
    
    # Common JPA annotations
    jpa_annotations = """
    # Common JPA Annotations
    
    ## Entity Annotations
    
    ### @Entity
    Specifies that the class is an entity. This annotation is applied to the entity class.
    
    ```java
    @Entity
    public class Employee {
        // ...
    }
    ```
    
    ### @Table
    Specifies the primary table for the annotated entity.
    
    ```java
    @Entity
    @Table(name = "employees")
    public class Employee {
        // ...
    }
    ```
    
    ### @Id
    Specifies the primary key of an entity.
    
    ```java
    @Entity
    public class Employee {
        @Id
        private Long id;
        // ...
    }
    ```
    
    ### @GeneratedValue
    Specifies how the primary key should be generated.
    
    ```java
    @Entity
    public class Employee {
        @Id
        @GeneratedValue(strategy = GenerationType.IDENTITY)
        private Long id;
        // ...
    }
    ```
    
    ## Field Annotations
    
    ### @Column
    Specifies the mapped column for a persistent property or field.
    
    ```java
    @Entity
    public class Employee {
        @Id
        @GeneratedValue(strategy = GenerationType.IDENTITY)
        private Long id;
        
        @Column(name = "first_name", length = 50, nullable = false)
        private String firstName;
        
        // ...
    }
    ```
    
    ### @Temporal
    Specifies the type of a temporal property.
    
    ```java
    @Entity
    public class Employee {
        // ...
        
        @Temporal(TemporalType.DATE)
        private Date birthDate;
        
        @Temporal(TemporalType.TIMESTAMP)
        private Date hireDate;
        
        // ...
    }
    ```
    
    ### @Enumerated
    Specifies that a persistent property or field should be persisted as an enumerated type.
    
    ```java
    @Entity
    public class Employee {
        // ...
        
        @Enumerated(EnumType.STRING)
        private EmployeeStatus status;
        
        // ...
    }
    ```
    
    ### @Lob
    Specifies that a persistent property or field should be persisted as a large object to a database-supported large object type.
    
    ```java
    @Entity
    public class Employee {
        // ...
        
        @Lob
        private String resume;
        
        @Lob
        private byte[] profilePicture;
        
        // ...
    }
    ```
    
    ## Relationship Annotations
    
    ### @OneToOne
    Specifies a single-valued association to another entity that has one-to-one multiplicity.
    
    ```java
    @Entity
    public class Employee {
        // ...
        
        @OneToOne(cascade = CascadeType.ALL)
        @JoinColumn(name = "address_id", referencedColumnName = "id")
        private Address address;
        
        // ...
    }
    ```
    
    ### @OneToMany
    Specifies a many-valued association with one-to-many multiplicity.
    
    ```java
    @Entity
    public class Employee {
        // ...
        
        @OneToMany(mappedBy = "employee", cascade = CascadeType.ALL, orphanRemoval = true)
        private List<Task> tasks = new ArrayList<>();
        
        // ...
    }
    ```
    
    ### @ManyToOne
    Specifies a single-valued association to another entity that has many-to-one multiplicity.
    
    ```java
    @Entity
    public class Task {
        // ...
        
        @ManyToOne
        @JoinColumn(name = "employee_id")
        private Employee employee;
        
        // ...
    }
    ```
    
    ### @ManyToMany
    Specifies a many-valued association with many-to-many multiplicity.
    
    ```java
    @Entity
    public class Employee {
        // ...
        
        @ManyToMany(cascade = {CascadeType.PERSIST, CascadeType.MERGE})
        @JoinTable(
            name = "employee_project",
            joinColumns = @JoinColumn(name = "employee_id"),
            inverseJoinColumns = @JoinColumn(name = "project_id")
        )
        private List<Project> projects = new ArrayList<>();
        
        // ...
    }
    ```
    
    ## Query Annotations
    
    ### @NamedQuery
    Specifies a static, named query in the Java Persistence query language.
    
    ```java
    @Entity
    @NamedQuery(
        name = "Employee.findByLastName",
        query = "SELECT e FROM Employee e WHERE e.lastName = :lastName"
    )
    public class Employee {
        // ...
    }
    ```
    
    ### @NamedQueries
    Specifies multiple named queries.
    
    ```java
    @Entity
    @NamedQueries({
        @NamedQuery(
            name = "Employee.findByLastName",
            query = "SELECT e FROM Employee e WHERE e.lastName = :lastName"
        ),
        @NamedQuery(
            name = "Employee.findByDepartment",
            query = "SELECT e FROM Employee e WHERE e.department = :department"
        )
    })
    public class Employee {
        // ...
    }
    ```
    """
    
    # JPA best practices
    jpa_best_practices = """
    # JPA/Hibernate Best Practices
    
    ## Entity Design
    
    1. **Use a business key**: Always define a business key (natural key) in addition to the primary key.
    
    2. **Prefer primitive types**: Use primitive types instead of wrapper classes for better performance.
    
    3. **Use appropriate fetch types**: Use FetchType.LAZY for most associations to avoid the N+1 query problem.
    
    4. **Bidirectional associations**: For bidirectional associations, always maintain both sides of the relationship.
    
    5. **Avoid deep inheritance hierarchies**: Deep inheritance hierarchies can lead to complex queries and performance issues.
    
    6. **Use appropriate cascade types**: Be careful with CascadeType.ALL, as it might lead to unintended consequences.
    
    7. **Use @Version for optimistic locking**: Add a version field to enable optimistic locking.
    
    ## Query Optimization
    
    1. **Use named queries**: Named queries are parsed and validated at startup, which can catch errors early.
    
    2. **Avoid cartesian products**: Be careful with joins to avoid cartesian products.
    
    3. **Use pagination**: Always use pagination for large result sets.
    
    4. **Use projections**: Use projections to select only the needed columns.
    
    5. **Use query hints**: Use query hints to optimize query execution.
    
    6. **Avoid N+1 queries**: Use join fetch to avoid the N+1 query problem.
    
    ## Performance Considerations
    
    1. **Use batch processing**: Use batch processing for bulk operations.
    
    2. **Use stateless sessions**: Use stateless sessions for bulk operations.
    
    3. **Use second-level cache**: Use second-level cache for frequently accessed, rarely changed data.
    
    4. **Use query cache**: Use query cache for frequently executed queries.
    
    5. **Use native queries for complex operations**: Use native queries for complex operations that are difficult to express in JPQL.
    
    6. **Monitor and tune the database**: Monitor and tune the database for optimal performance.
    
    ## Common Pitfalls
    
    1. **LazyInitializationException**: This occurs when trying to access a lazy-loaded association outside of a transaction.
    
    2. **N+1 query problem**: This occurs when a query returns N entities, and then N additional queries are executed to fetch a related entity.
    
    3. **Cartesian product**: This occurs when joining multiple entities without proper join conditions.
    
    4. **Detached entities**: This occurs when trying to update a detached entity without merging it first.
    
    5. **Flush mode**: The default flush mode is AUTO, which might lead to unexpected behavior.
    
    6. **Entity state transitions**: Be aware of entity state transitions (transient, persistent, detached, removed).
    
    7. **Cascading operations**: Be careful with cascading operations, as they might lead to unintended consequences.
    """
    
    # JPA and Liquibase integration
    jpa_liquibase_integration = """
    # JPA and Liquibase Integration
    
    ## Overview
    
    Liquibase and JPA/Hibernate are complementary tools for database management:
    
    - **Liquibase** manages database schema changes through version-controlled changesets.
    - **JPA/Hibernate** provides an object-relational mapping (ORM) framework for Java applications.
    
    When used together, they provide a complete solution for database management:
    
    1. **Liquibase** handles the database schema evolution.
    2. **JPA/Hibernate** handles the object-relational mapping.
    
    ## Best Practices for Integration
    
    1. **Use Liquibase for all schema changes**: Don't rely on Hibernate's schema generation (hbm2ddl) in production.
    
    2. **Keep entity classes and database schema in sync**: Ensure that your entity classes match the database schema defined by Liquibase.
    
    3. **Use Liquibase contexts**: Use Liquibase contexts to separate different types of changes (e.g., schema changes, data changes, test data).
    
    4. **Use Liquibase for database initialization**: Use Liquibase to initialize the database schema and load initial data.
    
    5. **Use Liquibase for database migration testing**: Use Liquibase to test database migrations before applying them to production.
    
    ## Common Workflows
    
    1. **Creating a new entity**:
       - Create the entity class with JPA annotations.
       - Create a Liquibase changeset to create the corresponding table.
       - Run the Liquibase migration to create the table.
       - Use the entity class in your application.
    
    2. **Modifying an existing entity**:
       - Update the entity class with the new fields or annotations.
       - Create a Liquibase changeset to modify the corresponding table.
       - Run the Liquibase migration to modify the table.
       - Use the updated entity class in your application.
    
    3. **Removing an entity**:
       - Create a Liquibase changeset to drop the corresponding table.
       - Run the Liquibase migration to drop the table.
       - Remove the entity class from your application.
    
    ## Example: Entity Class and Liquibase Changeset
    
    ### Entity Class
    
    ```java
    @Entity
    @Table(name = "employees")
    public class Employee {
        @Id
        @GeneratedValue(strategy = GenerationType.IDENTITY)
        private Long id;
        
        @Column(name = "first_name", length = 50, nullable = false)
        private String firstName;
        
        @Column(name = "last_name", length = 50, nullable = false)
        private String lastName;
        
        @Column(name = "email", length = 100, unique = true)
        private String email;
        
        @Temporal(TemporalType.DATE)
        @Column(name = "birth_date")
        private Date birthDate;
        
        @Temporal(TemporalType.TIMESTAMP)
        @Column(name = "hire_date", nullable = false)
        private Date hireDate;
        
        @Enumerated(EnumType.STRING)
        @Column(name = "status", nullable = false)
        private EmployeeStatus status;
        
        // Getters and setters
    }
    ```
    
    ### Liquibase Changeset (XML)
    
    ```xml
    <changeSet id="1" author="liquibase">
        <createTable tableName="employees">
            <column name="id" type="bigint" autoIncrement="true">
                <constraints primaryKey="true" nullable="false"/>
            </column>
            <column name="first_name" type="varchar(50)">
                <constraints nullable="false"/>
            </column>
            <column name="last_name" type="varchar(50)">
                <constraints nullable="false"/>
            </column>
            <column name="email" type="varchar(100)">
                <constraints unique="true"/>
            </column>
            <column name="birth_date" type="date"/>
            <column name="hire_date" type="timestamp">
                <constraints nullable="false"/>
            </column>
            <column name="status" type="varchar(20)">
                <constraints nullable="false"/>
            </column>
        </createTable>
    </changeSet>
    ```
    
    ### Liquibase Changeset (YAML)
    
    ```yaml
    databaseChangeLog:
      - changeSet:
          id: 1
          author: liquibase
          changes:
            - createTable:
                tableName: employees
                columns:
                  - column:
                      name: id
                      type: bigint
                      autoIncrement: true
                      constraints:
                        primaryKey: true
                        nullable: false
                  - column:
                      name: first_name
                      type: varchar(50)
                      constraints:
                        nullable: false
                  - column:
                      name: last_name
                      type: varchar(50)
                      constraints:
                        nullable: false
                  - column:
                      name: email
                      type: varchar(100)
                      constraints:
                        unique: true
                  - column:
                      name: birth_date
                      type: date
                  - column:
                      name: hire_date
                      type: timestamp
                      constraints:
                        nullable: false
                  - column:
                      name: status
                      type: varchar(20)
                      constraints:
                        nullable: false
    ```
    """
    
    # Save the additional resources
    file_path = os.path.join(jpa_docs_dir, "jpa_annotations.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(jpa_annotations)
    logger.info(f"Saved JPA annotations to {file_path}")
    
    file_path = os.path.join(jpa_docs_dir, "jpa_best_practices.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(jpa_best_practices)
    logger.info(f"Saved JPA best practices to {file_path}")
    
    file_path = os.path.join(jpa_docs_dir, "jpa_liquibase_integration.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(jpa_liquibase_integration)
    logger.info(f"Saved JPA-Liquibase integration guide to {file_path}")

if __name__ == "__main__":
    download_jpa_docs()
    download_additional_jpa_resources()
