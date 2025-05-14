
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
    