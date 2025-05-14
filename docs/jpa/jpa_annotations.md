
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
    