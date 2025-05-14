
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
    