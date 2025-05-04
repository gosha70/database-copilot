package com.example.databasecopilot.api;

import com.example.databasecopilot.settings.DatabaseCopilotSettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.intellij.openapi.components.ServiceManager;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.*;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * API client for the Database Copilot API.
 */
public class DatabaseCopilotApiClient {
    
    private static final MediaType MEDIA_TYPE_XML = MediaType.parse("application/xml; charset=utf-8");
    private static final MediaType MEDIA_TYPE_YAML = MediaType.parse("application/yaml; charset=utf-8");
    private static final MediaType MEDIA_TYPE_JSON = MediaType.parse("application/json; charset=utf-8");
    
    private final DatabaseCopilotApi api;
    
    /**
     * Constructor.
     */
    public DatabaseCopilotApiClient() {
        DatabaseCopilotSettings settings = ServiceManager.getService(DatabaseCopilotSettings.class);
        
        HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor();
        loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
        
        OkHttpClient client = new OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(60, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build();
        
        Gson gson = new GsonBuilder()
            .setLenient()
            .create();
        
        Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(settings.getApiUrl())
            .client(client)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build();
        
        api = retrofit.create(DatabaseCopilotApi.class);
    }
    
    /**
     * Review a Liquibase migration file.
     *
     * @param file The migration file.
     * @return The review result.
     * @throws IOException If an I/O error occurs.
     */
    public String reviewMigration(File file) throws IOException {
        RequestBody requestFile = RequestBody.create(
            file.getName().endsWith(".xml") ? MEDIA_TYPE_XML : MEDIA_TYPE_YAML,
            file
        );
        
        MultipartBody.Part filePart = MultipartBody.Part.createFormData(
            "file",
            file.getName(),
            requestFile
        );
        
        return api.reviewMigration(filePart).execute().body().review;
    }
    
    /**
     * Generate a Liquibase migration from a natural language description.
     *
     * @param description The description of the migration.
     * @param formatType The format type (xml or yaml).
     * @param author The author of the migration.
     * @return The generated migration.
     * @throws IOException If an I/O error occurs.
     */
    public String generateMigration(String description, String formatType, String author) throws IOException {
        MigrationGenerationRequest request = new MigrationGenerationRequest();
        request.description = description;
        request.format_type = formatType;
        request.author = author;
        
        return api.generateMigration(request).execute().body().migration;
    }
    
    /**
     * Answer a question about JPA/Hibernate or Liquibase.
     *
     * @param question The question to answer.
     * @param category The category of documentation to search in.
     * @return The answer to the question.
     * @throws IOException If an I/O error occurs.
     */
    public String answerQuestion(String question, String category) throws IOException {
        QuestionRequest request = new QuestionRequest();
        request.question = question;
        request.category = category;
        
        return api.answerQuestion(request).execute().body().answer;
    }
    
    /**
     * Generate a JPA entity from a Liquibase migration.
     *
     * @param file The migration file.
     * @param packageName The package name for the generated entity.
     * @param lombok Whether to use Lombok annotations.
     * @return The generated entity.
     * @throws IOException If an I/O error occurs.
     */
    public String generateEntity(File file, String packageName, boolean lombok) throws IOException {
        RequestBody requestFile = RequestBody.create(
            file.getName().endsWith(".xml") ? MEDIA_TYPE_XML : MEDIA_TYPE_YAML,
            file
        );
        
        MultipartBody.Part filePart = MultipartBody.Part.createFormData(
            "file",
            file.getName(),
            requestFile
        );
        
        RequestBody packageNamePart = RequestBody.create(
            MediaType.parse("text/plain"),
            packageName
        );
        
        RequestBody lombokPart = RequestBody.create(
            MediaType.parse("text/plain"),
            String.valueOf(lombok)
        );
        
        return api.generateEntity(filePart, packageNamePart, lombokPart).execute().body().entity;
    }
    
    /**
     * Generate test classes for a JPA entity.
     *
     * @param entityContent The content of the entity class.
     * @param packageName The package name for the generated test class.
     * @param testFramework The test framework to use.
     * @param includeRepositoryTests Whether to include repository tests.
     * @return The generated test class.
     * @throws IOException If an I/O error occurs.
     */
    public String generateTests(
        String entityContent,
        String packageName,
        String testFramework,
        boolean includeRepositoryTests
    ) throws IOException {
        TestGenerationRequest request = new TestGenerationRequest();
        request.entity_content = entityContent;
        request.package_name = packageName;
        request.test_framework = testFramework;
        request.include_repository_tests = includeRepositoryTests;
        
        return api.generateTests(request).execute().body().tests;
    }
    
    /**
     * API interface for the Database Copilot API.
     */
    private interface DatabaseCopilotApi {
        
        @Multipart
        @POST("/api/review-migration")
        Call<MigrationReviewResponse> reviewMigration(@Part MultipartBody.Part file);
        
        @POST("/api/generate-migration")
        Call<MigrationGenerationResponse> generateMigration(@Body MigrationGenerationRequest request);
        
        @POST("/api/answer-question")
        Call<QuestionResponse> answerQuestion(@Body QuestionRequest request);
        
        @Multipart
        @POST("/api/generate-entity")
        Call<EntityGenerationResponse> generateEntity(
            @Part MultipartBody.Part file,
            @Part("package_name") RequestBody packageName,
            @Part("lombok") RequestBody lombok
        );
        
        @POST("/api/generate-tests")
        Call<TestGenerationResponse> generateTests(@Body TestGenerationRequest request);
    }
    
    /**
     * Request model for generating a migration.
     */
    private static class MigrationGenerationRequest {
        public String description;
        public String format_type;
        public String author;
    }
    
    /**
     * Response model for generating a migration.
     */
    private static class MigrationGenerationResponse {
        public String migration;
    }
    
    /**
     * Response model for reviewing a migration.
     */
    private static class MigrationReviewResponse {
        public String review;
    }
    
    /**
     * Request model for answering a question.
     */
    private static class QuestionRequest {
        public String question;
        public String category;
    }
    
    /**
     * Response model for answering a question.
     */
    private static class QuestionResponse {
        public String answer;
    }
    
    /**
     * Response model for generating an entity.
     */
    private static class EntityGenerationResponse {
        public String entity;
    }
    
    /**
     * Request model for generating tests.
     */
    private static class TestGenerationRequest {
        public String entity_content;
        public String package_name;
        public String test_framework;
        public boolean include_repository_tests;
    }
    
    /**
     * Response model for generating tests.
     */
    private static class TestGenerationResponse {
        public String tests;
    }
}
