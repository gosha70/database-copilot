package com.example.databasecopilot.settings;

import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import com.intellij.util.xmlb.XmlSerializerUtil;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Persistent settings for the Database Copilot plugin.
 */
@State(
    name = "com.example.databasecopilot.settings.DatabaseCopilotSettings",
    storages = {@Storage("DatabaseCopilotSettings.xml")}
)
public class DatabaseCopilotSettings implements PersistentStateComponent<DatabaseCopilotSettings> {
    
    private String apiUrl = "http://localhost:8000";
    
    /**
     * Get the API URL.
     *
     * @return The API URL.
     */
    public String getApiUrl() {
        return apiUrl;
    }
    
    /**
     * Set the API URL.
     *
     * @param apiUrl The API URL.
     */
    public void setApiUrl(String apiUrl) {
        this.apiUrl = apiUrl;
    }
    
    @Nullable
    @Override
    public DatabaseCopilotSettings getState() {
        return this;
    }
    
    @Override
    public void loadState(@NotNull DatabaseCopilotSettings state) {
        XmlSerializerUtil.copyBean(state, this);
    }
}
