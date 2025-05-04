package com.example.databasecopilot.settings;

import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.options.Configurable;
import com.intellij.openapi.options.ConfigurationException;
import com.intellij.openapi.util.NlsContexts;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;
import java.awt.*;

/**
 * Configurable component for Database Copilot settings.
 */
public class DatabaseCopilotConfigurable implements Configurable {
    
    private JPanel myMainPanel;
    private JTextField apiUrlTextField;
    private JLabel apiUrlLabel;
    
    private final DatabaseCopilotSettings settings;
    
    /**
     * Constructor.
     */
    public DatabaseCopilotConfigurable() {
        settings = ServiceManager.getService(DatabaseCopilotSettings.class);
    }
    
    @Override
    public @NlsContexts.ConfigurableName String getDisplayName() {
        return "Database Copilot";
    }
    
    @Override
    public @Nullable JComponent createComponent() {
        myMainPanel = new JPanel(new GridBagLayout());
        
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.fill = GridBagConstraints.HORIZONTAL;
        constraints.weightx = 0.2;
        constraints.gridx = 0;
        constraints.gridy = 0;
        constraints.insets = new Insets(5, 5, 5, 5);
        
        apiUrlLabel = new JLabel("API URL:");
        myMainPanel.add(apiUrlLabel, constraints);
        
        constraints.gridx = 1;
        constraints.weightx = 0.8;
        
        apiUrlTextField = new JTextField();
        apiUrlTextField.setText(settings.getApiUrl());
        myMainPanel.add(apiUrlTextField, constraints);
        
        constraints.gridx = 0;
        constraints.gridy = 1;
        constraints.gridwidth = 2;
        constraints.weighty = 1.0;
        constraints.fill = GridBagConstraints.BOTH;
        
        JPanel spacer = new JPanel();
        myMainPanel.add(spacer, constraints);
        
        return myMainPanel;
    }
    
    @Override
    public boolean isModified() {
        return !apiUrlTextField.getText().equals(settings.getApiUrl());
    }
    
    @Override
    public void apply() throws ConfigurationException {
        settings.setApiUrl(apiUrlTextField.getText());
    }
    
    @Override
    public void reset() {
        apiUrlTextField.setText(settings.getApiUrl());
    }
}
