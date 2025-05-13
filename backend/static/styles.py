"""
Custom styles for Database Copilot.
"""

# Custom CSS for the Streamlit app
CUSTOM_CSS = """
<style>
/* Custom styling for expanders */
.streamlit-expanderHeader {
    font-weight: bold;
    color: #4CAF50;
}

/* Make expanders stand out more */
.streamlit-expander {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 5px;
    margin-bottom: 1rem;
}

/* Header styling */
.app-header {
    display: flex;
    align-items: left;
    margin-bottom: 1rem;
}

.app-logo {
    height: 150px;
    margin-right: 5px;
}

.app-title {
    font-size: 2rem;
    font-weight: bold;
    color: var(--primary-color);
}

/* Theme toggle switch */
.theme-toggle {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
}

.theme-toggle-label {
    margin-right: 10px;
    font-weight: bold;
}

/* Color picker container */
.color-pickers {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 1rem;
}

.color-picker-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* Expander customization */
.streamlit-expander {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 5px;
    margin-bottom: 1rem;
}

.streamlit-expander .streamlit-expanderHeader {
    font-weight: bold;
    color: var(--primary-color);
}

/* Button styling */
.stButton > button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: var(--accent-color);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

/* Code block styling */
.stCodeBlock {
    border-radius: 5px;
    border: 1px solid rgba(128, 128, 128, 0.2);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: rgba(128, 128, 128, 0.1);
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary-color);
    color: white;
}

/* File uploader styling */
.stFileUploader > div > button {
    background-color: var(--secondary-color);
}

.stFileUploader > div > button:hover {
    background-color: var(--accent-color);
}

/* Text area styling */
.stTextArea textarea {
    border-radius: 4px;
    border-color: rgba(128, 128, 128, 0.2);
}

.stTextArea textarea:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 1px var(--primary-color);
}

/* Sidebar styling */
.css-1d391kg {
    background-color: var(--background-color);
}

/* Custom classes for color customization */
.primary-text {
    color: var(--primary-color);
}

.secondary-text {
    color: var(--secondary-color);
}

.accent-text {
    color: var(--accent-color);
}

.custom-label {
    font-weight: bold;
    margin-bottom: 0.5rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .app-logo {
        margin-bottom: 10px;
    }
    
    .color-pickers {
        flex-direction: column;
    }
}
</style>
"""

# HTML for the theme toggle switch
THEME_TOGGLE_HTML = """
<div class="theme-toggle">
    <span class="theme-toggle-label">Theme:</span>
    <label class="switch">
        <input type="checkbox" id="theme-toggle-checkbox" onchange="toggleTheme()">
        <span class="slider round"></span>
    </label>
    <span id="theme-label">Light</span>
</div>

<script>
function toggleTheme() {
    const checkbox = document.getElementById('theme-toggle-checkbox');
    const themeLabel = document.getElementById('theme-label');
    
    if (checkbox.checked) {
        // Dark mode
        document.documentElement.setAttribute('data-theme', 'dark');
        themeLabel.textContent = 'Dark';
        localStorage.setItem('theme', 'dark');
    } else {
        // Light mode
        document.documentElement.removeAttribute('data-theme');
        themeLabel.textContent = 'Light';
        localStorage.setItem('theme', 'light');
    }
}

// Initialize theme based on localStorage
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme');
    const checkbox = document.getElementById('theme-toggle-checkbox');
    const themeLabel = document.getElementById('theme-label');
    
    if (savedTheme === 'dark') {
        checkbox.checked = true;
        document.documentElement.setAttribute('data-theme', 'dark');
        themeLabel.textContent = 'Dark';
    } else {
        checkbox.checked = false;
        document.documentElement.removeAttribute('data-theme');
        themeLabel.textContent = 'Light';
    }
});
</script>
"""

# HTML for color pickers
COLOR_PICKERS_HTML = """
<div class="color-pickers">
    <div class="color-picker-item">
        <label class="custom-label">Primary Color</label>
        <input type="color" id="primary-color-picker" value="#4CAF50" onchange="updateColor('primary-color', this.value)">
    </div>
    <div class="color-picker-item">
        <label class="custom-label">Secondary Color</label>
        <input type="color" id="secondary-color-picker" value="#2196F3" onchange="updateColor('secondary-color', this.value)">
    </div>
    <div class="color-picker-item">
        <label class="custom-label">Accent Color</label>
        <input type="color" id="accent-color-picker" value="#FF5722" onchange="updateColor('accent-color', this.value)">
    </div>
    <div class="color-picker-item">
        <label class="custom-label">Text Color</label>
        <input type="color" id="text-color-picker" value="#333333" onchange="updateColor('text-color', this.value)">
    </div>
</div>

<script>
function updateColor(property, value) {
    document.documentElement.style.setProperty(`--${property}`, value);
    localStorage.setItem(`color-${property}`, value);
}

// Initialize colors based on localStorage
document.addEventListener('DOMContentLoaded', function() {
    const colorProperties = ['primary-color', 'secondary-color', 'accent-color', 'text-color'];
    
    colorProperties.forEach(property => {
        const savedColor = localStorage.getItem(`color-${property}`);
        if (savedColor) {
            document.documentElement.style.setProperty(`--${property}`, savedColor);
            document.getElementById(`${property}-picker`).value = savedColor;
        }
    });
});
</script>
"""
