# app.py

import streamlit as st
import requests
import pandas as pd
import re
from urllib.parse import urlparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from bs4 import BeautifulSoup

# --- Page Configuration ---
st.set_page_config(page_title="AI Contact Researcher", layout="wide", initial_sidebar_state="expanded")

# --- Session State Initialization ---
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "processing_results" not in st.session_state:
    st.session_state.processing_results = []
if "all_models" not in st.session_state:
    st.session_state.all_models = None

# --- Core Functions ---

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_openrouter_models(api_key):
    """Fetches and categorizes models from the OpenRouter API."""
    if not api_key:
        return None
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        models_data = response.json().get("data", [])

        categorized_models = {
            "Top Rated": [],
            "Web Search": [],
            "Free": [],
            "All Models": []
        }

        for model in sorted(models_data, key=lambda x: x.get('id')):
            model_id = model['id']
            is_free = model.get('pricing', {}).get('prompt', "0.0") == "0.0" and model.get('pricing', {}).get('completion', "0.0") == "0.0"
            has_search = "search" in model_id.lower() or "web" in model_id.lower() or "sonar" in model_id.lower()
            is_top = "claude-3-opus" in model_id or "gpt-4o" in model_id or "gemini-1.5-pro" in model_id

            categorized_models["All Models"].append(model_id)
            if is_free:
                categorized_models["Free"].append(model_id)
            if has_search:
                categorized_models["Web Search"].append(model_id)
            if is_top:
                 categorized_models["Top Rated"].append(model_id)

        st.session_state.all_models = categorized_models
        return categorized_models
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to fetch models: {e}")
        return None

def create_final_prompt(company_name, company_website, country):
    """Creates the specific, user-defined prompt for the AI."""
    domain = urlparse(company_website).netloc.replace("www.", "")
    
    prompt = f"""Act as expert web researcher. search on given website and every source like directory, social media publicly available online. Please find the names and roles of the CEO OR Founder OR Managing Director OR Managing Partner for the company {company_name} with the website "{company_website}" based in {country}. With the person you found, Provide their personal LinkedIn URLs and work email addresses using the domain @{domain}. Find top one contact only. Also include the general company contact email. Present the information in a table with name, role, full linkedin url, work email, general email, source with real link, confidence."""
    
    return prompt

def query_openrouter(api_key, model, prompt):
    """Sends a query to the OpenRouter API and returns the response."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        return f"### API Error\nAn error occurred while contacting the API: {e}"
    except (KeyError, IndexError) as e:
        return f"### API Error\nCould not parse the API response: {e}\nResponse: {response.text}"

def scrape_website_title(url):
    """Scrapes just the title of a website to confirm it's accessible."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string if soup.title else "No title found."
    except requests.RequestException as e:
        return f"Could not access site: {e}"

def process_company(company_info, api_key, model):
    """Processes a single company's information."""
    company_name = company_info.get("company", "N/A")
    website = company_info.get("website", "N/A")
    country = company_info.get("country", "N/A")

    # Ensure website has a scheme
    if not re.match(r'http[s]?://', website):
        website = 'https://' + website

    # Scrape title for a quick check
    site_status = scrape_website_title(website)
    
    # Generate prompt and query AI
    ai_prompt = create_final_prompt(company_name, website, country)
    ai_result = query_openrouter(api_key, model, ai_prompt)

    return {
        "Company": company_name,
        "Website": website,
        "Country": country,
        "Site Status": site_status,
        "AI Result": ai_result
    }

def to_excel(results_df):
    """Converts a DataFrame to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Results')
    processed_data = output.getvalue()
    return processed_data

# --- UI Layout ---

st.title("🤖 AI Contact Researcher")
st.caption("A tool to find executive contacts using AI-driven web research.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Use Streamlit Secrets for the API key in deployment
    api_key = st.text_input("Enter your OpenRouter API Key", type="password", value=st.secrets.get("OPENROUTER_API_KEY", ""))

    models = get_openrouter_models(api_key)
    
    if models:
        # Determine the initial index for the selectbox
        if st.session_state.selected_model and st.session_state.selected_model in models["All Models"]:
            # Find the index of the previously selected model
            initial_index = models["All Models"].index(st.session_state.selected_model)
        else:
            initial_index = 0 # Default to the first model
        
        # Create a unified list with category headers
        display_options = []
        option_to_model_map = []

        for category, model_list in models.items():
            if model_list:
                display_options.append(f"--- {category} ---")
                option_to_model_map.append(None) # Header is not selectable
                for model_id in model_list:
                    display_options.append(model_id)
                    option_to_model_map.append(model_id)

        # We need to find the index in the new display list
        if st.session_state.selected_model:
            try:
                final_index = display_options.index(st.session_state.selected_model)
            except ValueError:
                final_index = 0
        else:
            final_index = 0

        selected_option = st.selectbox(
            "Select AI Model", 
            options=display_options,
            index=final_index,
            format_func=lambda x: "──────────" if x.startswith("---") else x,
            help="Choose a model. Models with 'search' or 'sonar' can browse the web."
        )

        # Update session state only if a valid model is chosen
        if selected_option and not selected_option.startswith("---"):
            st.session_state.selected_model = selected_option
        
    else:
        st.warning("Please enter a valid API key to load models.")
        st.session_state.selected_model = None


# --- Main Content Area ---
tab1, tab2 = st.tabs([" одиночний пошук ", " Пакетна обробка "])

# --- Single Search Tab ---
with tab1:
    st.subheader("Find a Single Contact")
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            company_name_single = st.text_input("Company Name", placeholder="e.g., Google")
        with col2:
            website_single = st.text_input("Company Website", placeholder="e.g., google.com")
        country_single = st.text_input("Country", placeholder="e.g., USA")

        submit_single = st.form_submit_button("🚀 Research Contact", use_container_width=True)

    if submit_single:
        if not company_name_single or not website_single:
            st.error("Please provide both Company Name and Website.")
        elif not st.session_state.selected_model:
            st.error("Please select an AI model in the sidebar.")
        else:
            with st.spinner(f"AI is researching {company_name_single}..."):
                company_info = {"company": company_name_single, "website": website_single, "country": country_single}
                result = process_company(company_info, api_key, st.session_state.selected_model)
                st.session_state.processing_results = [result]

# --- Batch Processing Tab ---
with tab2:
    st.subheader("Find Contacts in Bulk")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    st.markdown("Your CSV should have columns named `company`, `website`, and `country`.")

    submit_batch = st.button("🚀 Start Batch Research", use_container_width=True)

    if submit_batch and uploaded_file:
        if not st.session_state.selected_model:
            st.error("Please select an AI model in the sidebar.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if not all(col in df.columns for col in ["company", "website", "country"]):
                    st.error("CSV file is missing required columns: `company`, `website`, `country`.")
                else:
                    companies_to_process = df.to_dict('records')
                    st.session_state.processing_results = []
                    results = []
                    
                    progress_bar = st.progress(0, text="Starting batch processing...")
                    
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_company = {executor.submit(process_company, info, api_key, st.session_state.selected_model): info for info in companies_to_process}
                        
                        for i, future in enumerate(as_completed(future_to_company)):
                            result = future.result()
                            results.append(result)
                            progress = (i + 1) / len(companies_to_process)
                            progress_bar.progress(progress, text=f"Processed {result['Company']}...")
                    
                    st.session_state.processing_results = results
                    progress_bar.progress(1.0, text="Batch processing complete!")

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

# --- Display Results ---
if st.session_state.processing_results:
    st.divider()
    st.subheader("🔍 Research Results")
    
    results_df = pd.DataFrame(st.session_state.processing_results)
    
    # Download button for Excel
    excel_data = to_excel(results_df)
    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_data,
        file_name=f"contact_research_results_{time.strftime('%Y%m%d-%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    for result in st.session_state.processing_results:
        with st.expander(f"**{result['Company']}** - {result['Website']}"):
            st.markdown(result["AI Result"])
            st.caption(f"Site Status: {result['Site Status']}")
