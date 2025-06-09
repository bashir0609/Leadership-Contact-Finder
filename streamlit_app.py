# app.py

import streamlit as st
import requests
import pandas as pd
import re
from urllib.parse import urlparse, urljoin
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from bs4 import BeautifulSoup # For web scraping website titles
from email_validator import validate_email, EmailNotValidError # For email verification
import markdown # For parsing AI's markdown table
from bs4 import BeautifulSoup as MarkdownSoup # For parsing AI's markdown output

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

        # Prioritize specific known good models
        preferred_order = [
            "openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-1.5-pro-latest",
            "perplexity/llama-3-sonar-large-32k-online", "perplexity/llama-3-sonar-small-32k-online"
        ]
        
        other_models = []

        for model in sorted(models_data, key=lambda x: x.get('id')):
            model_id = model['id']
            is_free = model.get('pricing', {}).get('prompt', "0.0") == "0.0" and \
                      model.get('pricing', {}).get('completion', "0.0") == "0.0"
            has_search = "search" in model_id.lower() or "web" in model_id.lower() or \
                         "sonar" in model_id.lower() or "online" in model_id.lower() or \
                         "browser" in model_id.lower()
            is_top = "claude-3-opus" in model_id or "gpt-4o" in model_id or \
                     "gemini-1.5-pro" in model_id or "gemini-flash" in model_id

            is_preferred = model_id in preferred_order

            categorized_models["All Models"].append(model_id)
            if is_free:
                categorized_models["Free"].append(model_id)
            if has_search:
                # Add to Web Search, ensuring preferred models come first if they also have search
                if is_preferred and model_id not in categorized_models["Web Search"]:
                    categorized_models["Web Search"].insert(0, model_id)
                elif not is_preferred and model_id not in categorized_models["Web Search"]:
                    categorized_models["Web Search"].append(model_id)
            if is_top:
                 # Add to Top Rated, ensuring preferred models come first
                if is_preferred and model_id not in categorized_models["Top Rated"]:
                    categorized_models["Top Rated"].insert(0, model_id)
                elif not is_preferred and model_id not in categorized_models["Top Rated"]:
                    categorized_models["Top Rated"].append(model_id)
        
        # Ensure preferred models are at the top of their categories and remove duplicates
        for category in ["Top Rated", "Web Search", "Free"]:
            unique_models = []
            seen = set()
            # Add preferred models first, if they belong to this category
            for pref_model in preferred_order:
                if pref_model in categorized_models[category] and pref_model not in seen:
                    unique_models.append(pref_model)
                    seen.add(pref_model)
            # Add other models
            for model_id in categorized_models[category]:
                if model_id not in seen:
                    unique_models.append(model_id)
                    seen.add(model_id)
            categorized_models[category] = unique_models

        st.session_state.all_models = categorized_models
        return categorized_models
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to fetch models: {e}")
        return None

def create_final_prompt(company_name, company_website, country):
    """Creates the specific, user-defined prompt for the AI, emphasizing accuracy."""
    domain = urlparse(company_website).netloc.replace("www.", "")

    prompt = f"""Act as an expert and meticulous web researcher. Your primary goal is to find ACCURATE and VERIFIABLE contact information.
You MUST use your web browsing/search capabilities thoroughly if available.

For the company: {company_name}
Website: {company_website}
Country: {country}

1.  **Objective:** Find the single most relevant executive:
    *   First, try to find the CEO.
    *   If no CEO is clearly identifiable, then look for a Founder.
    *   If neither CEO nor Founder is clear, then look for a Managing Director OR Managing Partner.
    *   Identify ONLY ONE such person.

2.  **For this ONE executive, provide the following:**
    *   **Name:** Full name of the person.
    *   **Role:** Their exact title (e.g., CEO, Founder, Managing Partner).
    *   **LinkedIn URL:** Their personal, direct LinkedIn profile URL. (Verify it's a valid profile).
    *   **Work Email:** Their professional email address. Try to find it directly. If constructing it (e.g., firstname.lastname@{domain}, f.lastname@{domain}), clearly state "Constructed Email" and the pattern used. Prioritize directly found emails.

3.  **Additionally, find:**
    *   **General Email:** The general company contact email (e.g., info@{domain}, contact@{domain}).

4.  **Source & Confidence:**
    *   **Source (URL):** For EACH piece of information (Name, Role, LinkedIn, Work Email, General Email), provide the *exact* URL of the web page where this information was found. If multiple sources, list the primary one. Make sure these are clickable, full URLs.
    *   **Confidence:** A confidence level (High, Medium, Low) for the accuracy of the identified contact person and their details *as a whole*.

5.  **Output Format:** Present ALL information clearly in a Markdown table with the following columns precisely:
    `Name | Role | LinkedIn URL | Work Email | General Email | Source (URL) | Confidence`

6.  **CRITICAL INSTRUCTIONS:**
    *   **Verification is Key:** Do not guess. Cross-reference information if possible.
    *   **If Not Found:** If you cannot confidently identify an executive matching the roles from reliable sources (e.g., the company's official website 'About Us' or 'Leadership' page, official LinkedIn company page), then state "No specific executive found matching criteria" in the 'Name' and 'Role' fields. In this case, still attempt to find the 'General Email' if possible.
    *   **Accuracy Over Completion:** It is better to leave a field blank or state "Not Found" than to provide incorrect or fabricated information. For example, if a LinkedIn URL is not found, put "Not Found" in that cell. If a work email is not found, put "Not Found".
    *   **Website Check:** If the provided website {company_website} seems to be a placeholder, inaccessible, or not the official company site, please note this in your response (perhaps before the table).

Please proceed with the research.
"""
    return prompt

def query_openrouter(api_key, model, prompt):
    """Sends a query to the OpenRouter API and returns the response."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app-url.com", # Optional: Replace with your app's URL
                "X-Title": "AI Contact Researcher" # Optional: Replace with your app's name
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=180 # Increased timeout for models that browse
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.Timeout:
        return f"### API Error\nThe request to OpenRouter timed out after 180 seconds. The selected model might be too slow or unresponsive. Try a different model."
    except requests.RequestException as e:
        return f"### API Error\nAn error occurred while contacting the API: {e}"
    except (KeyError, IndexError, TypeError) as e: # Added TypeError
        return f"### API Error\nCould not parse the API response: {e}\nResponse: {response.text if 'response' in locals() else 'No response object'}"


def scrape_website_title(url):
    """Scrapes just the title of a website to confirm it's accessible."""
    try:
        # Add a common user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string.strip() if soup.title and soup.title.string else "No title found (site accessible)."
    except requests.exceptions.MissingSchema:
        return f"Invalid URL (missing http/https): {url}"
    except requests.exceptions.TooManyRedirects:
        return f"Could not access site: Too many redirects for {url}"
    except requests.exceptions.SSLError:
        return f"Could not access site: SSL Error for {url}. Try with http:// or check certificate."
    except requests.RequestException as e:
        return f"Could not access site: {e}"

# --- Email Verification and Extraction ---

def verify_individual_email(email_address):
    """Verifies a single email address for syntax and deliverability (MX record)."""
    if not email_address or not isinstance(email_address, str) or "@" not in email_address or "." not in email_address.split('@')[-1]:
        return "Invalid (format)"
    try:
        validation = validate_email(email_address, check_deliverability=True, timeout=10)
        return "Valid"
    except EmailNotValidError as e:
        # Clean up the error message slightly for better display
        error_message = str(e)
        if "The domain name" in error_message and "does not exist" in error_message:
            return "Invalid (domain does not exist)"
        if "The domain name" in error_message and "does not accept email" in error_message:
            return "Invalid (domain does not accept email)"
        if "Please use a different email address" in error_message: # Generic from library
            return "Invalid (syntax or deliverability)"
        return f"Invalid ({error_message[:50]})" # Keep it concise
    except Exception as e: # Catch other potential errors like DNS timeout during check_deliverability
        if "timed out" in str(e).lower():
            return "Verification timeout"
        return f"Verification error"

def parse_markdown_table_for_emails(md_text):
    """Parses a Markdown table from text and extracts emails from columns containing 'email' in header."""
    emails_in_table = []
    if not md_text or not isinstance(md_text, str):
        return emails_in_table

    try:
        # Convert Markdown to HTML, specifically enabling the 'tables' extension
        html = markdown.markdown(md_text, extensions=['markdown.extensions.tables'])
        soup_md = MarkdownSoup(html, 'html.parser')
        table = soup_md.find('table')
        if not table:
            return emails_in_table

        headers = [th.text.strip().lower() for th in table.find_all('th')]
        email_column_indices = []
        for i, header in enumerate(headers):
            # More specific check for work/general email columns
            if "work email" in header or "general email" in header or (header == "email" and "work" not in headers and "general" not in headers) :
                email_column_indices.append(i)
        
        if not email_column_indices:
            return emails_in_table

        rows = table.find_all('tr')
        email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        
        for row_idx, row in enumerate(rows):
            is_header_row = all(child.name == 'th' for child in row.find_all(recursive=False))
            if row_idx == 0 and (table.find('thead') or is_header_row): # Skip header row
                continue

            cols = row.find_all(['td', 'th'])
            
            for email_col_idx in email_column_indices:
                if email_col_idx < len(cols):
                    cell_text = cols[email_col_idx].text.strip()
                    # Extract all valid email patterns from the cell text
                    found_in_cell = re.findall(email_regex, cell_text)
                    # Filter out common placeholder/invalid emails found in tables
                    valid_emails_from_cell = [e for e in found_in_cell if e.lower() not in ["not found", "n/a", "none", "example@example.com"]]
                    emails_in_table.extend(valid_emails_from_cell)
        
        return list(set(e.lower() for e in emails_in_table if e)) # Unique, non-empty, lowercase emails
    except Exception:
        return []


def extract_and_verify_emails(text_content):
    """Extracts emails from text (prioritizing table parsing) and verifies them."""
    if not text_content or not isinstance(text_content, str) or "API Error" in text_content :
        return []

    # Try to parse table first
    potential_emails = parse_markdown_table_for_emails(text_content)
    
    # Fallback: if table parsing fails or yields no emails, use regex on the whole text
    if not potential_emails:
        email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        all_emails_in_text = re.findall(email_regex, text_content)
        # Filter out common placeholder/invalid emails
        potential_emails = list(set(e.lower() for e in all_emails_in_text if e.lower() not in ["not found", "n/a", "none", "example@example.com"] and len(e.split('@')[0]) > 0 and len(e.split('@')[-1].split('.')) > 1 and len(e.split('@')[-1].split('.')[-1]) >=2 ))


    if not potential_emails:
        return []

    verified_emails_data = []
    
    # Use ThreadPoolExecutor for concurrent verification if multiple emails
    num_emails_to_verify = len(potential_emails)
    if num_emails_to_verify == 0:
        return []
    
    # Limit workers to avoid overwhelming local resources or hitting rate limits on DNS
    max_workers_email = min(5, num_emails_to_verify) 

    with ThreadPoolExecutor(max_workers=max_workers_email) as executor:
        future_to_email = {executor.submit(verify_individual_email, email): email for email in potential_emails}
        for future in as_completed(future_to_email):
            email = future_to_email[future]
            try:
                status = future.result()
                verified_emails_data.append({"email": email, "status": status})
            except Exception: # Should be caught by verify_individual_email, but as a fallback
                verified_emails_data.append({"email": email, "status": "Verification exec error"})
    
    return sorted(verified_emails_data, key=lambda x: x['email'])

def process_company(company_info, api_key, model):
    """Processes a single company's information, including email verification."""
    company_name = company_info.get("company", "N/A")
    website = company_info.get("website", "N/A")
    country = company_info.get("country", "N/A")

    # Ensure website has a scheme
    if website and website != "N/A" and not re.match(r'http[s]?://', website):
        website = 'https://' + website.strip()
    elif website:
        website = website.strip()


    site_status = "N/A"
    if website and website != "N/A" and 'N/A' not in website:
        site_status = scrape_website_title(website)
    
    ai_prompt = create_final_prompt(company_name, website if website else "N/A", country)
    ai_result = query_openrouter(api_key, model, ai_prompt)

    email_verification_details = extract_and_verify_emails(ai_result)

    return {
        "Company": company_name,
        "Website": website,
        "Country": country,
        "Site Status": site_status,
        "AI Result": ai_result,
        "Email Verification": email_verification_details
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
st.caption("A tool to find executive contacts using AI-driven web research. Always verify results.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_default = ""
    try:
        api_key_default = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception: # Handles cases where secrets are not set up (local dev without secrets.toml)
        pass
        
    api_key = st.text_input("Enter your OpenRouter API Key", type="password", value=api_key_default)

    st.sidebar.info(
        "ℹ️ **Tip:** For best results, select a model from the 'Web Search' or 'Top Rated' categories. "
        "These models are generally better at browsing the web for current information. "
        "AI can still make mistakes, so always cross-verify critical information and sources."
    )

    models = get_openrouter_models(api_key)
    
    if models:
        display_options = []
        option_to_model_map = [] # To map display option back to actual model ID

        # Add categories in a preferred order
        preferred_categories = ["Web Search", "Top Rated", "Free", "All Models"]
        
        temp_all_models_flat_list = [] # For finding index of session_state.selected_model

        for category in preferred_categories:
            if category in models and models[category]:
                display_options.append(f"--- {category} ---")
                option_to_model_map.append(None) # Header is not selectable
                temp_all_models_flat_list.append(None)

                for model_id in models[category]:
                    # Avoid duplicating models already shown in preferred categories
                    if category == "All Models" and any(model_id in models[cat] for cat in preferred_categories if cat != "All Models"):
                        continue
                    
                    display_options.append(model_id)
                    option_to_model_map.append(model_id)
                    temp_all_models_flat_list.append(model_id)
        
        # Determine initial index for the selectbox
        final_index = 0 # Default
        if st.session_state.selected_model:
            try:
                # Try to find the selected model in the display_options list
                final_index = display_options.index(st.session_state.selected_model)
            except ValueError:
                # If not found (e.g., model list changed), try to find a good default
                if models["Web Search"]:
                    try: final_index = display_options.index(models["Web Search"][0])
                    except ValueError: pass
                elif models["Top Rated"]:
                    try: final_index = display_options.index(models["Top Rated"][0])
                    except ValueError: pass
        elif models["Web Search"]: # Default to first web search model if nothing selected
             try: final_index = display_options.index(models["Web Search"][0])
             except ValueError: pass
        elif models["Top Rated"]: # Else default to first top rated
             try: final_index = display_options.index(models["Top Rated"][0])
             except ValueError: pass


        selected_display_option = st.selectbox(
            "Select AI Model", 
            options=display_options,
            index=final_index,
            format_func=lambda x: x if x is None or not x.startswith("---") else x.replace("---", "▼").replace("---","").strip(), # Make headers look like dropdowns
            help="Choose a model. Models under 'Web Search' can browse the web live. 'Top Rated' are generally powerful."
        )

        # Get the actual model ID from the display option
        selected_model_id = None
        if selected_display_option:
            try:
                selected_idx = display_options.index(selected_display_option)
                selected_model_id = option_to_model_map[selected_idx]
            except (ValueError, IndexError):
                pass # Should not happen if selected_display_option is from options

        if selected_model_id: # Only update if a valid model (not a header) is chosen
            st.session_state.selected_model = selected_model_id
        
    else:
        if api_key: # Only show warning if key is entered but models failed to load
            st.warning("Could not load models. Check API key or OpenRouter status.")
        else:
            st.info("Enter your OpenRouter API Key to load models.")
        st.session_state.selected_model = None


# --- Main Content Area ---
tab1, tab2 = st.tabs(["➡️ Single Search", " 批量处理 (Batch)"])

# --- Single Search Tab ---
with tab1:
    st.subheader("Find a Single Contact")
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            company_name_single = st.text_input("Company Name", placeholder="e.g., OpenAI")
        with col2:
            website_single = st.text_input("Company Website (e.g., openai.com)", placeholder="openai.com")
        country_single = st.text_input("Country (Optional)", placeholder="e.g., USA")

        submit_single = st.form_submit_button("🚀 Research Contact", use_container_width=True)

    if submit_single:
        error_messages = []
        if not company_name_single:
            error_messages.append("Company Name is required.")
        if not website_single:
            error_messages.append("Company Website is required.")
        if not api_key:
            error_messages.append("OpenRouter API Key is required in the sidebar.")
        if not st.session_state.selected_model:
            error_messages.append("Please select an AI model in the sidebar.")
        
        if error_messages:
            for error in error_messages:
                st.error(error)
        else:
            with st.spinner(f"AI is researching {company_name_single}... This may take up to 3 minutes."):
                company_info = {"company": company_name_single, "website": website_single, "country": country_single if country_single else "Unknown"}
                result = process_company(company_info, api_key, st.session_state.selected_model)
                st.session_state.processing_results = [result] # Overwrite for single search

# --- Batch Processing Tab ---
with tab2:
    st.subheader("Find Contacts in Bulk")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", help="CSV must have columns: `company`, `website`. `country` is optional.")
    st.markdown("Your CSV should have columns named `company`, `website`. The `country` column is optional (if missing, it will be treated as 'Unknown').")

    max_threads_batch = st.slider("Max Concurrent Batch Workers", min_value=1, max_value=10, value=3, help="Number of companies to process in parallel. Higher values are faster but use more resources.")


    submit_batch = st.button("🚀 Start Batch Research", use_container_width=True)

    if submit_batch:
        error_messages_batch = []
        if not uploaded_file:
            error_messages_batch.append("Please upload a CSV file.")
        if not api_key:
            error_messages_batch.append("OpenRouter API Key is required in the sidebar.")
        if not st.session_state.selected_model:
            error_messages_batch.append("Please select an AI model in the sidebar.")

        if error_messages_batch:
            for error in error_messages_batch:
                st.error(error)
        else:
            try:
                df = pd.read_csv(uploaded_file)
                # Normalize column names to lowercase and strip spaces
                df.columns = [col.lower().strip() for col in df.columns]

                required_cols = ["company", "website"]
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"CSV file is missing required columns: {', '.join(missing_cols)}.")
                else:
                    # Add country column if it doesn't exist, filling with "Unknown"
                    if 'country' not in df.columns:
                        df['country'] = "Unknown"
                    else:
                        df['country'] = df['country'].fillna("Unknown") # Fill NaN in country with "Unknown"
                    
                    # Ensure company and website have values
                    df = df.dropna(subset=['company', 'website'])
                    df = df[df['company'].astype(str).str.strip() != '']
                    df = df[df['website'].astype(str).str.strip() != '']

                    if df.empty:
                        st.warning("The CSV file is empty or has no valid rows after cleaning (missing company/website).")
                    else:
                        companies_to_process = df.to_dict('records')
                        st.session_state.processing_results = [] # Clear previous batch results
                        results_collector = [] # Use a temporary list for thread safety
                        
                        progress_bar = st.progress(0, text="Starting batch processing...")
                        total_companies = len(companies_to_process)
                        
                        with ThreadPoolExecutor(max_workers=max_threads_batch) as executor:
                            future_to_company = {
                                executor.submit(process_company, info, api_key, st.session_state.selected_model): info 
                                for info in companies_to_process
                            }
                            
                            for i, future in enumerate(as_completed(future_to_company)):
                                company_info = future_to_company[future]
                                try:
                                    result = future.result()
                                    results_collector.append(result)
                                except Exception as e:
                                    # Log error for this specific company and continue
                                    results_collector.append({
                                        "Company": company_info.get("company", "N/A from error"),
                                        "Website": company_info.get("website", "N/A from error"),
                                        "Country": company_info.get("country", "N/A from error"),
                                        "Site Status": "Error during processing",
                                        "AI Result": f"### Processing Error\nAn error occurred: {e}",
                                        "Email Verification": []
                                    })
                                progress = (i + 1) / total_companies
                                progress_bar.progress(progress, text=f"Processed {i+1}/{total_companies}: {company_info.get('company', 'N/A')}...")
                        
                        st.session_state.processing_results = results_collector
                        progress_bar.progress(1.0, text=f"Batch processing complete! Processed {len(results_collector)} companies.")
                        if len(results_collector) < total_companies:
                            st.warning(f"{total_companies - len(results_collector)} companies could not be processed due to errors.")


            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.error("Please ensure your CSV is correctly formatted with `company` and `website` columns.")

# --- Display Results ---
if st.session_state.processing_results:
    st.divider()
    st.subheader("🔍 Research Results")
    
    # Create a DataFrame for Excel export (will be modified)
    results_df_orig = pd.DataFrame(st.session_state.processing_results)
    
    if not results_df_orig.empty:
        results_df_for_excel = results_df_orig.copy()
        
        # Ensure 'Email Verification' column exists and handle cases where it might be missing or not a list
        if 'Email Verification' not in results_df_for_excel.columns:
            results_df_for_excel['Email Verification'] = pd.Series([[] for _ in range(len(results_df_for_excel))])
        else:
            results_df_for_excel['Email Verification'] = results_df_for_excel['Email Verification'].apply(lambda x: x if isinstance(x, list) else [])


        results_df_for_excel['Verified Emails Info'] = results_df_for_excel['Email Verification'].apply(
            lambda x_list: "; ".join([f"{item['email']} ({item['status']})" for item in x_list]) if x_list else "N/A"
        )
        results_df_for_excel['Valid Emails Found'] = results_df_for_excel['Email Verification'].apply(
            lambda x_list: ", ".join([item['email'] for item in x_list if item['status'] == "Valid"]) if x_list else "N/A"
        )
        
        # Select and reorder columns for Excel
        excel_cols = ["Company", "Website", "Country", "Site Status", "Valid Emails Found", "Verified Emails Info", "AI Result"]
        # Ensure all expected columns are present before trying to use them
        cols_for_excel_export = [col for col in excel_cols if col in results_df_for_excel.columns]
        excel_export_df = results_df_for_excel[cols_for_excel_export]

        excel_data = to_excel(excel_export_df)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_data,
            file_name=f"contact_research_results_{time.strftime('%Y%m%d-%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_button"
        )

        # Display individual results using the original processing_results list of dicts
        for idx, result in enumerate(st.session_state.processing_results):
            expander_title = f"**{result.get('Company', 'N/A')}**"
            if result.get('Website', 'N/A') != 'N/A':
                expander_title += f" - {result.get('Website')}"
            
            with st.expander(expander_title, expanded=(idx == 0)): # Expand first result by default
                st.markdown(f"##### AI Output for {result.get('Company', 'N/A')}:")
                ai_result_text = result.get("AI Result", "No AI result available.")
                # Try to make URLs in AI result clickable (basic attempt)
                ai_result_text_html = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank">\1</a>', ai_result_text)
                st.markdown(ai_result_text_html, unsafe_allow_html=True)
                
                st.caption(f"Site Accessibility Status: {result.get('Site Status', 'N/A')}")
                
                email_ver_details = result.get("Email Verification", [])
                if email_ver_details:
                    st.markdown("---")
                    st.markdown("##### Email Verification Status:")
                    for item in email_ver_details:
                        status_text = item.get('status', 'Unknown')
                        email_text = item.get('email', 'N/A')
                        
                        if status_text == "Valid":
                            status_color = "green"
                        elif "timeout" in status_text.lower() or "Verification error" in status_text:
                            status_color = "orange"
                        elif "Invalid (format)" in status_text: # Less severe invalid
                            status_color = "gold"
                        else: # "Invalid (domain...", "Invalid (syntax..."
                            status_color = "red"
                        
                        st.markdown(f"- `{email_text}`: <span style='color:{status_color}; font-weight:bold;'>{status_text}</span>", unsafe_allow_html=True)
                elif "API Error" not in ai_result_text and "Processing Error" not in ai_result_text:
                     st.markdown("---")
                     st.markdown("##### Email Verification Status:")
                     st.markdown("_No emails found in AI result for verification, or extraction failed._")
    else:
        st.info("Results will be displayed here once processing is complete.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit | AI by OpenRouter | Email Verification via `email-validator`")
