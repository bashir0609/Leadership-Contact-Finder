import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime
import validators
import whois
from urllib.parse import urlparse, urljoin
import time
import json
import io
from concurrent.futures import ThreadPoolExecutor
import threading
from bs4 import BeautifulSoup
import urllib.robotparser

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

class WebScraper:
    """Simple web scraper for extracting contact information from websites"""
    
    def __init__(self, company_name=""):
        self.company_name = company_name
        self.contacts_found = {
            'emails': set(),
            'phones': set(),
            'names': set(),
            'addresses': set(),
            'social_links': set(),
            'pages_scraped': [],
            'contact_pages': []
        }
        
        # Email and phone regex patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(\+?\d{1,4}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        )
        
        # Contact page keywords
        self.contact_keywords = [
            'contact', 'kontakt', 'contacts', 'contact-us', 'contact_us',
            'about', 'about-us', 'about_us', 'team', 'staff', 'people',
            'impressum', 'imprint', 'mentions-legales', 'legal',
            'leadership', 'management', 'executives', 'directors',
            'office', 'offices', 'locations', 'address', 'phone'
        ]
        
        # Session for maintaining connections
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ContactFinder/1.0 (+https://contact-finder.app)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def can_fetch(self, url):
        """Check if we can fetch the URL according to robots.txt"""
        try:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(url, '/robots.txt'))
            rp.read()
            return rp.can_fetch('*', url)
        except:
            return True  # If we can't check robots.txt, assume it's okay
    
    def fetch_page(self, url, timeout=30):
        """Fetch a single page with error handling"""
        try:
            if not self.can_fetch(url):
                return None
            
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            st.warning(f"Failed to fetch {url}: {str(e)[:100]}...")
            return None
    
    def extract_contacts_from_text(self, text, url=""):
        """Extract contact information from text"""
        # Extract emails
        emails = self.email_pattern.findall(text)
        for email in emails:
            # Filter out common non-contact emails
            if not any(skip in email.lower() for skip in 
                      ['noreply', 'no-reply', 'donotreply', 'example.com', 'test.com']):
                self.contacts_found['emails'].add(email.lower())
        
        # Extract phone numbers
        phones = self.phone_pattern.findall(text)
        for phone in phones:
            # Clean and validate phone numbers
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if len(clean_phone) >= 7:  # Minimum phone length
                self.contacts_found['phones'].add(phone.strip())
    
    def extract_names_from_soup(self, soup):
        """Extract potential contact names from BeautifulSoup object"""
        # Common name contexts
        name_contexts = [
            'team', 'staff', 'contact', 'management', 'leadership',
            'ceo', 'cto', 'cfo', 'director', 'manager', 'head'
        ]
        
        # Find elements that might contain names
        for context in name_contexts:
            elements = soup.find_all(text=re.compile(context, re.I))
            for element in elements:
                parent = element.parent if element.parent else element
                # Extract potential names from nearby text
                text = parent.get_text() if hasattr(parent, 'get_text') else str(parent)
                names = self.extract_person_names(text)
                self.contacts_found['names'].update(names)
    
    def extract_person_names(self, text):
        """Extract person names using pattern matching"""
        names = set()
        
        # Pattern for names (First Last, Dr. First Last, etc.)
        name_pattern = r'\b(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(name_pattern, text)
        
        for match in matches:
            # Filter out common non-names
            if not any(word.lower() in match.lower() for word in 
                      ['contact', 'email', 'phone', 'address', 'website', 'company']):
                if len(match.split()) >= 2:  # At least first and last name
                    names.add(match.strip())
        
        return names
    
    def extract_addresses_from_text(self, text):
        """Extract physical addresses"""
        # Address patterns (simplified)
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[\s,]+[A-Za-z\s]+,?\s*\d{5}',
            r'[A-Za-z\s]+\d+\s*,\s*\d{5}\s+[A-Za-z\s]+',  # European format
        ]
        
        for pattern in address_patterns:
            addresses = re.findall(pattern, text, re.IGNORECASE)
            for address in addresses:
                self.contacts_found['addresses'].add(address.strip())
    
    def extract_social_links(self, soup):
        """Extract social media links"""
        social_domains = [
            'linkedin.com', 'xing.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'youtube.com', 'tiktok.com'
        ]
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            for domain in social_domains:
                if domain in href:
                    self.contacts_found['social_links'].add(href)
    
    def find_contact_links(self, soup, base_url):
        """Find links that likely lead to contact pages"""
        contact_links = []
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            link_text = href.lower()
            if any(keyword in link_text for keyword in self.contact_keywords):
                absolute_url = urljoin(base_url, href)
                contact_links.append(absolute_url)
        
        return contact_links
    
    def scrape_website(self, start_url, max_pages=5):
        """Scrape website for contact information"""
        try:
            # Ensure URL has protocol
            if not start_url.startswith(('http://', 'https://')):
                start_url = 'https://' + start_url
            
            domain = urlparse(start_url).netloc
            pages_to_visit = [start_url]
            visited_pages = set()
            
            # Common contact page paths to try
            common_paths = [
                '/contact', '/contact-us', '/about', '/team', '/staff',
                '/impressum', '/kontakt', '/about-us', '/leadership'
            ]
            
            # Add common contact pages
            for path in common_paths:
                contact_url = urljoin(start_url, path)
                pages_to_visit.append(contact_url)
            
            progress_placeholder = st.empty()
            
            for i, url in enumerate(pages_to_visit[:max_pages]):
                if url in visited_pages:
                    continue
                
                progress_placeholder.info(f"🌐 Scraping page {i+1}/{min(len(pages_to_visit), max_pages)}: {url}")
                
                response = self.fetch_page(url)
                if not response:
                    continue
                
                visited_pages.add(url)
                self.contacts_found['pages_scraped'].append(url)
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                
                # Extract contacts
                self.extract_contacts_from_text(text, url)
                self.extract_names_from_soup(soup)
                self.extract_addresses_from_text(text)
                self.extract_social_links(soup)
                
                # If this looks like a contact page, mark it
                if any(keyword in url.lower() for keyword in self.contact_keywords):
                    self.contacts_found['contact_pages'].append(url)
                
                # Find more contact links (only from homepage)
                if url == start_url:
                    new_contact_links = self.find_contact_links(soup, start_url)
                    for link in new_contact_links:
                        if link not in pages_to_visit and urlparse(link).netloc == domain:
                            pages_to_visit.append(link)
                
                # Add delay to be respectful
                time.sleep(1)
            
            progress_placeholder.success("✅ Web scraping completed!")
            
            # Convert sets to lists for JSON serialization
            return {
                'emails': list(self.contacts_found['emails']),
                'phones': list(self.contacts_found['phones']),
                'names': list(self.contacts_found['names']),
                'addresses': list(self.contacts_found['addresses']),
                'social_links': list(self.contacts_found['social_links']),
                'pages_scraped': self.contacts_found['pages_scraped'],
                'contact_pages': self.contacts_found['contact_pages']
            }
            
        except Exception as e:
            st.error(f"Web scraping failed: {e}")
            return None

def get_openrouter_models(api_key):
    """Get comprehensive list of OpenRouter models with categories"""
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", 
                           headers={"Authorization": f"Bearer {api_key}"})
        models = resp.json()["data"]
        
        # Categorize models
        free_models = []
        web_search_models = []
        premium_models = []
        
        for model in models:
            model_id = model["id"]
            model_name = model.get("name", model_id)
            pricing = model.get("pricing", {})
            
            # Check if free (some models have 0 cost)
            is_free = (pricing.get("prompt", "0") == "0" and 
                      pricing.get("completion", "0") == "0")
            
            # Web search capable models
            if any(keyword in model_id.lower() for keyword in 
                   ["perplexity", "online", "web", "search", "sonar"]):
                web_search_models.append((model_id, f"{model_name} (Web Search)"))
            elif is_free:
                free_models.append((model_id, f"{model_name} (Free)"))
            else:
                premium_models.append((model_id, f"{model_name}"))
        
        return {
            "web_search": sorted(web_search_models),
            "free": sorted(free_models),
            "premium": sorted(premium_models)
        }
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return {"web_search": [], "free": [], "premium": []}

def create_comprehensive_search_prompt(company, website, country, industry=""):
    """Enhanced prompt for comprehensive contact finding"""
    domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    
    prompt = f"""
You are an expert business intelligence researcher with access to real-time web data. Your task is to find comprehensive contact information for {company} (website: {website}) in {country}.

**COMPREHENSIVE SEARCH STRATEGY - Execute ALL sources:**

1. **Official Website Deep Analysis**:
   - {website}/contact, /about, /team, /staff, /leadership
   - {website}/impressum (German sites), /mentions-legales (French)
   - Subdirectories: /en/contact, /de/kontakt
   - Extract ALL emails, names, phone numbers, addresses

2. **LinkedIn Professional Search**:
   - Current employees at "{company}"
   - Search variations: "{company} GmbH", "{company} Ltd", etc.
   - Executives: CEO, CTO, CFO, VP, Director, Manager
   - Department heads: HR, Sales, Marketing, Operations
   - Get LinkedIn profile URLs and contact information

3. **Business Directory Mining**:
   - Google Business listings
   - Yelp, Yellow Pages, local directories
   - Industry-specific directories
   - Chamber of Commerce listings
   - Better Business Bureau (US)
   - Companies House (UK), Bundesanzeiger (Germany)

4. **Professional Networks**:
   - Xing.com (German-speaking countries)
   - AngelList (startups)
   - Crunchbase (funding/executive info)
   - Industry association member directories

5. **News & Press Coverage**:
   - Recent press releases mentioning executives
   - Industry publications and interviews
   - Conference speaker lists
   - Award announcements
   - Local news mentions

6. **Technical Sources**:
   - WHOIS domain registration data
   - SSL certificate contact info
   - DNS records and mail server information
   - Cached versions (Wayback Machine)

7. **Social Media & Web Presence**:
   - Company Facebook, Twitter, Instagram pages
   - YouTube channel information
   - Medium, blog author information
   - Podcast guest appearances

8. **Government & Legal Sources**:
   - Corporate registration databases
   - SEC filings (public companies)
   - Patent applications
   - Trademark registrations
   - Court filings

**OUTPUT FORMAT** (Markdown Table):

| Name | Role/Title | LinkedIn/Profile URL | Email | Phone | Source | Confidence | Notes |
|------|------------|---------------------|--------|-------|--------|------------|-------|
| [Name or "General Contact"] | [Exact Title] | [Full URL] | [Email] | [Phone] | [Source] | [High/Med/Low] | [Additional info] |

**EMAIL PATTERNS TO CHECK**:
- info@{domain}, contact@{domain}, hello@{domain}
- sales@, support@, office@, admin@
- firstname@, f.lastname@, firstname.lastname@
- Common patterns for this country/industry

**CONFIDENCE LEVELS**:
- **High**: Verified from official sources (website, LinkedIn, press)
- **Medium**: Business directories, news articles
- **Low**: Pattern-based estimates, unverified sources

**CRITICAL**: Search EVERY source mentioned above. Don't skip any category.

Begin comprehensive research for {company} now. Be thorough and systematic.
"""
    
    return prompt

def query_openrouter_enhanced(api_key, model, prompt, timeout=120):
    """Enhanced API query with better error handling and longer timeout"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-contact-finder.streamlit.app",
        "X-Title": "AI Contact Finder"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9
    }
    
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 15
                st.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif response.status_code == 402:
                st.error("Insufficient credits. Please check your OpenRouter account.")
                return None
            else:
                st.error(f"API error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.warning(f"Request timeout on attempt {attempt + 1}/3")
            if attempt < 2:
                time.sleep(10)
        except Exception as e:
            if attempt == 2:
                st.error(f"Failed after 3 attempts: {str(e)}")
                return None
            time.sleep(5)
    
    return None

def get_whois_contacts(domain):
    """Extract comprehensive contacts from WHOIS data"""
    try:
        w = whois.whois(domain)
        contacts = {
            'domain': domain,
            'registrar': getattr(w, 'registrar', None),
            'creation_date': getattr(w, 'creation_date', None),
            'expiration_date': getattr(w, 'expiration_date', None),
            'name_servers': getattr(w, 'name_servers', None),
            'emails': [],
            'org': getattr(w, 'org', None),
            'country': getattr(w, 'country', None)
        }
        
        # Extract all emails
        if hasattr(w, 'emails') and w.emails:
            if isinstance(w.emails, list):
                contacts['emails'] = list(set(w.emails))  # Remove duplicates
            else:
                contacts['emails'] = [w.emails]
        
        # Get additional fields
        for field in ['admin_email', 'tech_email', 'billing_email']:
            if hasattr(w, field):
                email = getattr(w, field)
                if email and email not in contacts['emails']:
                    contacts['emails'].append(email)
            
        return contacts
    except Exception as e:
        st.warning(f"WHOIS lookup failed for {domain}: {e}")
        return None

def process_single_company(api_key, model, company, website, country, industry="", search_methods=None):
    """Process a single company and return results"""
    try:
        # Validate website
        if not website.startswith(('http://', 'https://')):
            website = 'https://' + website
        
        if not validators.url(website):
            return {
                'company': company,
                'website': website,
                'error': 'Invalid website URL'
            }
        
        domain = urlparse(website).netloc.replace('www.', '')
        
        results = {
            'company': company,
            'website': website,
            'domain': domain,
            'country': country,
            'industry': industry,
            'whois_data': None,
            'ai_research': None,
            'web_scraping_data': None,
            'processed_at': datetime.now()
        }
        
        # WHOIS lookup
        if "WHOIS Lookup" in search_methods:
            results['whois_data'] = get_whois_contacts(domain)
        
        # Web scraping
        if "Website Scraping" in search_methods:
            scraper = WebScraper(company)
            results['web_scraping_data'] = scraper.scrape_website(website)
        
        # AI research
        if "AI Research" in search_methods:
            prompt = create_comprehensive_search_prompt(company, website, country, industry)
            results['ai_research'] = query_openrouter_enhanced(api_key, model, prompt)
        
        return results
        
    except Exception as e:
        return {
            'company': company,
            'website': website,
            'error': str(e)
        }

def display_web_scraping_results(scraping_data):
    """Display web scraping results"""
    if not scraping_data:
        st.warning("No web scraping data available")
        return
    
    with st.expander("🌐 Website Scraping Results", expanded=True):
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Emails Found", len(scraping_data.get('emails', [])))
        with col2:
            st.metric("Phone Numbers", len(scraping_data.get('phones', [])))
        with col3:
            st.metric("Names Extracted", len(scraping_data.get('names', [])))
        with col4:
            st.metric("Pages Scraped", len(scraping_data.get('pages_scraped', [])))
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📧 Email Addresses")
            emails = scraping_data.get('emails', [])
            if emails:
                for email in emails:
                    st.code(email)
            else:
                st.info("No emails found")
            
            st.subheader("📞 Phone Numbers")
            phones = scraping_data.get('phones', [])
            if phones:
                for phone in phones:
                    st.code(phone)
            else:
                st.info("No phone numbers found")
        
        with col2:
            st.subheader("👥 Names Extracted")
            names = scraping_data.get('names', [])
            if names:
                for name in names:
                    st.code(name)
            else:
                st.info("No names found")
            
            st.subheader("🏢 Addresses")
            addresses = scraping_data.get('addresses', [])
            if addresses:
                for address in addresses:
                    st.code(address)
            else:
                st.info("No addresses found")
        
        # Social links
        if scraping_data.get('social_links'):
            st.subheader("🔗 Social Media Links")
            for link in scraping_data.get('social_links', []):
                st.markdown(f"- [{link}]({link})")
        
        # Pages scraped
        if scraping_data.get('pages_scraped'):
            st.subheader("📄 Pages Scraped")
            for page in scraping_data.get('pages_scraped', []):
                st.markdown(f"- [{page}]({page})")

def parse_ai_results_to_dataframe(ai_result):
    """Parse AI research results into a structured dataframe"""
    if not ai_result:
        return None
    
    try:
        lines = ai_result.split("\n")
        table_lines = [line for line in lines if "|" in line and "---" not in line and "Name" in line or 
                      ("|" in line and "---" not in line and len([cell for cell in line.split("|") if cell.strip()]) >= 4)]
        
        if len(table_lines) >= 2:
            # Extract headers
            headers = [cell.strip() for cell in table_lines[0].split("|") if cell.strip()]
            
            # Extract data rows
            rows = []
            for line in table_lines[1:]:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                if len(cells) >= 3:  # Minimum viable row
                    # Pad with empty strings if needed
                    while len(cells) < len(headers):
                        cells.append("")
                    rows.append(cells[:len(headers)])
            
            if rows:
                df = pd.DataFrame(rows, columns=headers)
                return df
    except Exception as e:
        st.warning(f"Could not parse table format: {e}")
    
    return None

def main():
    st.set_page_config(
        page_title="AI-Powered Contact Finder",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI-Powered Contact Finder")
    st.markdown("*Web scraping + AI research + WHOIS lookup + Professional networks + Batch processing*")
    st.markdown("**✅ Streamlit Cloud Compatible Version**")
    
    # Load API key
    api_key = st.secrets.get("OPENROUTER_API_KEY") or st.text_input(
        "🔐 OpenRouter API Key", 
        type="password",
        help="Get your API key from https://openrouter.ai/"
    )
    
    if not api_key:
        st.error("API key required to proceed")
        st.info("💡 You can either add OPENROUTER_API_KEY to Streamlit secrets or enter it above")
        st.stop()
    
    # Get models with error handling
    with st.spinner("Loading available models..."):
        models_dict = get_openrouter_models(api_key)
    
    if not any(models_dict.values()):
        st.error("Failed to load models. Please check your API key.")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection with persistence
        st.subheader("🤖 AI Model Selection")
        
        model_category = st.selectbox(
            "Model Category",
            ["Web Search Models (Recommended)", "Free Models", "Premium Models"],
            help="Web Search models can access real-time internet data"
        )
        
        if model_category == "Web Search Models (Recommended)":
            available_models = models_dict["web_search"]
        elif model_category == "Free Models":
            available_models = models_dict["free"]
        else:
            available_models = models_dict["premium"]
        
        if available_models:
            model_options = [f"{name}" for _, name in available_models]
            model_ids = [model_id for model_id, _ in available_models]
            
            # Use session state to persist selection
            if st.session_state.selected_model and st.session_state.selected_model in model_ids:
                default_idx = model_ids.index(st.session_state.selected_model)
            else:
                default_idx = 0
            
            selected_idx = st.selectbox(
                "Select Model",
                range(len(model_options)),
                index=default_idx,
                format_func=lambda x: model_options[x]
            )
            
            selected_model = model_ids[selected_idx]
            st.session_state.selected_model = selected_model
        else:
            st.error(f"No models available in {model_category}")
            st.stop()
        
        # Search settings
        st.subheader("🔍 Search Methods")
        
        search_methods = st.multiselect(
            "Active Search Methods",
            ["Website Scraping", "AI Research", "WHOIS Lookup"],
            default=["Website Scraping", "AI Research", "WHOIS Lookup"],
            help="All methods work in Streamlit Cloud"
        )
        
        # Processing mode
        st.subheader("📊 Processing Mode")
        processing_mode = st.radio(
            "Choose Mode",
            ["Single Company", "Batch CSV Processing"]
        )
        
        st.markdown("---")
        st.markdown("**💡 Pro Tips:**")
        st.caption("• Website scraping extracts real contact data")
        st.caption("• Use Web Search models for comprehensive AI research")
        st.caption("• Combine all methods for maximum coverage")
        st.caption("• ✅ All features work in Streamlit Cloud")
    
    # Main content area
    if processing_mode == "Single Company":
        # Single company processing
        st.subheader("🏢 Single Company Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.text_input("Company Name", 
                                  placeholder="e.g., BBW Berufsbildungswerk Hamburg")
            website = st.text_input("Website URL", 
                                  placeholder="e.g., bbw.de or https://bbw.de")
        
        with col2:
            country = st.text_input("Country", value="Germany")
            industry = st.text_input("Industry (Optional)", 
                                   placeholder="e.g., Education, Technology")
        
        if st.button("🚀 Start Multi-Method Search", type="primary"):
            if not all([company, website, country]):
                st.error("Please fill in company name, website, and country")
                return
            
            if not search_methods:
                st.error("Please select at least one search method")
                return
            
            # Process single company
            with st.spinner("Conducting comprehensive multi-method research..."):
                result = process_single_company(
                    api_key, selected_model, company, website, country, industry, search_methods
                )
            
            # Display results
            display_single_result(result, search_methods)
    
    else:
        # Batch CSV processing
        st.subheader("📊 Batch CSV Processing")
        
        # CSV template download
        template_df = pd.DataFrame({
            'company': ['Example Corp', 'Another Company'],
            'website': ['example.com', 'anothercompany.com'],
            'country': ['Germany', 'USA'],
            'industry': ['Technology', 'Manufacturing']
        })
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV Template",
            data=csv_template,
            file_name="ai_contact_finder_template.csv",
            mime="text/csv"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with companies",
            type=['csv'],
            help="CSV should have columns: company, website, country, industry (optional)"
        )
        
        if uploaded_file:
            try:
                companies_df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['company', 'website', 'country']
                missing_cols = [col for col in required_cols if col not in companies_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    return
                
                st.success(f"Loaded {len(companies_df)} companies")
                st.dataframe(companies_df.head())
                
                if st.button("🚀 Process All Companies", type="primary"):
                    process_batch_csv(companies_df, api_key, selected_model, search_methods)
                    
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

def display_single_result(result, search_methods):
    """Display results for a single company"""
    if 'error' in result:
        st.error(f"Error processing {result.get('company', 'company')}: {result['error']}")
        return
    
    st.success(f"✅ Research completed for {result['company']}")
    
    # Web Scraping Results
    if "Website Scraping" in search_methods and result.get('web_scraping_data'):
        display_web_scraping_results(result['web_scraping_data'])
    
    # WHOIS Results
    if "WHOIS Lookup" in search_methods and result.get('whois_data'):
        with st.expander("📋 WHOIS Domain Information", expanded=True):
            whois_data = result['whois_data']
            
            col1, col2 = st.columns(2)
            with col1:
                if whois_data.get('org'):
                    st.info(f"**Registered Organization**: {whois_data['org']}")
                if whois_data.get('registrar'):
                    st.info(f"**Registrar**: {whois_data['registrar']}")
                if whois_data.get('country'):
                    st.info(f"**Country**: {whois_data['country']}")
            
            with col2:
                if whois_data.get('emails'):
                    st.info(f"**Contact Emails**: {', '.join(whois_data['emails'])}")
                if whois_data.get('creation_date'):
                    st.info(f"**Domain Created**: {whois_data['creation_date']}")
    
    # AI Research Results
    if "AI Research" in search_methods and result.get('ai_research'):
        st.subheader("🧠 AI Research Results")
        
        with st.expander("📄 Full Research Report", expanded=True):
            st.markdown(result['ai_research'])
        
        # Parse and display structured data
        df = parse_ai_results_to_dataframe(result['ai_research'])
        if df is not None:
            st.subheader("📊 Structured Contact Data")
            st.dataframe(df, use_container_width=True)
            
            # Export options
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            csv_data = df.to_csv(index=False).encode("utf-8")
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_data = excel_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"{result['company'].lower().replace(' ', '_')}_contacts_{timestamp}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"{result['company'].lower().replace(' ', '_')}_contacts_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

def process_batch_companies(api_key, model, companies_df, search_methods, progress_callback=None):
    """Process multiple companies with progress tracking"""
    results = []
    total = len(companies_df)
    
    for idx, row in companies_df.iterrows():
        if progress_callback:
            progress_callback(idx + 1, total, f"Processing {row.get('company', 'Unknown')}")
        
        result = process_single_company(
            api_key, model,
            row.get('company', ''),
            row.get('website', ''),
            row.get('country', ''),
            row.get('industry', ''),
            search_methods
        )
        results.append(result)
        
        # Add delay to avoid rate limiting
        time.sleep(2)
    
    return results

def process_batch_csv(companies_df, api_key, selected_model, search_methods):
    """Process batch CSV with progress tracking"""
    st.subheader("🔄 Batch Processing Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, company_name):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Processing {current}/{total}: {company_name}")
    
    # Process all companies
    results = process_batch_companies(
        api_key, selected_model, companies_df, search_methods, update_progress
    )
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.text("✅ Batch processing completed!")
    
    # Display results summary
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Processed", len(results))
    with col2:
        st.metric("Successful", len(successful_results))
    with col3:
        st.metric("Failed", len(failed_results))
    
    # Show failed results
    if failed_results:
        with st.expander("❌ Failed Processes"):
            for result in failed_results:
                st.error(f"{result.get('company', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    # Combine all successful results into downloadable format
    if successful_results:
        all_contacts = []
        
        for result in successful_results:
            # Combine Web Scraping and AI results
            company_contacts = []
            
            # Add Web Scraping results
            if result.get('web_scraping_data'):
                scraping_data = result['web_scraping_data']
                for email in scraping_data.get('emails', []):
                    company_contacts.append({
                        'Name': '',
                        'Role': '',
                        'Email': email,
                        'Phone': '',
                        'Source': 'Website Scraping',
                        'Confidence': 'High',
                        'Company': result['company'],
                        'Website': result['website'],
                        'Country': result['country']
                    })
                for phone in scraping_data.get('phones', []):
                    company_contacts.append({
                        'Name': '',
                        'Role': '',
                        'Email': '',
                        'Phone': phone,
                        'Source': 'Website Scraping',
                        'Confidence': 'High',
                        'Company': result['company'],
                        'Website': result['website'],
                        'Country': result['country']
                    })
            
            # Add AI research results
            if result.get('ai_research'):
                df = parse_ai_results_to_dataframe(result['ai_research'])
                if df is not None:
                    for _, row in df.iterrows():
                        company_contacts.append({
                            'Name': row.get('Name', ''),
                            'Role': row.get('Role/Title', row.get('Role', '')),
                            'Email': row.get('Email', ''),
                            'Phone': row.get('Phone', ''),
                            'Source': row.get('Source', 'AI Research'),
                            'Confidence': row.get('Confidence', 'Medium'),
                            'Company': result['company'],
                            'Website': result['website'],
                            'Country': result['country']
                        })
            
            if company_contacts:
                all_contacts.extend(company_contacts)
        
        if all_contacts:
            combined_df = pd.DataFrame(all_contacts)
            
            st.subheader("📊 Combined Results")
            st.dataframe(combined_df, use_container_width=True)
            
            # Export combined results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            csv_data = combined_df.to_csv(index=False).encode("utf-8")
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='All Contacts', index=False)
                
                # Create summary sheet
                summary_df = pd.DataFrame({
                    'Company': [r['company'] for r in successful_results],
                    'Website': [r['website'] for r in successful_results],
                    'Country': [r['country'] for r in successful_results],
                    'Scraping Emails': [len(r.get('web_scraping_data', {}).get('emails', [])) for r in successful_results],
                    'Scraping Phones': [len(r.get('web_scraping_data', {}).get('phones', [])) for r in successful_results],
                    'AI Research': ['Yes' if r.get('ai_research') else 'No' for r in successful_results],
                    'Processed At': [r['processed_at'].strftime("%Y-%m-%d %H:%M") for r in successful_results]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            excel_data = excel_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download All Contacts (CSV)",
                    data=csv_data,
                    file_name=f"ai_batch_contacts_{timestamp}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download All Contacts (Excel)",
                    data=excel_data,
                    file_name=f"ai_batch_contacts_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # Tips section
    with st.expander("💡 Tips & Best Practices"):
        st.markdown("""
        **🌐 Website Scraping:**
        - Extracts contact information directly from website pages
        - Respects robots.txt and follows ethical scraping guidelines
        - Searches contact pages, about pages, team directories
        - Extracts emails, phone numbers, names, and addresses
        
        **🧠 AI Research:**
        - Searches LinkedIn, business directories, news sources
        - Provides context and recent information
        - Best with Web Search models for real-time data
        - Cross-references multiple online sources
        
        **📋 WHOIS Lookup:**
        - Provides domain registration information
        - Technical and administrative contacts
        - Organization details and registration dates
        
        **✅ Streamlit Cloud Compatible:**
        - All features work reliably in Streamlit Cloud
        - No dependency conflicts or reactor issues
        - Scalable for batch processing
        - Secure API key handling through Streamlit secrets
        """)

if __name__ == "__main__":
    main()
